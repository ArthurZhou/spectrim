use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::audio::export::export_wav;
use crate::audio::preview::{PreviewCache, PreviewOutcome};
use crate::audio::process::apply_selections;
use crate::audio::spectrogram::{self, SpectrogramPatch};
use crate::audio::types::{AudioData, Selection, SelectionAction, Spectrogram};
use crate::playback::PlaybackHandle;
use crate::ui::spectrogram_view::SpectrogramView;

/// Debounce for re-rendering the preview while the user drags a region.
///
/// Edits arrive every UI frame (~60 Hz) during a drag. Rendering the full track
/// per frame would swamp the worker, so we wait for the pointer to settle
/// instead, then do one incremental render.
const PREVIEW_DEBOUNCE: Duration = Duration::from_millis(60);
const MAX_SPECTROGRAM_CELLS: usize = 8 * 1024 * 1024;
const MAX_DETAIL_SPECTROGRAM_CELLS: usize = 4 * 1024 * 1024;
const MAX_DETAIL_VIEW_SECONDS: f32 = 60.0;
type LoadedPayload = (
    PathBuf,
    Arc<AudioData>,
    Arc<AudioData>,
    Spectrogram,
    egui::ColorImage,
);

fn spectrogram_hop(frames: usize, n_fft: usize) -> usize {
    spectrogram_hop_with_budget(frames, n_fft, MAX_SPECTROGRAM_CELLS)
}

fn spectrogram_hop_with_budget(frames: usize, n_fft: usize, cell_budget: usize) -> usize {
    let n_freq = n_fft / 2 + 1;
    let minimum_hop = frames
        .saturating_mul(n_freq)
        .div_ceil(cell_budget.max(1))
        .max(512);
    minimum_hop.div_ceil(512).saturating_mul(512)
}

#[cfg(test)]
mod display_fft_tests {
    use super::*;

    #[test]
    fn detail_fft_hop_keeps_long_view_within_cell_budget() {
        let frames = 44_100 * 60;
        let hop = spectrogram_hop_with_budget(frames, 4096, MAX_DETAIL_SPECTROGRAM_CELLS);
        let columns = (frames - 4096) / hop + 1;
        assert!(columns * (4096 / 2 + 1) <= MAX_DETAIL_SPECTROGRAM_CELLS);
        assert_eq!(hop % 512, 0);
    }
}

/// The decoded track plus its analysis.
///
/// `audio` is an `Arc` so handing the worker thread a copy for preview/export
/// rendering is a refcount bump, not a clone of the sample buffer. The original
/// code cloned the whole `Vec<f32>` on every play, edit and export.
struct LoadedAudio {
    path: PathBuf,
    /// Original decoded samples and format, used only by exact export.
    source_audio: Arc<AudioData>,
    /// 44.1 kHz editor copy used by playback, preview, and FFT.
    audio: Arc<AudioData>,
    /// The dry interleaved samples, shared with the mixer for unprocessed
    /// playback. Built once at load so starting playback is a refcount bump;
    /// the mixer only ever reads it.
    dry_samples: Arc<Vec<f32>>,
    spectrogram: Spectrogram,
    texture: egui::TextureHandle,
}

enum LoadMessage {
    Progress(u64, String),
    Loaded(u64, Result<LoadedPayload, String>),
}

/// Work request for the background preview/export worker.
enum WorkerJob {
    /// Re-render the (fast, IIR-based) preview from the current selections.
    Preview {
        audio: Arc<AudioData>,
        selections: Vec<Selection>,
        spectrogram_n_fft: usize,
        spectrogram_hop: usize,
        revision: u64,
    },
    /// Exact STFT render for writing a WAV file.
    Export {
        audio: Arc<AudioData>,
        selections: Vec<Selection>,
        path: PathBuf,
        crop_silence: bool,
    },
}

/// FFT refresh jobs use a separate worker so the latency-sensitive IIR preview
/// thread can immediately process the next edit instead of waiting on FFT.
enum SpectrogramJob {
    PatchFromPcm {
        samples: Arc<Vec<f32>>,
        channels: u16,
        sample_rate: u32,
        n_fft: usize,
        hop: usize,
        changed_frames: (usize, usize),
        revision: u64,
    },
    FullRebuild {
        samples: Arc<Vec<f32>>,
        channels: u16,
        sample_rate: u32,
        n_fft: usize,
        hop: usize,
        revision: u64,
    },
    ViewDetail {
        samples: Arc<Vec<f32>>,
        channels: u16,
        sample_rate: u32,
        time_range: (f32, f32),
        n_fft: usize,
        revision: u64,
    },
}

enum WorkerMessage {
    /// Preview render finished; carries the interleaved samples plus which frame
    /// span was recomputed, so playback can resume without a position reset.
    PreviewDone {
        samples: Arc<Vec<f32>>,
        changed_frames: Option<(usize, usize)>,
        revision: u64,
    },
    SpectrogramDone(u64, SpectrogramPatch),
    SpectrogramRebuilt(u64, Spectrogram),
    ViewDetailReady(u64, f32, f32, Spectrogram),
    ExportDone(Result<(), String>),
}

pub struct SpectrimApp {
    loaded: Option<LoadedAudio>,
    status: String,
    status_is_error: bool,
    load_receiver: Option<Receiver<LoadMessage>>,
    load_cancel: Option<Arc<AtomicBool>>,
    load_cancel_requested: bool,
    load_revision: u64,
    load_started: Option<Instant>,
    /// Sender end of the persistent worker; `None` when no worker is running.
    worker_sender: Option<mpsc::Sender<WorkerJob>>,
    spectrogram_sender: Option<mpsc::Sender<SpectrogramJob>>,
    worker_receiver: Option<Receiver<WorkerMessage>>,
    worker_busy: bool,
    pending_spectrogram_revisions: HashSet<u64>,
    completed_spectrogram_revisions: HashSet<u64>,
    preview_revision: u64,
    /// True while an export is in flight (preview results must not disturb it).
    export_in_flight: bool,
    preview_in_flight: bool,
    playback: PlaybackHandle,
    is_playing: bool,
    paused: bool,
    plot_drag_active: bool,
    /// Logical play position when not playing, or the position playback started
    /// from when playing (the device counter is added on top).
    playhead_time: f32,
    processed_mode: bool,
    /// Rendered preview samples, shared with the playback thread.
    preview_samples: Option<Arc<Vec<f32>>>,
    /// Cache living on the worker thread; the UI only keeps the last result.
    preview_cache: PreviewCache,
    /// Set when a preview render has been requested but not yet dispatched.
    preview_dirty: bool,
    fft_size: usize,
    fft_zoom_target: Option<(usize, (f32, f32), Instant)>,
    detail_revision: u64,
    detail_in_flight: bool,
    /// When the last edit happened, for debouncing.
    last_edit: Option<Instant>,
    selections: Vec<Selection>,
    selected_selection: Option<u64>,
    next_selection_id: u64,
    default_action: SelectionAction,
    polygon_tool: bool,
    spectrogram_view: SpectrogramView,
    pointer_coordinates: Option<(f32, f32)>,
    visible_time_range: (f32, f32),
    export_dialog_open: bool,
    export_crop_silence: bool,
}

impl Default for SpectrimApp {
    fn default() -> Self {
        Self {
            loaded: None,
            status: "Open a WAV, MP3, FLAC, or Ogg audio file to begin.".to_owned(),
            status_is_error: false,
            load_receiver: None,
            load_cancel: None,
            load_cancel_requested: false,
            load_revision: 0,
            load_started: None,
            worker_sender: None,
            spectrogram_sender: None,
            worker_receiver: None,
            worker_busy: false,
            pending_spectrogram_revisions: HashSet::new(),
            completed_spectrogram_revisions: HashSet::new(),
            preview_revision: 0,
            export_in_flight: false,
            preview_in_flight: false,
            playback: PlaybackHandle::spawn(),
            is_playing: false,
            paused: false,
            plot_drag_active: false,
            playhead_time: 0.0,
            processed_mode: true,
            preview_samples: None,
            preview_cache: PreviewCache::new(),
            preview_dirty: false,
            fft_size: 2048,
            fft_zoom_target: None,
            detail_revision: 0,
            detail_in_flight: false,
            last_edit: None,
            selections: Vec::new(),
            selected_selection: None,
            next_selection_id: 1,
            default_action: SelectionAction::Gain,
            polygon_tool: false,
            spectrogram_view: SpectrogramView::default(),
            pointer_coordinates: None,
            visible_time_range: (0.0, 0.0),
            export_dialog_open: false,
            export_crop_silence: false,
        }
    }
}

impl SpectrimApp {
    fn start_load(&mut self, path: PathBuf) {
        if let Some(cancel) = self.load_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.playback.stop();
        self.is_playing = false;
        self.paused = false;
        self.plot_drag_active = false;
        self.playhead_time = 0.0;
        self.preview_samples = None;
        self.preview_cache = PreviewCache::new();
        self.stop_worker();

        let (sender, receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.load_cancel = Some(Arc::clone(&cancel));
        self.load_cancel_requested = false;
        self.load_revision = self.load_revision.wrapping_add(1);
        let revision = self.load_revision;
        self.load_receiver = Some(receiver);
        self.load_started = Some(Instant::now());
        self.status = format!("Decoding {}…", path.display());
        self.status_is_error = false;
        thread::spawn(move || {
            let started = Instant::now();
            let result = crate::audio::decode::decode_file_cancellable(&path, &cancel)
                .map_err(|error| error.to_string())
                .and_then(|audio| {
                    if cancel.load(Ordering::Relaxed) {
                        return Err("Loading cancelled".to_owned());
                    }
                    let decode_time = started.elapsed();
                    let _ = sender.send(LoadMessage::Progress(
                        revision,
                        format!(
                            "Decoded in {:.1}s; computing spectrogram…",
                            decode_time.as_secs_f32()
                        ),
                    ));
                    let source_audio = Arc::new(audio.source);
                    let audio = Arc::new(audio.editor);
                    let analysis_started = Instant::now();
                    let hop = spectrogram_hop(audio.frame_count(), 2048);
                    let Some(spec) = spectrogram::compute_cancellable(&audio, 2048, hop, &cancel)
                    else {
                        return Err("Loading cancelled".to_owned());
                    };
                    let analysis_time = analysis_started.elapsed();
                    println!(
                        "Spectrogram: {} × {} in {:.2}s",
                        spec.n_freq,
                        spec.n_time,
                        analysis_time.as_secs_f32()
                    );
                    let _ = sender.send(LoadMessage::Progress(
                        revision,
                        format!(
                            "Spectrogram ready in {:.1}s; preparing display…",
                            analysis_time.as_secs_f32()
                        ),
                    ));
                    let image_started = Instant::now();
                    let Some(image) =
                        crate::ui::spectrogram_view::color_image_cancellable(&spec, &cancel)
                    else {
                        return Err("Loading cancelled".to_owned());
                    };
                    if cancel.load(Ordering::Relaxed) {
                        return Err("Loading cancelled".to_owned());
                    }
                    println!(
                        "Display image prepared in {:.2}s",
                        image_started.elapsed().as_secs_f32()
                    );
                    Ok((path, source_audio, audio, spec, image))
                });
            let _ = sender.send(LoadMessage::Loaded(revision, result));
        });
    }

    /// Start the persistent preview/export worker.
    fn start_worker(&mut self) {
        if self.worker_sender.is_some() {
            return;
        }
        let (job_tx, job_rx) = mpsc::channel::<WorkerJob>();
        let (spectrogram_tx, spectrogram_rx) = mpsc::channel::<SpectrogramJob>();
        let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
        let mut cache = PreviewCache::new();
        self.worker_sender = Some(job_tx);
        self.spectrogram_sender = Some(spectrogram_tx.clone());
        self.worker_receiver = Some(msg_rx);
        // Move the existing cache into the worker so a reload is not needed.
        std::mem::swap(&mut cache, &mut self.preview_cache);
        let spectrogram_msg_tx = msg_tx.clone();
        thread::spawn(move || {
            while let Ok(job) = spectrogram_rx.recv() {
                match job {
                    SpectrogramJob::PatchFromPcm {
                        samples,
                        channels,
                        sample_rate,
                        n_fft,
                        hop,
                        changed_frames,
                        revision,
                    } => {
                        let patch = spectrogram::compute_interleaved_patch(
                            &samples,
                            channels,
                            sample_rate,
                            n_fft,
                            hop,
                            changed_frames.0,
                            changed_frames.1,
                        );
                        let _ = spectrogram_msg_tx
                            .send(WorkerMessage::SpectrogramDone(revision, patch));
                    }
                    SpectrogramJob::FullRebuild {
                        samples,
                        channels,
                        sample_rate,
                        n_fft,
                        hop,
                        revision,
                    } => {
                        let full = spectrogram::compute_interleaved(
                            &samples,
                            channels,
                            sample_rate,
                            n_fft,
                            hop,
                        );
                        let _ = spectrogram_msg_tx
                            .send(WorkerMessage::SpectrogramRebuilt(revision, full));
                    }
                    SpectrogramJob::ViewDetail {
                        samples,
                        channels,
                        sample_rate,
                        time_range,
                        n_fft,
                        revision,
                    } => {
                        let channel_count = channels.max(1) as usize;
                        let total_frames = samples.len() / channel_count;
                        let padding = n_fft / 2;
                        let first = ((time_range.0.max(0.0) * sample_rate as f32) as usize)
                            .saturating_sub(padding);
                        let last = ((time_range.1.max(0.0) * sample_rate as f32).ceil() as usize)
                            .saturating_add(padding)
                            .min(total_frames);
                        let hop = spectrogram_hop_with_budget(
                            last.saturating_sub(first),
                            n_fft,
                            MAX_DETAIL_SPECTROGRAM_CELLS,
                        );
                        let last = last.saturating_add(hop).min(total_frames);
                        let crop = &samples[first * channel_count..last * channel_count];
                        let detail = spectrogram::compute_interleaved(
                            crop,
                            channels,
                            sample_rate,
                            n_fft,
                            hop,
                        );
                        let detail_end =
                            (first + detail.n_time.saturating_sub(1) * hop + n_fft / 2).min(last);
                        let _ = spectrogram_msg_tx.send(WorkerMessage::ViewDetailReady(
                            revision,
                            (first + n_fft / 2) as f32 / sample_rate as f32,
                            detail_end as f32 / sample_rate as f32,
                            detail,
                        ));
                    }
                }
            }
        });
        thread::spawn(move || {
            let mut cache = cache;
            while let Ok(job) = job_rx.recv() {
                match job {
                    WorkerJob::Preview {
                        audio,
                        selections,
                        spectrogram_n_fft,
                        spectrogram_hop,
                        revision,
                    } => {
                        let outcome: PreviewOutcome = cache.render(&audio, &selections);
                        // Publish audible IIR output first. FFT work is queued
                        // to its own worker and cannot delay preview/playback.
                        let _ = msg_tx.send(WorkerMessage::PreviewDone {
                            samples: Arc::new(cache.samples.clone()),
                            changed_frames: outcome.changed_frames,
                            revision,
                        });
                        if let Some(changed_frames) = outcome.changed_frames {
                            let total_frames = audio.frame_count();
                            let n_fft = spectrogram_n_fft;
                            let hop = spectrogram_hop;
                            if outcome.full_rebuild {
                                let _ = spectrogram_tx.send(SpectrogramJob::FullRebuild {
                                    samples: Arc::new(cache.samples.clone()),
                                    channels: audio.channels,
                                    sample_rate: audio.sample_rate,
                                    n_fft,
                                    hop,
                                    revision,
                                });
                            } else {
                                let n_total_time = if total_frames >= n_fft {
                                    (total_frames - n_fft) / hop + 1
                                } else {
                                    0
                                };
                                let changed_frames = spectrogram::expanded_patch_frames(
                                    changed_frames,
                                    n_total_time,
                                    total_frames,
                                    n_fft,
                                    hop,
                                    8192,
                                );
                                let _ = spectrogram_tx.send(SpectrogramJob::PatchFromPcm {
                                    samples: Arc::new(cache.samples.clone()),
                                    channels: audio.channels,
                                    sample_rate: audio.sample_rate,
                                    n_fft,
                                    hop,
                                    changed_frames,
                                    revision,
                                });
                            }
                        }
                    }
                    WorkerJob::Export {
                        audio,
                        selections,
                        path,
                        crop_silence,
                    } => {
                        // Export stays exact: full STFT masking, not the IIR preview.
                        let mut samples = apply_selections(&audio, &selections);
                        if crop_silence {
                            samples = crate::audio::process::trim_silence(
                                &samples,
                                audio.channels,
                                0.001,
                            );
                        }
                        let result = export_wav(&path, &samples, audio.sample_rate, audio.channels)
                            .map_err(|error| format!("WAV export failed: {error}"));
                        let _ = msg_tx.send(WorkerMessage::ExportDone(result));
                    }
                }
            }
        });
    }

    fn stop_worker(&mut self) {
        self.worker_sender = None;
        self.spectrogram_sender = None;
        self.worker_receiver = None;
        self.worker_busy = false;
        self.pending_spectrogram_revisions.clear();
        self.completed_spectrogram_revisions.clear();
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.export_in_flight = false;
        self.preview_in_flight = false;
    }

    /// Ask the worker to re-render the preview from the current selections.
    ///
    /// Called after any edit; the actual render happens on the worker thread and
    /// the result is applied in `poll_workers`, which is what keeps the
    /// playhead from resetting.
    fn request_preview(&mut self) {
        // Take the Arc first and end the borrow, otherwise `start_worker` cannot
        // borrow `self` mutably.
        let Some(audio) = self.loaded.as_ref().map(|loaded| Arc::clone(&loaded.audio)) else {
            return;
        };
        self.start_worker();
        let Some(sender) = self.worker_sender.clone() else {
            return;
        };
        let selections = self.selections.clone();
        let (spectrogram_n_fft, spectrogram_hop) = self
            .loaded
            .as_ref()
            .map(|loaded| (loaded.spectrogram.n_fft, loaded.spectrogram.hop_length))
            .unwrap_or((2048, 512));
        self.preview_revision = self.preview_revision.wrapping_add(1);
        let revision = self.preview_revision;
        self.preview_dirty = true;
        self.preview_in_flight = true;
        if sender
            .send(WorkerJob::Preview {
                audio,
                selections,
                spectrogram_n_fft,
                spectrogram_hop,
                revision,
            })
            .is_err()
        {
            self.preview_in_flight = false;
            self.status = "Preview worker stopped unexpectedly.".to_owned();
            self.status_is_error = true;
        }
    }

    fn poll_workers(&mut self, ctx: &egui::Context) {
        let load_messages = self
            .load_receiver
            .as_ref()
            .map(|receiver| receiver.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        for message in load_messages {
            match message {
                LoadMessage::Progress(revision, message)
                    if revision == self.load_revision && !self.load_cancel_requested =>
                {
                    self.status = message
                }
                LoadMessage::Progress(_, _) => {}
                LoadMessage::Loaded(revision, result) if revision == self.load_revision => {
                    self.load_receiver = None;
                    self.load_started = None;
                    self.load_cancel = None;
                    self.load_cancel_requested = false;
                    match result {
                        Ok((path, source_audio, audio, spec, image)) => {
                            let texture = ctx.load_texture(
                                "spectrogram",
                                image,
                                egui::TextureOptions::LINEAR,
                            );
                            let dry_samples = Arc::new(audio.samples.clone());
                            self.loaded = Some(LoadedAudio {
                                path,
                                source_audio,
                                audio,
                                dry_samples,
                                spectrogram: spec,
                                texture,
                            });
                            self.fft_size = 2048;
                            self.fft_zoom_target = None;
                            self.playhead_time = 0.0;
                            self.spectrogram_view = SpectrogramView::default();
                            self.selections.clear();
                            self.preview_samples = None;
                            self.preview_cache = PreviewCache::new();
                            self.status.clear();
                        }
                        Err(error) => {
                            if error == "Loading cancelled" {
                                self.status.clear();
                                self.status_is_error = false;
                            } else {
                                self.status = error;
                                self.status_is_error = true;
                            }
                        }
                    }
                }
                LoadMessage::Loaded(_, _) => {}
            }
        }

        let worker_messages = self
            .worker_receiver
            .as_ref()
            .map(|receiver| receiver.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        for message in worker_messages {
            match message {
                WorkerMessage::PreviewDone {
                    samples,
                    changed_frames,
                    revision,
                } => {
                    if revision == self.preview_revision {
                        self.preview_in_flight = false;
                    }
                    if revision == self.preview_revision {
                        self.preview_dirty = false;
                        self.preview_samples = Some(Arc::clone(&samples));
                        let fft_already_done =
                            self.completed_spectrogram_revisions.remove(&revision);
                        if changed_frames.is_some() && !fft_already_done {
                            self.pending_spectrogram_revisions.insert(revision);
                        }

                        // Playback continuity: replace the buffer without restarting
                        // the mixer, preserving its absolute frame position.
                        if self.processed_mode && self.is_playing && !self.paused {
                            self.playback.update_buffer(Arc::clone(&samples));
                        }
                        if changed_frames.is_some() && !fft_already_done {
                            self.status = "Updating edited preview and FFT…".to_owned();
                        } else if !self.export_in_flight {
                            self.status.clear();
                        }
                    }
                }
                WorkerMessage::SpectrogramDone(revision, patch) => {
                    if !self.pending_spectrogram_revisions.remove(&revision)
                        && revision == self.preview_revision
                    {
                        self.completed_spectrogram_revisions.insert(revision);
                    }
                    if revision == self.preview_revision {
                        if let Some(loaded) = &mut self.loaded {
                            if patch.n_freq == loaded.spectrogram.n_freq
                                && patch.first_time_frame + patch.n_time
                                    <= loaded.spectrogram.n_time
                            {
                                for frequency in 0..patch.n_freq {
                                    let src = &patch.matrix
                                        [frequency * patch.n_time..(frequency + 1) * patch.n_time];
                                    let start = frequency * loaded.spectrogram.n_time
                                        + patch.first_time_frame;
                                    loaded.spectrogram.matrix[start..start + patch.n_time]
                                        .copy_from_slice(src);
                                }
                                // Recompute the global percentile range after edits.
                                // Otherwise edited columns use an obsolete range and
                                // their apparent intensity drifts until manual FFT.
                                let (db_min, db_max) =
                                    spectrogram::compute_db_range(&loaded.spectrogram.matrix);
                                loaded.spectrogram.db_min = db_min;
                                loaded.spectrogram.db_max = db_max;
                                let image =
                                    crate::ui::spectrogram_view::color_image(&loaded.spectrogram);
                                loaded.texture.set(image, egui::TextureOptions::LINEAR);
                            }
                        }
                    }
                    if self.pending_spectrogram_revisions.is_empty()
                        && !self.preview_in_flight
                        && self.status.starts_with("Updating edited preview")
                    {
                        self.status.clear();
                    }
                }
                WorkerMessage::SpectrogramRebuilt(revision, rebuilt) => {
                    if !self.pending_spectrogram_revisions.remove(&revision)
                        && revision == self.preview_revision
                    {
                        self.completed_spectrogram_revisions.insert(revision);
                    }
                    if revision == self.preview_revision {
                        if let Some(loaded) = &mut self.loaded {
                            let image = crate::ui::spectrogram_view::color_image(&rebuilt);
                            loaded.texture.set(image, egui::TextureOptions::LINEAR);
                            loaded.spectrogram = rebuilt;
                        }
                        if self.pending_spectrogram_revisions.is_empty()
                            && !self.preview_in_flight
                            && (self.status.starts_with("Recalculating FFT")
                                || self.status.starts_with("Updating edited preview"))
                        {
                            self.status.clear();
                        }
                    }
                }
                WorkerMessage::ViewDetailReady(revision, time_start, time_end, detail) => {
                    if revision == self.detail_revision {
                        self.detail_in_flight = false;
                        if let Some(loaded) = &self.loaded {
                            let image = crate::ui::spectrogram_view::color_image_with_range(
                                &detail,
                                loaded.spectrogram.db_min,
                                loaded.spectrogram.db_max,
                                loaded.spectrogram.n_fft,
                            );
                            let texture = ctx.load_texture(
                                "spectrogram-detail",
                                image,
                                egui::TextureOptions::LINEAR,
                            );
                            self.spectrogram_view
                                .set_detail(texture, time_start, time_end);
                        }
                    }
                }
                WorkerMessage::ExportDone(result) => {
                    self.export_in_flight = false;
                    self.worker_busy = false;
                    self.status = match result {
                        Ok(()) => String::new(),
                        Err(error) => error,
                    };
                    self.status_is_error = !self.status.is_empty();
                }
            }
        }

        if self.load_receiver.is_some()
            || self.preview_dirty
            || self.preview_in_flight
            || self.export_in_flight
            || !self.pending_spectrogram_revisions.is_empty()
        {
            ctx.request_repaint_after(Duration::from_millis(30));
        }
        if self.load_started.is_some() {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }

    /// The buffer the mixer should be playing right now.
    fn active_samples(&self) -> Option<Arc<Vec<f32>>> {
        if self.processed_mode {
            self.preview_samples.clone()
        } else {
            // Share the decoded buffer directly; the mixer only reads it.
            self.loaded
                .as_ref()
                .map(|loaded| Arc::clone(&loaded.dry_samples))
        }
    }

    fn play(&mut self) {
        let Some(loaded) = &self.loaded else {
            return;
        };

        // In edited mode we need a preview render before we can start. If the
        // selections have never been rendered, kick one off and wait for it;
        // otherwise play immediately from the cached buffer.
        if self.processed_mode && self.preview_samples.is_none() {
            self.request_preview();
            self.status = "Rendering preview…".to_owned();
            return;
        }

        let Some(samples) = self.active_samples() else {
            return;
        };
        self.playback.play_from(
            samples,
            loaded.audio.sample_rate,
            loaded.audio.channels,
            self.playhead_time,
        );
        self.is_playing = true;
        self.paused = false;
    }

    fn pause(&mut self) {
        // Latch the device position into `playhead_time` *before* pausing so
        // `current_time()` stays continuous across the transition (it reads the
        // device while playing and the latch while stopped).
        if self.is_playing {
            self.playhead_time = self.playback.position();
        }
        self.playback.pause();
        self.is_playing = false;
        self.paused = true;
    }

    /// Seek while (possibly) playing: move the device position, do not rebuild
    /// the sink, do not touch the preview buffer.
    fn seek(&mut self, time: f32) {
        let Some(loaded) = &self.loaded else {
            self.playhead_time = time;
            return;
        };
        let clamped = time.clamp(0.0, loaded.audio.duration);
        self.playhead_time = clamped;
        if self.is_playing {
            // Move the device's frame index in place: no sink rebuild, no buffer
            // re-feed, so scrubbing the notch while playing stays smooth and the
            // edited audio underneath is unaffected.
            self.playback.seek(clamped, loaded.audio.sample_rate);
        }
    }

    /// Current playhead time in seconds.
    ///
    /// While playing, the device's absolute frame counter *is* the position —
    /// the mixer is fed the whole track, so `playback.position()` is already
    /// track-relative. Adding `playhead_time` on top (as the pre-rewrite code
    /// did, back when position was an elapsed-since-start delta) would
    /// double-count and run the playhead at up to 2× speed.
    /// `playhead_time` is the remembered position used when playback is not
    /// running: after a pause, a seek while stopped, or a natural end.
    fn current_time(&self) -> f32 {
        if self.is_playing {
            self.playback.position()
        } else {
            self.playhead_time
        }
    }

    fn export(&mut self, path: PathBuf, crop_silence: bool) {
        let Some(audio) = self
            .loaded
            .as_ref()
            .map(|loaded| Arc::clone(&loaded.source_audio))
        else {
            return;
        };
        self.start_worker();
        let Some(sender) = self.worker_sender.clone() else {
            return;
        };
        let selections = self.selections.clone();
        self.export_in_flight = true;
        self.worker_busy = true;
        self.status = "Processing and exporting WAV…".to_owned();
        self.status_is_error = false;
        let _ = sender.send(WorkerJob::Export {
            audio,
            selections,
            path,
            crop_silence,
        });
    }

    fn request_full_fft(&mut self) {
        let Some((channels, sample_rate, dry)) = self.loaded.as_ref().map(|loaded| {
            (
                loaded.audio.channels,
                loaded.audio.sample_rate,
                Arc::clone(&loaded.dry_samples),
            )
        }) else {
            return;
        };
        let samples = self.preview_samples.clone().unwrap_or(dry);
        let n_fft = self.fft_size;
        // Bound high-resolution matrix allocations on long recordings. This
        // changes only display analysis density, never source PCM or exports.
        let frames = samples.len() / (channels.max(1) as usize);
        let hop = spectrogram_hop(frames, n_fft);
        self.start_worker();
        let Some(sender) = self.spectrogram_sender.clone() else {
            return;
        };
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.pending_spectrogram_revisions.clear();
        self.completed_spectrogram_revisions.clear();
        let revision = self.preview_revision;
        self.pending_spectrogram_revisions.insert(revision);
        self.status = format!("Recalculating FFT ({n_fft}, hop {hop})…");
        self.status_is_error = false;
        if sender
            .send(SpectrogramJob::FullRebuild {
                samples,
                channels,
                sample_rate,
                n_fft,
                hop,
                revision,
            })
            .is_err()
        {
            self.pending_spectrogram_revisions.remove(&revision);
            self.status = "FFT worker stopped unexpectedly.".to_owned();
            self.status_is_error = true;
        }
    }

    fn request_view_detail(&mut self, n_fft: usize, time_range: (f32, f32)) {
        let Some((channels, sample_rate, dry)) = self.loaded.as_ref().map(|loaded| {
            (
                loaded.audio.channels,
                loaded.audio.sample_rate,
                Arc::clone(&loaded.dry_samples),
            )
        }) else {
            return;
        };
        let samples = self.preview_samples.clone().unwrap_or(dry);
        self.start_worker();
        let Some(sender) = self.spectrogram_sender.clone() else {
            return;
        };
        self.detail_revision = self.detail_revision.wrapping_add(1);
        let revision = self.detail_revision;
        self.detail_in_flight = true;
        self.spectrogram_view.clear_detail();
        self.status = "Computing zoomed FFT detail…".to_owned();
        self.status_is_error = false;
        if sender
            .send(SpectrogramJob::ViewDetail {
                samples,
                channels,
                sample_rate,
                time_range,
                n_fft,
                revision,
            })
            .is_err()
        {
            self.detail_in_flight = false;
            self.status = "FFT worker stopped unexpectedly.".to_owned();
            self.status_is_error = true;
        }
    }

    /// An edit happened. Unlike the original, this does **not** stop playback and
    /// does **not** reset the playhead; it schedules a debounced incremental
    /// preview render and lets the mixer swap buffers underneath.
    fn selections_changed(&mut self) {
        // Invalidate in-flight preview/FFT work immediately, including during
        // the debounce window before the replacement preview is dispatched.
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.pending_spectrogram_revisions.clear();
        self.completed_spectrogram_revisions.clear();
        if self.status.starts_with("Recalculating FFT")
            || self.status.starts_with("Preview")
            || self.status == "Computing zoomed FFT detail…"
        {
            self.status.clear();
            self.status_is_error = false;
        }
        self.last_edit = Some(Instant::now());
        self.preview_dirty = true;
        self.detail_revision = self.detail_revision.wrapping_add(1);
        self.detail_in_flight = false;
        self.spectrogram_view.clear_detail();
        let refine_visible = self.loaded.as_ref().is_some_and(|loaded| {
            let time_span = self.visible_time_range.1 - self.visible_time_range.0;
            time_span > 0.0
                && time_span <= MAX_DETAIL_VIEW_SECONDS
                && time_span < loaded.audio.duration * 0.75
        });
        if refine_visible {
            self.fft_zoom_target = Some((4096, self.visible_time_range, Instant::now()));
        }
    }

    /// Flush a pending debounced preview request.
    fn flush_preview_debounce(&mut self, ctx: &egui::Context) {
        if !self.preview_dirty {
            return;
        }
        // An export owns the worker's attention; defer preview work until it is
        // done so the two jobs cannot interleave on the same cache.
        if self.export_in_flight {
            ctx.request_repaint_after(PREVIEW_DEBOUNCE);
            return;
        }
        let Some(last) = self.last_edit else {
            return;
        };
        if last.elapsed() < PREVIEW_DEBOUNCE {
            ctx.request_repaint_after(PREVIEW_DEBOUNCE - last.elapsed());
            return;
        }
        self.preview_dirty = false;
        self.last_edit = None;
        self.request_preview();
    }
}

impl eframe::App for SpectrimApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_workers(ctx);
        self.flush_preview_debounce(ctx);

        let processing = self.export_in_flight
            || self.load_receiver.is_some()
            || self.preview_dirty
            || self.preview_in_flight
            || self.detail_in_flight
            || self.fft_zoom_target.is_some()
            || !self.pending_spectrogram_revisions.is_empty();
        let duration = self
            .loaded
            .as_ref()
            .map_or(0.0, |loaded| loaded.audio.duration);

        // Natural end of track: the mixer source ran out and the audio thread
        // dropped the sink. Ask the device rather than comparing times, so a
        // seek to near-the-end cannot be mistaken for a finished track.
        if self.is_playing && !self.playback.is_active() {
            self.playhead_time = if duration > 0.0 {
                duration
            } else {
                self.playback.position()
            };
            self.is_playing = false;
            self.paused = false;
        }
        let current_time = self.current_time();

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Audio", &["wav", "mp3", "flac", "ogg"])
                        .pick_file()
                    {
                        self.start_load(path);
                    }
                }
                let events = crate::ui::transport::show(
                    ui,
                    self.is_playing,
                    current_time,
                    duration,
                    &mut self.processed_mode,
                    self.loaded.is_some() && !self.export_in_flight,
                );
                if events.play {
                    self.paused = false;
                    self.play();
                }
                if events.pause {
                    self.pause();
                }
                ui.separator();
                ui.label("New selection:");
                ui.selectable_value(&mut self.default_action, SelectionAction::Gain, "Gain");
                ui.selectable_value(&mut self.default_action, SelectionAction::Solo, "Solo")
                    .on_hover_text("Solo regions are combined as a union; while any Solo exists, Gain regions are ignored.");
                if ui
                    .selectable_label(self.polygon_tool, "Polygon tool")
                    .on_hover_text("Click to add vertices. Close by clicking the first point, double-clicking, or pressing Enter. Esc cancels. Drag vertices later to edit.")
                    .clicked()
                {
                    self.polygon_tool = !self.polygon_tool;
                    self.spectrogram_view.set_polygon_tool(self.polygon_tool);
                }
                if ui
                    .add_enabled(
                        self.loaded.is_some()
                            && !self.preview_dirty
                            && !self.preview_in_flight
                            && !self.export_in_flight
                            && !self.detail_in_flight
                            && self.fft_zoom_target.is_none()
                            && self.pending_spectrogram_revisions.is_empty(),
                        egui::Button::new("Recalculate FFT"),
                    )
                    .clicked()
                {
                    self.request_full_fft();
                }
                if ui
                    .add_enabled(
                        self.loaded.is_some() && !self.export_in_flight,
                        egui::Button::new("Export WAV…"),
                    )
                    .clicked()
                {
                    self.export_crop_silence = false;
                    self.export_dialog_open = true;
                }
            });
        });

        egui::SidePanel::right("regions")
            .default_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                let nyquist = self
                    .loaded
                    .as_ref()
                    .map_or(22_050.0, |loaded| loaded.audio.sample_rate as f32 * 0.5);
                let events = crate::ui::region_list::show(
                    ui,
                    &mut self.selections,
                    self.selected_selection,
                    nyquist,
                );
                if let Some(selected) = events.selected {
                    self.selected_selection = selected;
                }
                if events.changed {
                    self.selections_changed();
                }
            });

        let mut selections_modified = false;
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(loaded) = &self.loaded {
                ui.horizontal(|ui| {
                    ui.label(
                        loaded
                            .path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("Audio"),
                    );
                    ui.label(format!(
                        "{:.2}s · {} Hz · {} channel(s)",
                        loaded.audio.duration, loaded.audio.sample_rate, loaded.audio.channels
                    ));
                    if let Some((time, frequency)) = self.pointer_coordinates {
                        ui.label(format!("· {time:.3}s · {frequency:.1} Hz"));
                    } else {
                        ui.label("· —");
                    }
                });
                let events = self.spectrogram_view.show(
                    ui,
                    Some(&loaded.texture),
                    &loaded.spectrogram,
                    &loaded.audio,
                    &mut self.selections,
                    self.selected_selection,
                    current_time,
                    self.polygon_tool,
                );
                let visible_frequency_span = events.visible_ranges.3 - events.visible_ranges.2;
                self.visible_time_range = (events.visible_ranges.0, events.visible_ranges.1);
                if events.view_changed {
                    self.detail_revision = self.detail_revision.wrapping_add(1);
                    self.detail_in_flight = false;
                    self.spectrogram_view.clear_detail();
                    let visible_time_span = events.visible_ranges.1 - events.visible_ranges.0;
                    let should_refine = visible_time_span <= MAX_DETAIL_VIEW_SECONDS
                        && (visible_time_span < loaded.audio.duration * 0.75
                            || visible_frequency_span < loaded.audio.sample_rate as f32 * 0.4);
                    self.fft_zoom_target = should_refine.then_some((
                        4096,
                        (events.visible_ranges.0, events.visible_ranges.1),
                        Instant::now(),
                    ));
                }
                if let Some(selected) = events.selected {
                    self.selected_selection = selected;
                }
                if events.selections_changed {
                    selections_modified = true;
                }
                self.pointer_coordinates = events.pointer_coordinates;
                if events.drag_started {
                    self.plot_drag_active = true;
                    if self.is_playing {
                        self.playback.set_muted(true);
                    }
                }
                if events.drag_stopped {
                    self.plot_drag_active = false;
                    if self.is_playing {
                        self.playback.set_muted(false);
                    }
                }
                if let Some(time) = events.seek_to {
                    self.seek(time);
                }
                if let Some((time_start, time_end, freq_low, freq_high, full_band)) = events.created
                {
                    self.selections.push(Selection::new(
                        self.next_selection_id,
                        time_start,
                        time_end,
                        freq_low,
                        freq_high,
                        self.default_action,
                        full_band,
                    ));
                    self.selected_selection = Some(self.next_selection_id);
                    self.next_selection_id += 1;
                    selections_modified = true;
                }
                if let Some(points) = events.created_polygon {
                    self.selections.push(Selection::polygon(
                        self.next_selection_id,
                        points,
                        self.default_action,
                    ));
                    self.selected_selection = Some(self.next_selection_id);
                    self.next_selection_id += 1;
                    selections_modified = true;
                }
                if let Some(id) = events.deleted {
                    self.selections.retain(|selection| selection.id != id);
                    if self.selected_selection == Some(id) {
                        self.selected_selection = None;
                    }
                    selections_modified = true;
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an audio file to display its spectrogram.");
                });
            }
        });
        if selections_modified {
            self.selections_changed();
        }

        if self.export_dialog_open {
            let modal = egui::Modal::new(egui::Id::new("export_options_modal")).show(ctx, |ui| {
                ui.set_min_width(300.0);
                ui.heading("Export WAV");
                ui.checkbox(&mut self.export_crop_silence, "裁剪处理后音频的首尾静音");
                ui.label("静音阈值：-60 dBFS。其他导出选项可在此继续添加。");
                ui.horizontal(|ui| {
                    let confirm = ui.button("选择路径并导出…").clicked();
                    let cancel = ui.button("取消").clicked();
                    (confirm, cancel)
                })
                .inner
            });
            let (confirm, cancel) = modal.inner;
            if cancel || modal.should_close() {
                self.export_dialog_open = false;
            } else if confirm {
                self.export_dialog_open = false;
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV", &["wav"])
                    .set_file_name("edited.wav")
                    .save_file()
                {
                    self.export(path, self.export_crop_silence);
                }
            }
        }

        if let Some((target_fft, time_range, since)) = self.fft_zoom_target {
            if since.elapsed() >= Duration::from_millis(400)
                && !self.preview_dirty
                && !self.preview_in_flight
                && self.pending_spectrogram_revisions.is_empty()
                && !self.export_in_flight
                && self.loaded.is_some()
            {
                self.fft_size = target_fft;
                self.fft_zoom_target = None;
                self.request_view_detail(target_fft, time_range);
            } else {
                ctx.request_repaint_after(Duration::from_millis(50));
            }
        }

        egui::TopBottomPanel::bottom("status_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.load_receiver.is_some() {
                    if let Some(started) = self.load_started {
                        ui.label(format!(
                            "{} ({:.1}s)",
                            self.status,
                            started.elapsed().as_secs_f32()
                        ));
                    } else if !self.status.is_empty() {
                        ui.label(&self.status);
                    }
                    if ui.button("Cancel load").clicked() {
                        if let Some(cancel) = &self.load_cancel {
                            cancel.store(true, Ordering::Relaxed);
                        }
                        self.load_cancel_requested = true;
                        self.status = "Cancelling load…".to_owned();
                    }
                } else if processing {
                    if self.fft_zoom_target.is_some() {
                        ui.label("Adjusting FFT resolution for zoom…");
                    } else if !self.status.is_empty() {
                        ui.label(&self.status);
                    }
                } else if self.status_is_error {
                    ui.label(&self.status);
                }
                if processing {
                    ui.spinner();
                }
            });
        });

        if self.is_playing || self.preview_dirty {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        if self.export_in_flight
            || self.load_receiver.is_some()
            || self.preview_in_flight
            || self.detail_in_flight
            || self.fft_zoom_target.is_some()
            || !self.pending_spectrogram_revisions.is_empty()
        {
            ctx.request_repaint_after(Duration::from_millis(30));
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
            if self.is_playing {
                self.pause();
            } else {
                self.paused = false;
                self.play();
            }
        }
        if self.plot_drag_active
            && !ctx.input(|input| input.pointer.button_down(egui::PointerButton::Primary))
        {
            self.plot_drag_active = false;
            if self.is_playing {
                self.playback.set_muted(false);
            }
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Delete)) {
            if let Some(id) = self.selected_selection.take() {
                self.selections.retain(|selection| selection.id != id);
                self.selections_changed();
            }
        }
    }
}
