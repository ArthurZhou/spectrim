use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::audio::decode::decode_file;
use crate::audio::export::export_wav;
use crate::audio::preview::{PreviewCache, PreviewOutcome};
use crate::audio::process::apply_selections;
use crate::audio::spectrogram;
use crate::audio::types::{AudioData, Selection, SelectionAction, Spectrogram};
use crate::playback::PlaybackHandle;
use crate::ui::spectrogram_view::SpectrogramView;

/// Debounce for re-rendering the preview while the user drags a region.
///
/// Edits arrive every UI frame (~60 Hz) during a drag. Rendering the full track
/// per frame would swamp the worker, so we wait for the pointer to settle
/// instead, then do one incremental render.
const PREVIEW_DEBOUNCE: Duration = Duration::from_millis(60);

/// The decoded track plus its analysis.
///
/// `audio` is an `Arc` so handing the worker thread a copy for preview/export
/// rendering is a refcount bump, not a clone of the sample buffer. The original
/// code cloned the whole `Vec<f32>` on every play, edit and export.
struct LoadedAudio {
    path: PathBuf,
    audio: Arc<AudioData>,
    /// The dry interleaved samples, shared with the mixer for unprocessed
    /// playback. Built once at load so starting playback is a refcount bump;
    /// the mixer only ever reads it.
    dry_samples: Arc<Vec<f32>>,
    spectrogram: Spectrogram,
    texture: egui::TextureHandle,
}

enum LoadMessage {
    Progress(String),
    Loaded(Result<(PathBuf, Arc<AudioData>, Spectrogram, egui::ColorImage), String>),
}

/// Work request for the background preview/export worker.
enum WorkerJob {
    /// Re-render the (fast, IIR-based) preview from the current selections.
    Preview {
        audio: Arc<AudioData>,
        selections: Vec<Selection>,
    },
    /// Exact STFT render for writing a WAV file.
    Export {
        audio: Arc<AudioData>,
        selections: Vec<Selection>,
        path: PathBuf,
    },
}

enum WorkerMessage {
    /// Preview render finished; carries the interleaved samples plus which frame
    /// span was recomputed, so playback can resume without a position reset.
    PreviewDone {
        samples: Arc<Vec<f32>>,
        changed_frames: Option<(usize, usize)>,
        full_rebuild: bool,
        spectrogram: Spectrogram,
    },
    ExportDone(Result<(), String>),
}

pub struct SpectrimApp {
    loaded: Option<LoadedAudio>,
    status: String,
    load_receiver: Option<Receiver<LoadMessage>>,
    load_started: Option<Instant>,
    /// Sender end of the persistent worker; `None` when no worker is running.
    worker_sender: Option<mpsc::Sender<WorkerJob>>,
    worker_receiver: Option<Receiver<WorkerMessage>>,
    worker_busy: bool,
    /// True while an export is in flight (preview results must not disturb it).
    export_in_flight: bool,
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
    /// When the last edit happened, for debouncing.
    last_edit: Option<Instant>,
    selections: Vec<Selection>,
    selected_selection: Option<u64>,
    next_selection_id: u64,
    default_action: SelectionAction,
    spectrogram_view: SpectrogramView,
}

impl Default for SpectrimApp {
    fn default() -> Self {
        Self {
            loaded: None,
            status: "Open a WAV, MP3, FLAC, or Ogg audio file to begin.".to_owned(),
            load_receiver: None,
            load_started: None,
            worker_sender: None,
            worker_receiver: None,
            worker_busy: false,
            export_in_flight: false,
            playback: PlaybackHandle::spawn(),
            is_playing: false,
            paused: false,
            plot_drag_active: false,
            playhead_time: 0.0,
            processed_mode: true,
            preview_samples: None,
            preview_cache: PreviewCache::new(),
            preview_dirty: false,
            last_edit: None,
            selections: Vec::new(),
            selected_selection: None,
            next_selection_id: 1,
            default_action: SelectionAction::Mute,
            spectrogram_view: SpectrogramView::default(),
        }
    }
}

impl SpectrimApp {
    fn start_load(&mut self, path: PathBuf) {
        self.playback.stop();
        self.is_playing = false;
        self.paused = false;
        self.plot_drag_active = false;
        self.playhead_time = 0.0;
        self.preview_samples = None;
        self.preview_cache = PreviewCache::new();
        self.stop_worker();

        let (sender, receiver) = mpsc::channel();
        self.load_receiver = Some(receiver);
        self.load_started = Some(Instant::now());
        self.status = format!("Decoding {}…", path.display());
        thread::spawn(move || {
            let started = Instant::now();
            let result = decode_file(&path)
                .map_err(|error| error.to_string())
                .and_then(|audio| {
                    let decode_time = started.elapsed();
                    let _ = sender.send(LoadMessage::Progress(format!(
                        "Decoded in {:.1}s; computing spectrogram…",
                        decode_time.as_secs_f32()
                    )));
                    let audio = Arc::new(audio);
                    let analysis_started = Instant::now();
                    let spec = spectrogram::compute(&audio, 2048, 512);
                    let analysis_time = analysis_started.elapsed();
                    println!(
                        "Spectrogram: {} × {} in {:.2}s",
                        spec.n_freq,
                        spec.n_time,
                        analysis_time.as_secs_f32()
                    );
                    let _ = sender.send(LoadMessage::Progress(format!(
                        "Spectrogram ready in {:.1}s; preparing display…",
                        analysis_time.as_secs_f32()
                    )));
                    let image_started = Instant::now();
                    let image = crate::ui::spectrogram_view::color_image(&spec);
                    println!(
                        "Display image prepared in {:.2}s",
                        image_started.elapsed().as_secs_f32()
                    );
                    Ok((path, audio, spec, image))
                });
            let _ = sender.send(LoadMessage::Loaded(result));
        });
    }

    /// Start the persistent preview/export worker.
    fn start_worker(&mut self) {
        if self.worker_sender.is_some() {
            return;
        }
        let (job_tx, job_rx) = mpsc::channel::<WorkerJob>();
        let (msg_tx, msg_rx) = mpsc::channel::<WorkerMessage>();
        let mut cache = PreviewCache::new();
        self.worker_sender = Some(job_tx);
        self.worker_receiver = Some(msg_rx);
        // Move the existing cache into the worker so a reload is not needed.
        std::mem::swap(&mut cache, &mut self.preview_cache);
        thread::spawn(move || {
            let mut cache = cache;
            while let Ok(job) = job_rx.recv() {
                match job {
                    WorkerJob::Preview { audio, selections } => {
                        let outcome: PreviewOutcome = cache.render(&audio, &selections);
                        let samples = Arc::new(cache.samples.clone());
                        let spectrogram = spectrogram::compute_interleaved(
                            &samples,
                            audio.channels,
                            audio.sample_rate,
                            2048,
                            512,
                        );
                        let _ = msg_tx.send(WorkerMessage::PreviewDone {
                            samples,
                            changed_frames: outcome.changed_frames,
                            full_rebuild: outcome.full_rebuild,
                            spectrogram,
                        });
                    }
                    WorkerJob::Export {
                        audio,
                        selections,
                        path,
                    } => {
                        // Export stays exact: full STFT masking, not the IIR preview.
                        let samples = apply_selections(&audio, &selections);
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
        self.worker_receiver = None;
        self.worker_busy = false;
        self.export_in_flight = false;
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
        self.preview_dirty = true;
        let _ = sender.send(WorkerJob::Preview { audio, selections });
    }

    fn poll_workers(&mut self, ctx: &egui::Context) {
        let load_messages = self
            .load_receiver
            .as_ref()
            .map(|receiver| receiver.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        for message in load_messages {
            match message {
                LoadMessage::Progress(message) => self.status = message,
                LoadMessage::Loaded(result) => {
                    self.load_receiver = None;
                    self.load_started = None;
                    match result {
                        Ok((path, audio, spec, image)) => {
                            let texture = ctx.load_texture(
                                "spectrogram",
                                image,
                                egui::TextureOptions::LINEAR,
                            );
                            let dry_samples = Arc::new(audio.samples.clone());
                            self.loaded = Some(LoadedAudio {
                                path,
                                audio,
                                dry_samples,
                                spectrogram: spec,
                                texture,
                            });
                            self.playhead_time = 0.0;
                            self.spectrogram_view = SpectrogramView::default();
                            self.selections.clear();
                            self.preview_samples = None;
                            self.preview_cache = PreviewCache::new();
                            self.status = "Audio loaded successfully.".to_owned();
                        }
                        Err(error) => self.status = error,
                    }
                }
            }
        }

        let message = self
            .worker_receiver
            .as_ref()
            .and_then(|receiver| receiver.try_recv().ok());
        match message {
            Some(WorkerMessage::PreviewDone {
                samples,
                changed_frames,
                full_rebuild,
                spectrogram: edited_spectrogram,
            }) => {
                self.preview_dirty = false;
                self.preview_samples = Some(Arc::clone(&samples));
                if let Some(loaded) = &mut self.loaded {
                    let image = crate::ui::spectrogram_view::color_image(&edited_spectrogram);
                    loaded.texture.set(image, egui::TextureOptions::LINEAR);
                    loaded.spectrogram = edited_spectrogram;
                }

                // Playback continuity: if we are playing in edited mode, hand the
                // new buffer to the mixer *without* restarting it, so the
                // position is preserved. The device keeps its absolute frame
                // index; only the samples under it change.
                if self.processed_mode && self.is_playing && !self.paused {
                    self.playback.update_buffer(Arc::clone(&samples));
                    if full_rebuild || changed_frames.is_some() {
                        self.status = "Preview updated (playing position preserved).".to_owned();
                    }
                } else {
                    self.status = "Preview updated.".to_owned();
                }
            }
            Some(WorkerMessage::ExportDone(result)) => {
                self.export_in_flight = false;
                self.worker_busy = false;
                self.status = match result {
                    Ok(()) => "WAV export completed.".to_owned(),
                    Err(error) => error,
                };
            }
            None => {}
        }

        if self.load_receiver.is_some() || self.worker_receiver.is_some() {
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

    fn export(&mut self, path: PathBuf) {
        let Some(audio) = self.loaded.as_ref().map(|loaded| Arc::clone(&loaded.audio)) else {
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
        let _ = sender.send(WorkerJob::Export {
            audio,
            selections,
            path,
        });
    }

    /// An edit happened. Unlike the original, this does **not** stop playback and
    /// does **not** reset the playhead; it schedules a debounced incremental
    /// preview render and lets the mixer swap buffers underneath.
    fn selections_changed(&mut self) {
        self.last_edit = Some(Instant::now());
        self.preview_dirty = true;
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

        let processing = self.export_in_flight || self.preview_dirty;
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
                ui.selectable_value(&mut self.default_action, SelectionAction::Mute, "Mute");
                ui.selectable_value(&mut self.default_action, SelectionAction::Solo, "Solo");
                ui.selectable_value(&mut self.default_action, SelectionAction::Gain, "Gain");
                if ui
                    .add_enabled(
                        self.loaded.is_some() && !self.export_in_flight,
                        egui::Button::new("Export WAV…"),
                    )
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("WAV", &["wav"])
                        .set_file_name("edited.wav")
                        .save_file()
                    {
                        self.export(path);
                    }
                }
            });
        });

        egui::SidePanel::right("regions")
            .default_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                let events =
                    crate::ui::region_list::show(ui, &mut self.selections, self.selected_selection);
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
                });
                let events = self.spectrogram_view.show(
                    ui,
                    Some(&loaded.texture),
                    &loaded.spectrogram,
                    &loaded.audio,
                    &mut self.selections,
                    self.selected_selection,
                    current_time,
                );
                if let Some(selected) = events.selected {
                    self.selected_selection = selected;
                }
                if events.selections_changed {
                    selections_modified = true;
                }
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

        egui::TopBottomPanel::bottom("status_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(started) = self.load_started {
                    ui.label(format!(
                        "{} ({:.1}s)",
                        self.status,
                        started.elapsed().as_secs_f32()
                    ));
                } else {
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
