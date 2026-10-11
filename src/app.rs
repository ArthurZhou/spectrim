use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use eframe::egui;

use crate::audio::decode::decode_file;
use crate::audio::export::export_wav;
use crate::audio::process::apply_selections;
use crate::audio::spectrogram;
use crate::audio::types::{AudioData, Selection, SelectionAction, Spectrogram};
use crate::playback::PlaybackHandle;
use crate::ui::spectrogram_view::SpectrogramView;

struct LoadedAudio {
    path: PathBuf,
    audio: AudioData,
    spectrogram: Spectrogram,
    texture: egui::TextureHandle,
}

enum LoadMessage {
    Loaded(Result<(PathBuf, AudioData, Spectrogram, egui::ColorImage), String>),
}

pub struct SpectrimApp {
    loaded: Option<LoadedAudio>,
    status: String,
    load_receiver: Option<Receiver<LoadMessage>>,
    processing_receiver: Option<Receiver<Vec<f32>>>,
    export_receiver: Option<Receiver<Result<(), String>>>,
    playback: PlaybackHandle,
    is_playing: bool,
    paused: bool,
    playhead_time: f32,
    processed_mode: bool,
    processed_samples: Option<Vec<f32>>,
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
            processing_receiver: None,
            export_receiver: None,
            playback: PlaybackHandle::spawn(),
            is_playing: false,
            paused: false,
            playhead_time: 0.0,
            processed_mode: false,
            processed_samples: None,
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
        self.playhead_time = 0.0;
        let (sender, receiver) = mpsc::channel();
        self.load_receiver = Some(receiver);
        self.status = format!("Loading and analyzing {}…", path.display());
        thread::spawn(move || {
            let result = decode_file(&path)
                .map_err(|error| error.to_string())
                .map(|audio| {
                    let spec = spectrogram::compute(&audio, 2048, 512);
                    println!(
                        "Spectrogram: {} × {} (frequency bins × time frames)",
                        spec.n_freq, spec.n_time
                    );
                    let image = crate::ui::spectrogram_view::color_image(&spec);
                    (path, audio, spec, image)
                });
            let _ = sender.send(LoadMessage::Loaded(result));
        });
    }

    fn poll_workers(&mut self, ctx: &egui::Context) {
        let load_result = self
            .load_receiver
            .as_ref()
            .and_then(|receiver| receiver.try_recv().ok());
        if let Some(LoadMessage::Loaded(result)) = load_result {
            self.load_receiver = None;
            match result {
                Ok((path, audio, spec, image)) => {
                    let texture =
                        ctx.load_texture("spectrogram", image, egui::TextureOptions::LINEAR);
                    self.loaded = Some(LoadedAudio {
                        path,
                        audio,
                        spectrogram: spec,
                        texture,
                    });
                    self.playhead_time = 0.0;
                    self.spectrogram_view = SpectrogramView::default();
                    self.selections.clear();
                    self.processed_samples = None;
                    self.status = "Audio loaded successfully.".to_owned();
                }
                Err(error) => self.status = error,
            }
        }

        let processed = self
            .processing_receiver
            .as_ref()
            .and_then(|receiver| receiver.try_recv().ok());
        if let Some(samples) = processed {
            self.processing_receiver = None;
            self.processed_samples = Some(samples);
            if let Some(loaded) = &self.loaded {
                if self.processed_mode {
                    if let Some(samples) = &self.processed_samples {
                        let start_frame = (self.playhead_time * loaded.audio.sample_rate as f32)
                            as usize
                            * loaded.audio.channels as usize;
                        self.playback.play(
                            samples[start_frame.min(samples.len())..].to_vec(),
                            loaded.audio.sample_rate,
                            loaded.audio.channels,
                        );
                        self.is_playing = true;
                        self.paused = false;
                    }
                }
            }
            self.status = "Audio processing complete.".to_owned();
        }

        let exported = self
            .export_receiver
            .as_ref()
            .and_then(|receiver| receiver.try_recv().ok());
        if let Some(result) = exported {
            self.export_receiver = None;
            self.status = match result {
                Ok(()) => "WAV export completed.".to_owned(),
                Err(error) => error,
            };
        }
        if self.load_receiver.is_some()
            || self.processing_receiver.is_some()
            || self.export_receiver.is_some()
        {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn play(&mut self) {
        let Some(loaded) = &self.loaded else {
            return;
        };
        if !self.processed_mode {
            let start_frame = (self.playhead_time * loaded.audio.sample_rate as f32) as usize
                * loaded.audio.channels as usize;
            self.playback.play(
                loaded.audio.samples[start_frame.min(loaded.audio.samples.len())..].to_vec(),
                loaded.audio.sample_rate,
                loaded.audio.channels,
            );
            self.is_playing = true;
            self.paused = false;
        } else if let Some(samples) = &self.processed_samples {
            let start_frame = (self.playhead_time * loaded.audio.sample_rate as f32) as usize
                * loaded.audio.channels as usize;
            self.playback.play(
                samples[start_frame.min(samples.len())..].to_vec(),
                loaded.audio.sample_rate,
                loaded.audio.channels,
            );
            self.is_playing = true;
            self.paused = false;
        } else {
            let (sender, receiver) = mpsc::channel();
            let audio = AudioData {
                samples: loaded.audio.samples.clone(),
                sample_rate: loaded.audio.sample_rate,
                channels: loaded.audio.channels,
                duration: loaded.audio.duration,
            };
            let selections = self.selections.clone();
            self.processing_receiver = Some(receiver);
            self.status = "Processing edited audio…".to_owned();
            thread::spawn(move || {
                let _ = sender.send(apply_selections(&audio, &selections));
            });
        }
    }

    fn export(&mut self, path: PathBuf) {
        let Some(loaded) = &self.loaded else {
            return;
        };
        let audio = AudioData {
            samples: loaded.audio.samples.clone(),
            sample_rate: loaded.audio.sample_rate,
            channels: loaded.audio.channels,
            duration: loaded.audio.duration,
        };
        let selections = self.selections.clone();
        let (sender, receiver) = mpsc::channel();
        self.export_receiver = Some(receiver);
        self.status = "Processing and exporting WAV…".to_owned();
        thread::spawn(move || {
            let samples = apply_selections(&audio, &selections);
            let result = export_wav(&path, &samples, audio.sample_rate, audio.channels)
                .map_err(|error| format!("WAV export failed: {error}"));
            let _ = sender.send(result);
        });
    }

    fn selections_changed(&mut self) {
        self.processed_samples = None;
        if self.processed_mode {
            if self.is_playing {
                self.playhead_time += self.playback.position();
            }
            self.playback.stop();
            self.is_playing = false;
            self.paused = false;
        }
    }
}

impl eframe::App for SpectrimApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_workers(ctx);
        let processing = self.processing_receiver.is_some() || self.export_receiver.is_some();
        let current_time = if self.is_playing {
            self.playhead_time + self.playback.position()
        } else {
            self.playhead_time
        };
        let duration = self
            .loaded
            .as_ref()
            .map_or(0.0, |loaded| loaded.audio.duration);
        if self.is_playing && duration > 0.0 && current_time >= duration {
            self.is_playing = false;
            self.paused = false;
            self.playhead_time = duration;
        }

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
                    self.loaded.is_some() && !processing,
                );
                if events.play {
                    if self.paused {
                        self.paused = false;
                    }
                    self.play();
                }
                if events.pause {
                    self.playhead_time = current_time;
                    self.playback.stop();
                    self.is_playing = false;
                    self.paused = true;
                }
                if events.stop {
                    self.playback.stop();
                    self.is_playing = false;
                    self.paused = false;
                    self.playhead_time = 0.0;
                }
                ui.separator();
                ui.label("New selection:");
                ui.selectable_value(&mut self.default_action, SelectionAction::Mute, "Mute");
                ui.selectable_value(&mut self.default_action, SelectionAction::Solo, "Solo");
                ui.selectable_value(&mut self.default_action, SelectionAction::Gain, "Gain");
                if ui
                    .add_enabled(
                        self.loaded.is_some() && !processing,
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
                if let Some(time) = events.seek_to {
                    self.playhead_time = time.clamp(0.0, loaded.audio.duration);
                    if self.is_playing {
                        let start_frame = (self.playhead_time * loaded.audio.sample_rate as f32)
                            as usize
                            * loaded.audio.channels as usize;
                        let samples = if self.processed_mode {
                            self.processed_samples
                                .as_ref()
                                .unwrap_or(&loaded.audio.samples)
                        } else {
                            &loaded.audio.samples
                        };
                        self.playback.play(
                            samples[start_frame.min(samples.len())..].to_vec(),
                            loaded.audio.sample_rate,
                            loaded.audio.channels,
                        );
                    }
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
                ui.label(&self.status);
                if processing {
                    ui.spinner();
                }
            });
        });

        if self.is_playing {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
            if self.is_playing {
                self.playback.stop();
                self.playhead_time = current_time;
                self.is_playing = false;
                self.paused = true;
            } else if self.paused {
                self.paused = false;
                self.play();
            } else {
                self.play();
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
