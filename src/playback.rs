//! Audio output thread with sample-accurate position tracking and seamless
//! buffer swaps.
//!
//! The previous implementation rebuilt a `Sink` with a fresh `SamplesBuffer`
//! containing only `samples[current_position..]` on every edit, and tracked
//! position with a wall clock (`Instant::elapsed`). That produced the two
//! symptoms this rewrite removes:
//!   * **position reset on edit** — restarting the sink zeroed its clock, so the
//!     playhead jumped back to the start of the newly-fed slice;
//!   * **clock drift** — `Instant::elapsed` is not the device's sample counter,
//!     so the drawn playhead slowly diverged from what was actually audible.
//!
//! The design now:
//!   * the buffer handed to the mixer is always the **whole track**, and the
//!     source tracks an absolute sample index into it;
//!   * `position_frames` is that absolute index divided by the channel count,
//!     advanced by the mixer as it pulls samples, so the UI position is
//!     sample-accurate by construction;
//!   * an edit installs a new buffer through an atomic generation counter and
//!     the source keeps its index — playback continues from the same musical
//!     position with no restart and no gap.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use rodio::{OutputStreamBuilder, Sink, Source};

/// Sentinel meaning "no seek requested".
const NO_SEEK: u64 = u64::MAX;

enum AudioCmd {
    /// Attach a new buffer and start from an absolute frame position.
    Play {
        samples: Arc<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
        start_frame: usize,
    },
    /// Replace the buffer content in place. The source keeps its current
    /// position, so this is the seamless path used after an edit.
    Update {
        samples: Arc<Vec<f32>>,
    },
    Pause,
    Resume,
    /// Jump to an absolute frame without rebuilding the sink, so scrubbing the
    /// playhead while playing stays smooth.
    Seek {
        frame: usize,
    },
    Stop,
}

/// Shared state between the command thread and the mixer source.
struct StreamState {
    inner: Mutex<Arc<Vec<f32>>>,
    /// Bumped on every buffer swap; the source latches a new `Arc` when it
    /// notices the value change.
    generation: AtomicU64,
    /// Absolute frame position, written by the source as it plays.
    position_frames: AtomicU64,
    /// Pending seek target. `u64::MAX` means "none"; the source consumes it with
    /// a swap so exactly one reader picks it up.
    seek_frames: AtomicU64,
    stopped: AtomicBool,
}

impl StreamState {
    fn new(samples: Arc<Vec<f32>>, start_frame: usize) -> Arc<Self> {
        let state = Arc::new(Self {
            inner: Mutex::new(Arc::clone(&samples)),
            generation: AtomicU64::new(1),
            position_frames: AtomicU64::new(start_frame as u64),
            seek_frames: AtomicU64::new(NO_SEEK),
            stopped: AtomicBool::new(false),
        });
        state
    }

    fn request_seek(&self, frame: usize) {
        self.seek_frames.store(frame as u64, Ordering::SeqCst);
        self.position_frames.store(frame as u64, Ordering::Relaxed);
    }

    fn swap_buffer(&self, samples: Arc<Vec<f32>>) {
        *self.inner.lock().unwrap() = samples;
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    fn buffer(&self) -> Arc<Vec<f32>> {
        Arc::clone(&self.inner.lock().unwrap())
    }
}

/// A `Source` streaming from a swappable whole-track buffer.
///
/// `index` is an absolute interleaved-sample offset. Because the buffer always
/// covers the entire track, a swap leaves `index` valid and playback continues
/// from the same musical position.
struct SwapSource {
    state: Arc<StreamState>,
    buffer: Arc<Vec<f32>>,
    channels: usize,
    sample_rate: u32,
    index: usize,
    generation: u64,
}

impl Source for SwapSource {
    fn current_span_len(&self) -> Option<usize> {
        // Report the rest of the buffer; rodio only uses this for batching.
        Some(self.buffer.len().saturating_sub(self.index))
    }

    fn channels(&self) -> u16 {
        self.channels as u16
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

impl Iterator for SwapSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.state.stopped.load(Ordering::Relaxed) {
            return None;
        }

        // Honour a pending seek before pulling the next sample.
        let pending = self.state.seek_frames.swap(NO_SEEK, Ordering::SeqCst);
        if pending != NO_SEEK {
            self.index = (pending as usize).saturating_mul(self.channels.max(1));
        }

        // Pick up a newer buffer if one was installed since the last sample.
        // The index is deliberately preserved: that is what makes an edit
        // seamless instead of a restart.
        let generation = self.state.generation.load(Ordering::SeqCst);
        if generation != self.generation {
            self.buffer = self.state.buffer();
            self.generation = generation;
            // A shorter buffer (should not happen for same-track edits) must not
            // read out of bounds; clamp and let the stream end naturally.
            let frame = self.index / self.channels.max(1);
            self.state
                .position_frames
                .store(frame as u64, Ordering::Relaxed);
        }

        if self.index >= self.buffer.len() {
            let frame = self.index / self.channels.max(1);
            self.state
                .position_frames
                .store(frame as u64, Ordering::Relaxed);
            return None;
        }

        let sample = self.buffer[self.index];
        self.index += 1;
        // Publish position once per complete frame so the UI sees frame-aligned
        // time rather than a channel-interleaved fraction.
        if self.index % self.channels.max(1) == 0 {
            self.state
                .position_frames
                .store((self.index / self.channels.max(1)) as u64, Ordering::Relaxed);
        }
        Some(sample)
    }
}

#[derive(Clone)]
pub struct PlaybackHandle {
    tx: Sender<AudioCmd>,
    /// Shared with the audio thread's current `StreamState` so `position()`
    /// works even between commands.
    state: Arc<Mutex<Option<Arc<StreamState>>>>,
    sample_rate: Arc<AtomicU64>,
    /// Set while a source is attached to the mixer; cleared when the source
    /// runs out or on `Stop`. Lets the UI detect a natural end of track.
    is_active: Arc<AtomicBool>,
}

impl PlaybackHandle {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<AudioCmd>();
        let state: Arc<Mutex<Option<Arc<StreamState>>>> = Arc::new(Mutex::new(None));
        let sample_rate = Arc::new(AtomicU64::new(44_100));
        let is_active = Arc::new(AtomicBool::new(false));

        let thread_state = Arc::clone(&state);
        let thread_sample_rate = Arc::clone(&sample_rate);
        let thread_active = Arc::clone(&is_active);

        thread::spawn(move || {
            let stream = match OutputStreamBuilder::open_default_stream() {
                Ok(stream) => stream,
                Err(error) => {
                    eprintln!("Audio output unavailable: {error}");
                    return;
                }
            };
            let mixer = stream.mixer();
            let mut sink: Option<Sink> = None;

            loop {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(AudioCmd::Play {
                        samples,
                        sample_rate: rate,
                        channels,
                        start_frame,
                    }) => {
                        if let Some(old_sink) = sink.take() {
                            old_sink.stop();
                        }
                        thread_sample_rate.store(rate as u64, Ordering::Relaxed);

                        let new_state = StreamState::new(Arc::clone(&samples), start_frame);
                        let channel_count = channels.max(1) as usize;
                        let source = SwapSource {
                            state: Arc::clone(&new_state),
                            buffer: samples,
                            channels: channel_count,
                            sample_rate: rate,
                            index: start_frame * channel_count,
                            generation: new_state.generation.load(Ordering::SeqCst),
                        };
                        let new_sink = Sink::connect_new(mixer);
                        new_sink.append(source);
                        new_sink.play();
                        sink = Some(new_sink);
                        *thread_state.lock().unwrap() = Some(new_state);
                        thread_active.store(true, Ordering::Relaxed);
                    }
                    Ok(AudioCmd::Seek { frame }) => {
                        if let Some(current) = thread_state.lock().unwrap().as_ref() {
                            current.request_seek(frame);
                        }
                    }
                    Ok(AudioCmd::Update { samples }) => {
                        if let Some(current) = thread_state.lock().unwrap().as_ref() {
                            current.swap_buffer(samples);
                        }
                    }
                    // Pause/Resume go through the sink itself and deliberately do
                    // not touch `is_active` or `position_frames`: the source stays
                    // attached at its current frame, so the playhead holds still
                    // and resuming continues from exactly there.
                    Ok(AudioCmd::Pause) => {
                        if let Some(sink) = &sink {
                            sink.pause();
                        }
                    }
                    Ok(AudioCmd::Resume) => {
                        if let Some(sink) = &sink {
                            sink.play();
                        }
                    }
                    Ok(AudioCmd::Stop) => {
                        if let Some(active) = sink.take() {
                            active.stop();
                        }
                        if let Some(current) = thread_state.lock().unwrap().as_ref() {
                            current.stopped.store(true, Ordering::Relaxed);
                            current.position_frames.store(0, Ordering::Relaxed);
                        }
                        *thread_state.lock().unwrap() = None;
                        thread_active.store(false, Ordering::Relaxed);
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }

                // Reached the end of the buffer -> mark inactive so the UI stops
                // the transport on its own instead of spinning forever.
                if sink.as_ref().is_some_and(|s| s.empty()) {
                    sink = None;
                    thread_active.store(false, Ordering::Relaxed);
                }
            }
            drop(stream);
        });

        Self {
            tx,
            state,
            sample_rate,
            is_active,
        }
    }

    /// Start playback of the whole track from `position_seconds`.
    pub fn play_from(
        &self,
        samples: Arc<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
        position_seconds: f32,
    ) {
        let start_frame = (position_seconds.max(0.0) * sample_rate as f32).round() as usize;
        let _ = self.tx.send(AudioCmd::Play {
            samples,
            sample_rate,
            channels,
            start_frame,
        });
    }

    /// Swap in edited audio without restarting. Playback continues from its
    /// current position — this is the fix for "每次改动后播放位置被重置".
    pub fn update_buffer(&self, samples: Arc<Vec<f32>>) {
        let _ = self.tx.send(AudioCmd::Update { samples });
    }

    /// Jump to an absolute position while keeping the current sink alive.
    pub fn seek(&self, seconds: f32, sample_rate: u32) {
        let frame = (seconds.max(0.0) * sample_rate as f32).round() as usize;
        let _ = self.tx.send(AudioCmd::Seek { frame });
    }

    pub fn stop(&self) {
        let _ = self.tx.send(AudioCmd::Stop);
    }

    pub fn pause(&self) {
        let _ = self.tx.send(AudioCmd::Pause);
    }

    pub fn resume(&self) {
        let _ = self.tx.send(AudioCmd::Resume);
    }

    /// Absolute play position in seconds, counted in frames actually consumed by
    /// the audio device.
    pub fn position(&self) -> f32 {
        let sample_rate = self.sample_rate.load(Ordering::Relaxed).max(1) as f32;
        let frames = self
            .state
            .lock()
            .unwrap()
            .as_ref()
            .map(|state| state.position_frames.load(Ordering::Relaxed))
            .unwrap_or(0);
        frames as f32 / sample_rate
    }

    /// True while a source is attached to the mixer (playing or paused). Goes
    /// false when the track finishes on its own, which is how the UI stops the
    /// transport without guessing from elapsed time.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
}
