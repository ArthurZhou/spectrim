use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use rodio::{buffer::SamplesBuffer, OutputStreamBuilder, Sink};

pub enum AudioCmd {
    Play {
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    },
    Pause,
    Resume,
    Stop,
}

#[derive(Clone)]
pub struct PlaybackHandle {
    tx: mpsc::Sender<AudioCmd>,
    position_ms: Arc<AtomicU32>,
}

impl PlaybackHandle {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<AudioCmd>();
        let position_ms = Arc::new(AtomicU32::new(0));
        let thread_position = Arc::clone(&position_ms);
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
            let mut started = Instant::now();
            let mut paused_at: Option<Instant> = None;
            let mut paused_total = Duration::ZERO;
            loop {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(AudioCmd::Play {
                        samples,
                        sample_rate,
                        channels,
                    }) => {
                        if let Some(old_sink) = sink.take() {
                            old_sink.stop();
                        }
                        let new_sink = Sink::connect_new(mixer);
                        new_sink.append(SamplesBuffer::new(channels, sample_rate, samples));
                        new_sink.play();
                        sink = Some(new_sink);
                        started = Instant::now();
                        paused_at = None;
                        paused_total = Duration::ZERO;
                        thread_position.store(0, Ordering::Relaxed);
                    }
                    Ok(AudioCmd::Pause) => {
                        if let Some(active) = &sink {
                            active.pause();
                            paused_at = Some(Instant::now());
                        }
                    }
                    Ok(AudioCmd::Resume) => {
                        if let Some(active) = &sink {
                            active.play();
                            if let Some(pause_start) = paused_at.take() {
                                paused_total += pause_start.elapsed();
                            }
                        }
                    }
                    Ok(AudioCmd::Stop) => {
                        if let Some(active) = sink.take() {
                            active.stop();
                        }
                        thread_position.store(0, Ordering::Relaxed);
                        paused_at = None;
                        paused_total = Duration::ZERO;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
                if sink.is_some() && paused_at.is_none() {
                    let elapsed = started.elapsed().saturating_sub(paused_total);
                    thread_position.store(
                        elapsed.as_millis().min(u32::MAX as u128) as u32,
                        Ordering::Relaxed,
                    );
                }
                if sink.as_ref().is_some_and(Sink::empty) {
                    sink = None;
                    paused_at = None;
                    paused_total = Duration::ZERO;
                }
            }
            drop(stream);
        });
        Self { tx, position_ms }
    }

    pub fn play(&self, samples: Vec<f32>, sample_rate: u32, channels: u16) {
        let _ = self.tx.send(AudioCmd::Play {
            samples,
            sample_rate,
            channels,
        });
    }
    pub fn pause(&self) {
        let _ = self.tx.send(AudioCmd::Pause);
    }
    pub fn resume(&self) {
        let _ = self.tx.send(AudioCmd::Resume);
    }
    pub fn stop(&self) {
        let _ = self.tx.send(AudioCmd::Stop);
    }
    pub fn position(&self) -> f32 {
        self.position_ms.load(Ordering::Relaxed) as f32 / 1000.0
    }
}
