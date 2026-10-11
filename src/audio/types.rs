use std::f32::consts::PI;

/// Decoded interleaved PCM audio.
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration: f32,
}

impl AudioData {
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    pub fn to_mono(&self) -> Vec<f32> {
        let channels = self.channels as usize;
        if channels == 0 {
            return Vec::new();
        }
        self.samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    }

    pub fn time_to_frame(&self, time: f32) -> usize {
        if !time.is_finite() || self.sample_rate == 0 {
            return 0;
        }
        (time.max(0.0) as f64 * self.sample_rate as f64) as usize
    }
}

/// STFT result; matrix storage is frequency-major: matrix[freq_bin * n_time + time_frame].
pub struct Spectrogram {
    pub matrix: Vec<f32>,
    pub n_freq: usize,
    pub n_time: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    pub db_min: f32,
    pub db_max: f32,
}

impl Spectrogram {
    pub fn at(&self, freq_bin: usize, time_frame: usize) -> f32 {
        if freq_bin >= self.n_freq || time_frame >= self.n_time {
            return 0.0;
        }
        self.matrix[freq_bin * self.n_time + time_frame]
    }

    pub fn bin_to_hz(&self, bin: usize) -> f32 {
        if self.n_fft == 0 {
            return 0.0;
        }
        bin as f32 * self.sample_rate as f32 / self.n_fft as f32
    }

    pub fn hz_to_bin(&self, hz: f32) -> usize {
        if self.sample_rate == 0 {
            return 0;
        }
        ((hz.max(0.0) * self.n_fft as f32 / self.sample_rate as f32).round() as usize)
            .min(self.n_freq.saturating_sub(1))
    }

    pub fn frame_to_time(&self, frame: usize) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        frame as f32 * self.hop_length as f32 / self.sample_rate as f32
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SelectionAction {
    Mute,
    Solo,
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub id: u64,
    pub time_start: f32,
    pub time_end: f32,
    pub freq_low: f32,
    pub freq_high: f32,
    pub action: SelectionAction,
    pub full_band: bool,
}

impl Selection {
    pub fn new(
        id: u64,
        time_a: f32,
        time_b: f32,
        freq_a: f32,
        freq_b: f32,
        action: SelectionAction,
        full_band: bool,
    ) -> Self {
        Self {
            id,
            time_start: time_a.min(time_b),
            time_end: time_a.max(time_b),
            freq_low: freq_a.min(freq_b),
            freq_high: freq_a.max(freq_b),
            action,
            full_band,
        }
    }
}

pub(crate) fn hann_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    (0..length)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / (length - 1) as f32).cos())
        .collect()
}
