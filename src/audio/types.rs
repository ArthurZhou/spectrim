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
    /// Per-time-frame local maxima above the analysis floor, in frequency bins.
    pub peaks: Vec<Vec<usize>>,
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
    Solo,
    Gain,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SelectionShape {
    Rect,
    Polygon {
        points: Vec<(f32, f32)>,
    },
    /// Per-STFT-frame frequency-bin intervals. Frame/bin coordinates refer to
    /// the analysis spectrogram used when this mask was created.
    Mask {
        first_frame: usize,
        ranges: Vec<(usize, usize)>,
        sample_rate: u32,
        hop_length: usize,
        n_fft: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionTool {
    Rect,
    Polygon,
    Snap,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub id: u64,
    pub time_start: f32,
    pub time_end: f32,
    pub freq_low: f32,
    pub freq_high: f32,
    pub action: SelectionAction,
    pub enabled: bool,
    /// Gain in dB; -60 dB is the practical silence floor and 0 dB is unity.
    pub gain_db: f32,
    pub full_band: bool,
    pub shape: SelectionShape,
}

impl Selection {
    pub fn gain_linear(&self) -> f32 {
        if !self.gain_db.is_finite() || self.gain_db <= -60.0 {
            0.0
        } else {
            10.0_f32.powf(self.gain_db.min(12.0) / 20.0)
        }
    }

    pub fn is_polygon(&self) -> bool {
        matches!(self.shape, SelectionShape::Polygon { .. })
    }

    pub fn is_mask(&self) -> bool {
        matches!(self.shape, SelectionShape::Mask { .. })
    }

    pub fn requires_spectral_processing(&self) -> bool {
        !matches!(self.shape, SelectionShape::Rect)
    }

    pub fn refresh_bounds(&mut self) {
        if let SelectionShape::Polygon { points } = &self.shape {
            if points.is_empty() {
                return;
            }
            self.time_start = points
                .iter()
                .map(|point| point.0)
                .fold(f32::INFINITY, f32::min);
            self.time_end = points
                .iter()
                .map(|point| point.0)
                .fold(f32::NEG_INFINITY, f32::max);
            self.freq_low = points
                .iter()
                .map(|point| point.1)
                .fold(f32::INFINITY, f32::min);
            self.freq_high = points
                .iter()
                .map(|point| point.1)
                .fold(f32::NEG_INFINITY, f32::max);
        }
    }

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
            enabled: true,
            gain_db: 0.0,
            full_band,
            shape: SelectionShape::Rect,
        }
    }

    pub fn polygon(id: u64, points: Vec<(f32, f32)>, action: SelectionAction) -> Self {
        let time_start = points
            .iter()
            .map(|point| point.0)
            .fold(f32::INFINITY, f32::min);
        let time_end = points
            .iter()
            .map(|point| point.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let freq_low = points
            .iter()
            .map(|point| point.1)
            .fold(f32::INFINITY, f32::min);
        let freq_high = points
            .iter()
            .map(|point| point.1)
            .fold(f32::NEG_INFINITY, f32::max);
        Self {
            id,
            time_start,
            time_end,
            freq_low,
            freq_high,
            action,
            enabled: true,
            gain_db: 0.0,
            full_band: false,
            shape: SelectionShape::Polygon { points },
        }
    }

    pub fn mask(
        id: u64,
        first_frame: usize,
        ranges: Vec<(usize, usize)>,
        hop: usize,
        sample_rate: u32,
        n_fft: usize,
        action: SelectionAction,
    ) -> Self {
        let time_start = first_frame as f32 * hop as f32 / sample_rate.max(1) as f32;
        let time_end = (first_frame + ranges.len().saturating_sub(1)) as f32 * hop as f32
            / sample_rate.max(1) as f32;
        let freq_low = ranges.iter().map(|range| range.0).min().unwrap_or(0) as f32
            * sample_rate as f32
            / n_fft.max(1) as f32;
        let freq_high = ranges.iter().map(|range| range.1).max().unwrap_or(0) as f32
            * sample_rate as f32
            / n_fft.max(1) as f32;
        Self {
            id,
            time_start,
            time_end,
            freq_low,
            freq_high,
            action,
            enabled: true,
            gain_db: 0.0,
            full_band: false,
            shape: SelectionShape::Mask {
                first_frame,
                ranges,
                sample_rate,
                hop_length: hop,
                n_fft,
            },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_db_converts_to_linear_amplitude_and_infinity_mutes() {
        let mut selection = Selection::new(1, 0.0, 1.0, 0.0, 1_000.0, SelectionAction::Gain, true);
        assert!((selection.gain_linear() - 1.0).abs() < 1e-6);
        selection.gain_db = -6.0206;
        assert!((selection.gain_linear() - 0.5).abs() < 1e-4);
        selection.gain_db = 6.0;
        assert!((selection.gain_linear() - 1.9952623).abs() < 1e-5);
        selection.gain_db = f32::NEG_INFINITY;
        assert_eq!(selection.gain_linear(), 0.0);
    }
}
