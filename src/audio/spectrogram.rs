use std::borrow::Cow;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use super::types::{hann_window, AudioData, Spectrogram};

pub fn compute(audio: &AudioData, n_fft: usize, hop: usize) -> Spectrogram {
    let mono = mono_from_interleaved(&audio.samples, audio.channels);
    compute_mono(mono.as_ref(), audio.sample_rate, n_fft, hop)
}

/// Compute the spectrogram of an interleaved PCM buffer without cloning it into
/// an `AudioData`. Used for updating the edited-playback visualization.
pub fn compute_interleaved(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
) -> Spectrogram {
    let mono = mono_from_interleaved(samples, channels);
    compute_mono(mono.as_ref(), sample_rate, n_fft, hop)
}

fn mono_from_interleaved(samples: &[f32], channels: u16) -> Cow<'_, [f32]> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return Cow::Borrowed(&[]);
    }
    if channel_count == 1 {
        return Cow::Borrowed(samples);
    }
    Cow::Owned(
        samples
            .par_chunks_exact(channel_count)
            .map(|frame| frame.iter().sum::<f32>() / channel_count as f32)
            .collect(),
    )
}

fn compute_mono(mono: &[f32], sample_rate: u32, n_fft: usize, hop: usize) -> Spectrogram {
    let n_freq = n_fft / 2 + 1;
    if n_fft == 0 || hop == 0 || mono.len() < n_fft {
        return Spectrogram {
            matrix: Vec::new(),
            n_freq,
            n_time: 0,
            n_fft,
            hop_length: hop,
            sample_rate,
            db_min: 0.0,
            db_max: 0.0,
        };
    }

    let n_time = (mono.len() - n_fft) / hop + 1;
    let window = hann_window(n_fft);
    // rustfft 6 plans are Send + Sync, so one shared forward plan serves every
    // worker; each frame gets its own scratch spectrum inside the map closure.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Store results time-major first, so each Rayon worker writes a contiguous
    // frame and no per-frame Vec allocations are needed. Transpose once into
    // the frequency-major layout used by the UI/DSP contracts.
    let mut time_major = vec![0.0f32; n_freq * n_time];
    time_major.par_chunks_mut(n_freq).enumerate().for_each_init(
        || vec![Complex32::default(); n_fft],
        |frame, (time, output)| {
            let start = time * hop;
            for index in 0..n_fft {
                frame[index] = Complex32::new(mono[start + index] * window[index], 0.0);
            }
            fft.process(frame.as_mut_slice());
            for (frequency, value) in output.iter_mut().enumerate() {
                *value = 20.0 * (frame[frequency].norm() + 1.0e-9).log10();
            }
        },
    );

    let mut matrix = vec![0.0f32; n_freq * n_time];
    matrix
        .par_chunks_mut(n_time)
        .enumerate()
        .for_each(|(frequency, row)| {
            for (time, value) in row.iter_mut().enumerate() {
                *value = time_major[time * n_freq + frequency];
            }
        });

    let (db_min, db_max) = compute_db_range(&matrix);
    Spectrogram {
        matrix,
        n_freq,
        n_time,
        n_fft,
        hop_length: hop,
        sample_rate,
        db_min,
        db_max,
    }
}

/// 5th/95th percentile of the dB distribution, used to normalise the colormap.
///
/// This used to clone the whole matrix and fully sort it — O(n log n) over tens
/// of millions of floats, and it was the dominant tail of load time on long
/// tracks. A fixed-bin histogram reduced in parallel gives the same percentiles
/// for colour-mapping purposes in O(n), which is plenty accurate here: the
/// values only drive a normalisation range, not audio.
fn compute_db_range(matrix: &[f32]) -> (f32, f32) {
    const BINS: usize = 4096;
    if matrix.is_empty() {
        return (0.0, 0.0);
    }

    let (min, max) = matrix
        .par_iter()
        .fold(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), value| (lo.min(*value), hi.max(*value)),
        )
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(a_lo, a_hi), (b_lo, b_hi)| (a_lo.min(b_lo), a_hi.max(b_hi)),
        );
    let span = max - min;
    if !span.is_finite() || span < 1.0e-6 {
        return (min, max);
    }

    let scale = (BINS - 1) as f32 / span;
    let chunk = (matrix.len() / (rayon::current_num_threads() * 4).max(1) + 1).max(1);
    let total: Vec<u64> = matrix
        .par_chunks(chunk)
        .map(|chunk| {
            let mut histogram = vec![0u64; BINS];
            for value in chunk {
                let bin = (((value - min) * scale).round() as usize).min(BINS - 1);
                histogram[bin] += 1;
            }
            histogram
        })
        .reduce_with(|mut a, b| {
            for (slot, count) in a.iter_mut().zip(b.into_iter()) {
                *slot += count;
            }
            a
        })
        .unwrap_or_else(|| vec![0u64; BINS]);

    let percentile = |fraction: f64| {
        let target = (matrix.len() as f64 * fraction) as u64;
        let mut cumulative = 0u64;
        for (bin, count) in total.iter().enumerate() {
            cumulative += count;
            if cumulative > target {
                return min + bin as f32 / scale;
            }
        }
        max
    };
    (percentile(0.05), percentile(0.95))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn test_audio(frames: usize, sample_rate: u32) -> AudioData {
        AudioData {
            samples: (0..frames)
                .map(|i| (i as f32 * 0.01).sin() + (i as f32 * 0.13).cos() * 0.5)
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        }
    }

    #[test]
    fn spectrogram_shape_and_peak_bin_are_sane() {
        let sample_rate = 8000;
        let frames = 16_000;
        let audio = AudioData {
            samples: (0..frames)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        };
        let spec = compute(&audio, 512, 128);
        assert_eq!(spec.n_freq, 257);
        assert_eq!(spec.n_time, (frames - 512) / 128 + 1);
        assert!(spec.db_min < spec.db_max);

        let peak_bin = (0..spec.n_freq)
            .map(|bin| {
                let energy: f32 = (0..spec.n_time).map(|t| spec.at(bin, t)).sum();
                (bin, energy)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(bin, _)| bin)
            .unwrap();
        let peak_hz = spec.bin_to_hz(peak_bin);
        assert!((peak_hz - 440.0).abs() < 40.0, "peak at {peak_hz} Hz");
    }

    #[test]
    fn parallel_compute_matches_sequential_reference() {
        let audio = test_audio(4_000, 8_000);
        let spec = compute(&audio, 256, 64);

        // Straightforward single-threaded reference implementation.
        let mono = audio.to_mono();
        let window = hann_window(256);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(256);
        let n_freq = 129;
        let n_time = (mono.len() - 256) / 64 + 1;
        let mut max_err = 0.0f32;
        let mut frame = vec![Complex32::default(); 256];
        for time in 0..n_time {
            let start = time * 64;
            for index in 0..256 {
                frame[index] = Complex32::new(mono[start + index] * window[index], 0.0);
            }
            fft.process(&mut frame);
            for frequency in 0..n_freq {
                let db = 20.0 * (frame[frequency].norm() + 1.0e-9).log10();
                max_err = max_err.max((db - spec.at(frequency, time)).abs());
            }
        }
        assert!(max_err < 1.0e-3, "parallel spectrogram drift {max_err}");
    }

    #[test]
    fn db_range_brackets_the_bulk_of_values() {
        let audio = test_audio(40_000, 44_100);
        let spec = compute(&audio, 512, 128);
        assert!(spec.db_min.is_finite() && spec.db_max.is_finite());
        assert!(spec.db_min < spec.db_max);
        // Percentiles must sit inside the true min/max.
        let true_min = spec.matrix.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        let true_max = spec.matrix.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        assert!(spec.db_min >= true_min - 1.0e-3);
        assert!(spec.db_max <= true_max + 1.0e-3);
        // 5th percentile should be above the true minimum for real audio.
        assert!(spec.db_min > true_min);
    }

    #[test]
    fn empty_or_short_audio_returns_empty_spectrogram() {
        let audio = test_audio(100, 44_100);
        let spec = compute(&audio, 512, 128);
        assert_eq!(spec.n_time, 0);
        assert!(spec.matrix.is_empty());
    }

    #[test]
    #[ignore = "manual three-minute STFT throughput check"]
    fn benchmark_three_minute_mono_stft() {
        let sample_rate = 44_100;
        let frames = sample_rate as usize * 180;
        let samples = (0..frames)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect::<Vec<_>>();
        let audio = AudioData {
            samples,
            sample_rate,
            channels: 1,
            duration: 180.0,
        };
        let started = Instant::now();
        let spec = compute(&audio, 2048, 512);
        eprintln!(
            "3-minute STFT: {:.2}s, {} Rayon workers, {}×{} matrix",
            started.elapsed().as_secs_f32(),
            rayon::current_num_threads(),
            spec.n_freq,
            spec.n_time
        );
        assert_eq!(spec.n_time, (frames - 2048) / 512 + 1);
    }
}
