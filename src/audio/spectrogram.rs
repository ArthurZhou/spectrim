use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use super::types::{hann_window, AudioData, Spectrogram};

pub fn compute(audio: &AudioData, n_fft: usize, hop: usize) -> Spectrogram {
    let mono = mono_from_interleaved(&audio.samples, audio.channels);
    compute_mono(mono.as_ref(), audio.sample_rate, n_fft, hop)
}

/// Cancellable full transform used during file loading. Workers check the token
/// between frames, then stop doing FFT work once cancellation is requested.
pub fn compute_cancellable(
    audio: &AudioData,
    n_fft: usize,
    hop: usize,
    cancelled: &AtomicBool,
) -> Option<Spectrogram> {
    let mono = mono_from_interleaved(&audio.samples, audio.channels);
    compute_mono_inner(
        mono.as_ref(),
        audio.sample_rate,
        n_fft,
        hop,
        Some(cancelled),
    )
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

pub fn compute_interleaved_cancellable(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    cancelled: &AtomicBool,
) -> Option<Spectrogram> {
    let mono = mono_from_interleaved(samples, channels);
    compute_mono_inner(mono.as_ref(), sample_rate, n_fft, hop, Some(cancelled))
}

/// A frequency-major patch of spectrogram columns changed by a PCM frame span.
#[derive(Debug)]
pub struct SpectrogramPatch {
    pub first_time_frame: usize,
    pub n_time: usize,
    pub n_freq: usize,
    pub matrix: Vec<f32>,
}

/// Source-frame span which must be invalidated so patch FFT windows cover every
/// changed PCM sample, while expanding to whole display columns when texture
/// generation uses time downsampling.
pub fn expanded_patch_frames(
    changed_frames: (usize, usize),
    n_total_time: usize,
    total_pcm_frames: usize,
    n_fft: usize,
    hop: usize,
    max_texture_width: usize,
) -> (usize, usize) {
    if n_total_time == 0 || total_pcm_frames < n_fft || n_fft == 0 || hop == 0 {
        return changed_frames;
    }
    let width = n_total_time.min(max_texture_width.max(1));
    let mut first_fft = if changed_frames.0 < n_fft {
        0
    } else {
        (changed_frames.0 - n_fft) / hop + 1
    };
    first_fft = first_fft.min(n_total_time);
    let mut last_fft = changed_frames.1.saturating_sub(1) / hop + 1;
    if last_fft <= first_fft {
        return changed_frames;
    }

    // Expand to exactly the display columns intersected by the changed FFT
    // frames; egui patches can then replace full columns without stale pixels.
    let first_x = first_fft * width / n_total_time;
    let last_x_exclusive = (last_fft * width).div_ceil(n_total_time).min(width);
    first_fft = first_x * n_total_time / width;
    last_fft = (last_x_exclusive * n_total_time)
        .div_ceil(width)
        .min(n_total_time);
    let pcm_start = first_fft.saturating_mul(hop);
    let pcm_end = last_fft
        .saturating_sub(1)
        .saturating_mul(hop)
        .saturating_add(n_fft)
        .min(total_pcm_frames);
    (pcm_start, pcm_end.max(pcm_start))
}

/// Recompute only STFT windows intersecting the changed PCM frame range
/// `[changed_start, changed_end)`. Frame values retain the full spectrogram's
/// global dB normalization; callers can splice this patch into the cached matrix.
pub fn compute_interleaved_patch(
    samples: &[f32],
    channels: u16,
    _sample_rate: u32,
    n_fft: usize,
    hop: usize,
    changed_start: usize,
    changed_end: usize,
) -> SpectrogramPatch {
    let channel_count = channels as usize;
    let n_freq = n_fft / 2 + 1;
    let total_frames = if channel_count == 0 {
        0
    } else {
        samples.len() / channel_count
    };
    let start = changed_start.min(total_frames);
    let end = changed_end.min(total_frames);
    if channel_count == 0 || n_fft == 0 || hop == 0 || end <= start || total_frames < n_fft {
        return SpectrogramPatch {
            first_time_frame: 0,
            n_time: 0,
            n_freq,
            matrix: Vec::new(),
        };
    }

    let total_time = (total_frames - n_fft) / hop + 1;
    // A window beginning before `end` and ending after `start` overlaps the
    // changed samples. The half-open ranges avoid recomputing adjacent columns.
    let first = if start < n_fft {
        0
    } else {
        (start - n_fft) / hop + 1
    }
    .min(total_time);
    let last_exclusive = ((end - 1) / hop + 1).min(total_time);
    if last_exclusive <= first {
        return SpectrogramPatch {
            first_time_frame: first,
            n_time: 0,
            n_freq,
            matrix: Vec::new(),
        };
    }

    let window = hann_window(n_fft);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let patch_time = last_exclusive - first;
    let mut time_major = vec![0.0f32; n_freq * patch_time];
    time_major.par_chunks_mut(n_freq).enumerate().for_each_init(
        || vec![Complex32::default(); n_fft],
        |spectrum, (local_time, output)| {
            let pcm_start = (first + local_time) * hop;
            for i in 0..n_fft {
                let frame =
                    &samples[(pcm_start + i) * channel_count..(pcm_start + i + 1) * channel_count];
                let mono = frame.iter().sum::<f32>() / channel_count as f32;
                spectrum[i] = Complex32::new(mono * window[i], 0.0);
            }
            fft.process(spectrum.as_mut_slice());
            for (frequency, value) in output.iter_mut().enumerate() {
                let magnitude = spectrum[frequency].norm();
                *value = if magnitude <= 1.0e-10 {
                    f32::NEG_INFINITY
                } else {
                    20.0 * magnitude.log10()
                };
            }
        },
    );
    let mut matrix = vec![0.0f32; n_freq * patch_time];
    matrix
        .par_chunks_mut(patch_time)
        .enumerate()
        .for_each(|(frequency, row)| {
            for (time, value) in row.iter_mut().enumerate() {
                *value = time_major[time * n_freq + frequency];
            }
        });

    SpectrogramPatch {
        first_time_frame: first,
        n_time: patch_time,
        n_freq,
        matrix,
    }
}

pub fn build_peak_map_for_spectrogram(spectrogram: &Spectrogram) -> Vec<Vec<usize>> {
    let floor = spectrogram.db_min - 40.0;
    build_peak_map(
        &spectrogram.matrix,
        spectrogram.n_freq,
        spectrogram.n_time,
        floor,
    )
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
    compute_mono_inner(mono, sample_rate, n_fft, hop, None)
        .expect("non-cancellable spectrogram computation returns a result")
}

fn compute_mono_inner(
    mono: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    cancelled: Option<&AtomicBool>,
) -> Option<Spectrogram> {
    let n_freq = n_fft / 2 + 1;
    if n_fft == 0 || hop == 0 || mono.len() < n_fft {
        return Some(Spectrogram {
            matrix: Vec::new(),
            n_freq,
            n_time: 0,
            n_fft,
            hop_length: hop,
            sample_rate,
            db_min: 0.0,
            db_max: 0.0,
            peaks: Vec::new(),
        });
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
            if cancelled.is_some_and(|token| token.load(Ordering::Relaxed)) {
                return;
            }
            let start = time * hop;
            for index in 0..n_fft {
                frame[index] = Complex32::new(mono[start + index] * window[index], 0.0);
            }
            fft.process(frame.as_mut_slice());
            for (frequency, value) in output.iter_mut().enumerate() {
                let magnitude = frame[frequency].norm();
                *value = if magnitude <= 1.0e-10 {
                    f32::NEG_INFINITY
                } else {
                    20.0 * magnitude.log10()
                };
            }
        },
    );

    if cancelled.is_some_and(|token| token.load(Ordering::Relaxed)) {
        return None;
    }

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
    // Keep a bounded peak set so sensitivity can move the threshold down from
    // the default without retaining every noise local-maximum in memory.
    let peaks = build_peak_map(&matrix, n_freq, n_time, db_min - 40.0);
    Some(Spectrogram {
        matrix,
        n_freq,
        n_time,
        n_fft,
        hop_length: hop,
        sample_rate,
        db_min,
        db_max,
        peaks,
    })
}

pub(crate) fn build_peak_map(
    matrix: &[f32],
    n_freq: usize,
    n_time: usize,
    threshold_db: f32,
) -> Vec<Vec<usize>> {
    (0..n_time)
        .into_par_iter()
        .map(|time| {
            if n_freq < 3 {
                return Vec::new();
            }
            let mut peaks = Vec::new();
            for frequency in 1..n_freq - 1 {
                let value = matrix[frequency * n_time + time];
                if value >= threshold_db
                    && value > matrix[(frequency - 1) * n_time + time]
                    && value > matrix[(frequency + 1) * n_time + time]
                {
                    peaks.push(frequency);
                }
            }
            peaks
        })
        .collect()
}

pub(crate) fn rebuild_peak_map(spectrogram: &mut Spectrogram, floor_offset_db: f32) {
    spectrogram.peaks = build_peak_map(
        &spectrogram.matrix,
        spectrogram.n_freq,
        spectrogram.n_time,
        spectrogram.db_min + floor_offset_db,
    );
}

pub(crate) fn rasterize_stored_mask(
    first_source_frame: usize,
    source_ranges: &[(usize, usize)],
    source_sample_rate: u32,
    source_hop: usize,
    source_n_fft: usize,
    target_frames: usize,
    target_freq_bins: usize,
    target_sample_rate: u32,
    target_hop: usize,
    target_n_fft: usize,
) -> Vec<Vec<(usize, usize)>> {
    if source_ranges.is_empty()
        || target_frames == 0
        || target_freq_bins == 0
        || source_sample_rate == 0
        || target_sample_rate == 0
    {
        return Vec::new();
    }
    (0..target_frames)
        .map(|target_frame| {
            let time = target_frame as f64 * target_hop as f64 / target_sample_rate as f64;
            let source_position = time * source_sample_rate as f64 / source_hop.max(1) as f64;
            let source_index = source_position.round() as isize - first_source_frame as isize;
            if source_index < 0 || source_index as usize >= source_ranges.len() {
                return Vec::new();
            }
            let (source_low, source_high) = source_ranges[source_index as usize];
            let hz_per_source_bin = source_sample_rate as f64 / source_n_fft.max(1) as f64;
            let hz_low = source_low as f64 * hz_per_source_bin;
            let hz_high = (source_high + 1) as f64 * hz_per_source_bin;
            let first_bin =
                (hz_low * target_n_fft as f64 / target_sample_rate as f64).ceil() as usize;
            let last_bin = ((hz_high * target_n_fft as f64 / target_sample_rate as f64).floor()
                - 1.0)
                .max(0.0) as usize;
            let last_bin = last_bin.min(target_freq_bins.saturating_sub(1));
            if first_bin <= last_bin && first_bin < target_freq_bins {
                vec![(first_bin, last_bin)]
            } else {
                Vec::new()
            }
        })
        .collect()
}

/// Grow a peak seed into a variable-width mask by tracking nearby frame peaks
/// and expanding each row to valleys. `max_valley_crossings=1` selects one lobe.
pub fn grow_peak_mask(
    spectrogram: &Spectrogram,
    time: f32,
    frequency_hz: f32,
    sensitivity_db: f32,
    max_valley_crossings: usize,
) -> Option<(usize, Vec<(usize, usize)>)> {
    if spectrogram.peaks.is_empty() || spectrogram.n_freq < 3 {
        return None;
    }
    let hover_frame = (time * spectrogram.sample_rate as f32 / spectrogram.hop_length.max(1) as f32)
        .round() as usize;
    let hover_frame = hover_frame.min(spectrogram.n_time.saturating_sub(1));
    let hover_bin = spectrogram.hz_to_bin(frequency_hz);
    let min_frame = hover_frame.saturating_sub(4);
    let max_frame = hover_frame
        .saturating_add(4)
        .min(spectrogram.n_time.saturating_sub(1));
    // Larger UI sensitivity lowers the threshold and grows a larger region.
    let threshold = spectrogram.db_min + (30.0 - sensitivity_db.clamp(0.0, 45.0));

    let seed = (min_frame..=max_frame)
        .flat_map(|frame| {
            spectrogram
                .peaks
                .get(frame)
                .into_iter()
                .flatten()
                .map(move |&bin| {
                    let value = spectrogram.at(bin, frame);
                    (frame, bin, value)
                })
        })
        .filter(|(_, _, value)| *value >= threshold)
        .min_by_key(|(frame, bin, _)| frame.abs_diff(hover_frame) * 2 + bin.abs_diff(hover_bin))?;

    let track = |direction: isize| -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let mut frame = seed.0 as isize;
        let mut center = seed.1;
        loop {
            frame += direction;
            if frame < 0 || frame >= spectrogram.n_time as isize {
                break;
            }
            let frame = frame as usize;
            let Some(candidates) = spectrogram.peaks.get(frame) else {
                break;
            };
            let next = candidates
                .iter()
                .copied()
                .filter(|bin| bin.abs_diff(center) <= (center / 16).max(8))
                .max_by(|a, b| {
                    spectrogram
                        .at(*a, frame)
                        .total_cmp(&spectrogram.at(*b, frame))
                });
            let Some(next) = next else { break };
            if spectrogram.at(next, frame) < threshold {
                break;
            }
            center = next;
            result.push((frame, center));
        }
        result
    };
    let left = track(-1);
    let right = track(1);
    let first_frame = left.last().map_or(seed.0, |point| point.0);
    let last_frame = right.last().map_or(seed.0, |point| point.0);
    let mut centers = vec![seed.1; last_frame - first_frame + 1];
    centers[seed.0 - first_frame] = seed.1;
    for (frame, bin) in left.into_iter().chain(right) {
        centers[frame - first_frame] = bin;
    }

    let max_crossings = max_valley_crossings.clamp(1, 8);
    let mut ranges = Vec::with_capacity(centers.len());
    for (offset, center) in centers.iter().copied().enumerate() {
        let frame = first_frame + offset;
        let row = &spectrogram.matrix;
        let mut low = center;
        let mut high = center;
        let mut low_valleys = 0;
        let mut high_valleys = 0;
        let max_half_width = (spectrogram.n_freq / 8).max(16);
        while low > 0 && center - low < max_half_width && low_valleys < max_crossings {
            let candidate = low - 1;
            let value = row[candidate * spectrogram.n_time + frame];
            if candidate > 0
                && value <= row[(candidate - 1) * spectrogram.n_time + frame]
                && value <= row[(candidate + 1) * spectrogram.n_time + frame]
            {
                low_valleys += 1;
            }
            low = candidate;
            if value < threshold && low_valleys >= max_crossings {
                break;
            }
        }
        while high + 1 < spectrogram.n_freq
            && high - center < max_half_width
            && high_valleys < max_crossings
        {
            let candidate = high + 1;
            let value = row[candidate * spectrogram.n_time + frame];
            if candidate + 1 < spectrogram.n_freq
                && value <= row[(candidate - 1) * spectrogram.n_time + frame]
                && value <= row[(candidate + 1) * spectrogram.n_time + frame]
            {
                high_valleys += 1;
            }
            high = candidate;
            if value < threshold && high_valleys >= max_crossings {
                break;
            }
        }
        ranges.push((low, high));
    }
    Some((first_frame, ranges))
}

/// 5th/95th percentile of the dB distribution, used to normalise the colormap.
///
/// This used to clone the whole matrix and fully sort it — O(n log n) over tens
/// of millions of floats, and it was the dominant tail of load time on long
/// tracks. A fixed-bin histogram reduced in parallel gives the same percentiles
/// for colour-mapping purposes in O(n), which is plenty accurate here: the
/// values only drive a normalisation range, not audio.
pub(crate) fn compute_db_range(matrix: &[f32]) -> (f32, f32) {
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
    let mut histogram_min = min;
    let mut histogram_max = max;
    if histogram_min == f32::NEG_INFINITY {
        histogram_min = -180.0;
    }
    if !histogram_max.is_finite() {
        histogram_max = 0.0;
    }
    let range = histogram_max - histogram_min;
    if !range.is_finite() || range < 1.0e-6 {
        return (histogram_min, histogram_max);
    }
    let scale = (BINS - 1) as f32 / range;
    let chunk = (matrix.len() / (rayon::current_num_threads() * 4).max(1) + 1).max(1);
    let total: Vec<u64> = matrix
        .par_chunks(chunk)
        .map(|chunk| {
            let mut histogram = vec![0u64; BINS];
            for value in chunk {
                let finite_value = if value.is_finite() {
                    *value
                } else {
                    histogram_min
                };
                let bin = (((finite_value - histogram_min) * scale).round() as usize).min(BINS - 1);
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
                return histogram_min + bin as f32 / scale;
            }
        }
        histogram_max
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
    fn peak_mask_sensitivity_is_monotonic() {
        let sample_rate = 8_000;
        let frames = 16_000;
        let audio = AudioData {
            samples: (0..frames)
                .map(|index| {
                    let time = index as f32 / sample_rate as f32;
                    let envelope = if time < 0.3 {
                        time / 0.3
                    } else if time > 1.2 {
                        ((1.6 - time) / 0.4).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    envelope * (2.0 * std::f32::consts::PI * 1_000.0 * time).sin()
                })
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        };
        let spec = compute(&audio, 1024, 128);
        let low = grow_peak_mask(&spec, 0.8, 1_000.0, 0.0, 1).unwrap();
        let high = grow_peak_mask(&spec, 0.8, 1_000.0, 45.0, 1).unwrap();
        assert!(high.1.len() >= low.1.len());
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
                let magnitude = frame[frequency].norm();
                let db = if magnitude <= 1.0e-10 {
                    f32::NEG_INFINITY
                } else {
                    20.0 * magnitude.log10()
                };
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
    fn incremental_patch_matches_columns_from_full_transform() {
        let channels = 2;
        let rate = 8_000;
        let frames = 4096;
        let samples: Vec<f32> = (0..frames * channels)
            .map(|i| (i as f32 * 0.071).sin() * 0.4)
            .collect();
        let full = compute_interleaved(&samples, channels as u16, rate, 256, 64);
        let patch = compute_interleaved_patch(&samples, channels as u16, rate, 256, 64, 700, 1200);
        assert!(patch.n_time > 0);
        for frequency in 0..patch.n_freq {
            for local_time in 0..patch.n_time {
                let expected = full.at(frequency, patch.first_time_frame + local_time);
                let actual = patch.matrix[frequency * patch.n_time + local_time];
                if expected.is_infinite() {
                    assert_eq!(actual, expected);
                } else {
                    assert!((actual - expected).abs() < 1.0e-4);
                }
            }
        }
    }

    #[test]
    fn exact_silence_is_negative_infinity_db() {
        let audio = AudioData {
            samples: vec![0.0; 4096],
            sample_rate: 44_100,
            channels: 1,
            duration: 4096.0 / 44_100.0,
        };
        let spec = compute(&audio, 256, 64);
        assert!(spec.matrix.iter().all(|value| *value == f32::NEG_INFINITY));
        assert!(spec.db_min.is_finite());
        assert!(spec.db_max.is_finite());
    }

    #[test]
    fn expanded_frame_span_covers_overlapping_fft_windows() {
        let changed = (5000, 6000);
        let expanded = expanded_patch_frames(changed, 100, 52_736, 2048, 512, 8192);
        let first_fft = if expanded.0 < 2048 {
            0
        } else {
            (expanded.0 - 2048) / 512 + 1
        };
        let last_fft = (expanded.1 - 1) / 512 + 1;
        assert!(first_fft * 512 < changed.0);
        assert!((last_fft - 1) * 512 + 2048 >= changed.1);
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
