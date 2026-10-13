//! Exact offline processing — the export path.
//!
//! Export keeps the overlap-add STFT with brick-wall bin masking so the written
//! WAV matches the region rectangles precisely. The fast preview path lives in
//! `audio::preview` and uses IIR bands instead; do not "optimise" this module by
//! routing it through the preview renderer.
//!
//! Two things were done to make export fast without changing its output:
//!   * the mask is precomputed as per-selection frame/bin ranges, so the inner
//!     loop no longer tests every selection against every bin (it used to be
//!     O(n_time * n_freq * n_selections) float comparisons);
//!   * frames are processed in parallel blocks with rayon, each block doing its
//!     own overlap-add into a local buffer, then merged.

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use super::spectrogram::rasterize_stored_mask;
use super::types::{hann_window, AudioData, Selection, SelectionAction, SelectionShape};

pub fn apply_selections(audio: &AudioData, selections: &[Selection]) -> Vec<f32> {
    let enabled = selections
        .iter()
        .filter(|selection| selection.enabled)
        .cloned()
        .collect::<Vec<_>>();
    if enabled.len() != selections.len() {
        return apply_selections(audio, &enabled);
    }
    if selections.is_empty() {
        return audio.samples.clone();
    }
    if selections
        .iter()
        .all(|selection| selection.full_band && !selection.requires_spectral_processing())
    {
        apply_time_domain(audio, selections)
    } else {
        apply_spectral_masks(audio, selections, 2048, 512)
    }
}

/// Remove leading/trailing silence while preserving interleaved channel frames.
/// `threshold` is a linear peak-amplitude threshold (0.001 is approximately -60 dBFS).
pub fn trim_silence(samples: &[f32], channels: u16, threshold: f32) -> Vec<f32> {
    let channels = channels as usize;
    if channels == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    let frames = samples.len() / channels;
    let threshold = threshold.max(0.0);
    let is_silent = |frame: usize| {
        samples[frame * channels..(frame + 1) * channels]
            .iter()
            .all(|sample| sample.abs() <= threshold)
    };
    let first = (0..frames).find(|frame| !is_silent(*frame));
    let Some(first) = first else {
        return Vec::new();
    };
    let last = (first..frames)
        .rev()
        .find(|frame| !is_silent(*frame))
        .unwrap_or(first);
    samples[first * channels..(last + 1) * channels].to_vec()
}

pub fn apply_time_domain(audio: &AudioData, selections: &[Selection]) -> Vec<f32> {
    let enabled = selections
        .iter()
        .filter(|selection| selection.enabled)
        .cloned()
        .collect::<Vec<_>>();
    if enabled.len() != selections.len() {
        return apply_time_domain(audio, &enabled);
    }
    if selections
        .iter()
        .any(Selection::requires_spectral_processing)
    {
        return apply_spectral_masks(audio, selections, 2048, 512);
    }
    let channels = audio.channels as usize;
    if channels == 0 {
        return audio.samples.clone();
    }
    let has_solo = selections
        .iter()
        .any(|selection| selection.action == SelectionAction::Solo);
    let sample_rate = audio.sample_rate.max(1) as f32;

    // Resolve seconds to frame bounds once instead of per frame per selection.
    let resolved: Vec<ResolvedRegion> =
        selections.iter().map(|s| resolve(s, sample_rate)).collect();

    let frames = audio.frame_count();
    let mut output = audio.samples.clone();
    let block = frame_block(frames);
    output
        .par_chunks_mut(block * channels)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let mut frame_index = block_index * block;
            for frame in chunk.chunks_exact_mut(channels) {
                let time = frame_index as f32 / sample_rate;
                let in_solo = resolved.iter().any(|region| {
                    region.action == SelectionAction::Solo && region.contains_time(time)
                });
                if has_solo && !in_solo {
                    frame.fill(0.0);
                } else {
                    // Solo is exclusive: ignore Gain actions whenever a Solo
                    // selection exists; outside Solo windows is already silent.
                    let gain = if has_solo {
                        1.0
                    } else {
                        resolved
                            .iter()
                            .filter(|region| {
                                region.action == SelectionAction::Gain && region.contains_time(time)
                            })
                            .map(|region| region.gain)
                            .product::<f32>()
                    };
                    for sample in frame {
                        *sample *= gain;
                    }
                }
                frame_index += 1;
            }
        });
    output
}

/// A selection with its time bounds kept in seconds (the original predicate
/// compares `frame_time` against seconds, so keeping seconds preserves the
/// exact boundary behaviour) and its gain pre-clamped.
struct ResolvedRegion {
    action: SelectionAction,
    time_start: f32,
    time_end: f32,
    /// Full frequency range, or the band when not `full_band`.
    full_band: bool,
    freq_low: f32,
    freq_high: f32,
    gain: f32,
}

impl ResolvedRegion {
    fn contains_time(&self, time: f32) -> bool {
        time >= self.time_start && time <= self.time_end
    }

    fn contains_bin(&self, frequency: f32) -> bool {
        self.full_band || (frequency >= self.freq_low && frequency <= self.freq_high)
    }
}

fn resolve(selection: &Selection, _sample_rate: f32) -> ResolvedRegion {
    ResolvedRegion {
        action: selection.action,
        time_start: selection.time_start,
        time_end: selection.time_end,
        full_band: selection.full_band,
        freq_low: selection.freq_low,
        freq_high: selection.freq_high,
        gain: selection.gain_linear(),
    }
}

/// Frames per rayon work block: enough FFTs per task to amortise scheduling,
/// enough blocks to saturate the cores.
fn frame_block(total: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (total / (threads * 4)).max(256).min(total.max(1)).max(1)
}

/// Inclusive frame range over which `region` is active, computed with the exact
/// same predicate the sequential version used.
///
/// The estimate is corrected by evaluating the real predicate, so a boundary
/// that lands exactly on a frame time behaves identically to before.
fn active_frame_range(
    region: &ResolvedRegion,
    n_time: usize,
    hop: usize,
    sample_rate: f32,
) -> Option<(usize, usize)> {
    if n_time == 0 {
        return None;
    }
    let frames_per_second = sample_rate / hop as f32;
    let estimate_low = (region.time_start * frames_per_second).floor();
    let estimate_high = (region.time_end * frames_per_second).ceil();
    let mut low = estimate_low.clamp(0.0, n_time as f32) as usize;
    let mut high = estimate_high.clamp(0.0, (n_time - 1) as f32) as usize;

    let time_at = |frame: usize| frame as f32 * hop as f32 / sample_rate;
    // Walk inward to the first/last frame that actually satisfies the predicate.
    while low < n_time && !region.contains_time(time_at(low)) {
        low += 1;
    }
    if low >= n_time {
        return None;
    }
    while high > low && !region.contains_time(time_at(high)) {
        high -= 1;
    }
    if !region.contains_time(time_at(high)) {
        return None;
    }
    Some((low, high))
}

/// Inclusive bin range over which `region` applies, using the exact predicate
/// `frequency >= freq_low && frequency <= freq_high`.
fn active_bin_range(
    region: &ResolvedRegion,
    n_freq: usize,
    n_fft: usize,
    sample_rate: f32,
) -> Option<(usize, usize)> {
    if region.full_band {
        return Some((0, n_freq - 1));
    }
    if n_freq == 0 {
        return None;
    }
    let bins_per_hz = n_fft as f32 / sample_rate;
    let estimate_low = (region.freq_low * bins_per_hz).floor();
    let estimate_high = (region.freq_high * bins_per_hz).ceil();
    let mut low = estimate_low.clamp(0.0, (n_freq - 1) as f32) as usize;
    let mut high = estimate_high.clamp(0.0, (n_freq - 1) as f32) as usize;

    let hz_at = |bin: usize| bin as f32 * sample_rate / n_fft as f32;
    while low < n_freq && !region.contains_bin(hz_at(low)) {
        low += 1;
    }
    if low >= n_freq {
        return None;
    }
    while high > low && !region.contains_bin(hz_at(high)) {
        high -= 1;
    }
    if !region.contains_bin(hz_at(high)) {
        return None;
    }
    Some((low, high))
}

/// Per-selection mask geometry: which frames and which bins it touches.
struct MaskRegion {
    action: SelectionAction,
    frame_start: usize,
    frame_end: usize,
    bins: MaskBins,
    gain: f32,
}

enum MaskBins {
    Rect(Vec<(usize, usize)>),
    Polygon(Vec<Vec<(usize, usize)>>),
}

impl MaskRegion {
    fn bins_at(&self, time_frame: usize) -> &[(usize, usize)] {
        match &self.bins {
            MaskBins::Rect(intervals) => intervals,
            MaskBins::Polygon(rows) => rows
                .get(time_frame.saturating_sub(self.frame_start))
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rasterize_polygon(
    points: &[(f32, f32)],
    first_frame: usize,
    last_frame: usize,
    n_time: usize,
    n_freq: usize,
    n_fft: usize,
    hop: usize,
    sample_rate: f32,
) -> Vec<Vec<(usize, usize)>> {
    if points.len() < 3 || n_freq == 0 || n_time == 0 {
        return Vec::new();
    }
    let mut rows = Vec::with_capacity(last_frame.saturating_sub(first_frame) + 1);
    for time_frame in first_frame..=last_frame.min(n_time.saturating_sub(1)) {
        let time = time_frame as f32 * hop as f32 / sample_rate;
        let mut intersections = Vec::new();
        for index in 0..points.len() {
            let (x1, y1) = points[index];
            let (x2, y2) = points[(index + 1) % points.len()];
            if (x1 <= time && time < x2) || (x2 <= time && time < x1) {
                let fraction = (time - x1) / (x2 - x1);
                intersections.push(y1 + fraction * (y2 - y1));
            }
        }
        intersections.sort_by(f32::total_cmp);
        let mut intervals = Vec::new();
        for pair in intersections.chunks_exact(2) {
            let first_bin = (pair[0] * n_fft as f32 / sample_rate).ceil() as isize;
            let last_bin = (pair[1] * n_fft as f32 / sample_rate).floor() as isize;
            let first_bin = first_bin.clamp(0, n_freq.saturating_sub(1) as isize) as usize;
            let last_bin = last_bin.clamp(0, n_freq.saturating_sub(1) as isize) as usize;
            if first_bin <= last_bin {
                intervals.push((first_bin, last_bin));
            }
        }
        rows.push(intervals);
    }
    rows
}

pub fn apply_spectral_masks(
    audio: &AudioData,
    selections: &[Selection],
    n_fft: usize,
    hop: usize,
) -> Vec<f32> {
    let enabled = selections
        .iter()
        .filter(|selection| selection.enabled)
        .cloned()
        .collect::<Vec<_>>();
    if enabled.len() != selections.len() {
        return apply_spectral_masks(audio, &enabled, n_fft, hop);
    }
    let channels = audio.channels as usize;
    if channels == 0 || n_fft < 2 || hop == 0 || audio.samples.is_empty() {
        return audio.samples.clone();
    }
    let input_frames = audio.frame_count();
    if input_frames < n_fft {
        if selections
            .iter()
            .any(Selection::requires_spectral_processing)
        {
            return audio.samples.clone();
        }
        return apply_time_domain(audio, selections);
    }
    let n_freq = n_fft / 2 + 1;
    let n_time = (input_frames - n_fft) / hop + 1;
    let sample_rate = audio.sample_rate.max(1) as f32;
    let window = hann_window(n_fft);
    let has_solo = selections
        .iter()
        .any(|selection| selection.action == SelectionAction::Solo);

    // Precompute mask geometry; regions that touch no frame or no bin drop out.
    let mask_regions: Vec<MaskRegion> = selections
        .iter()
        .filter_map(|selection| {
            let region = resolve(selection, sample_rate);
            let frames = active_frame_range(&region, n_time, hop, sample_rate)?;
            let bins = match &selection.shape {
                SelectionShape::Rect => {
                    let (first, last) = active_bin_range(&region, n_freq, n_fft, sample_rate)?;
                    MaskBins::Rect(vec![(first, last)])
                }
                SelectionShape::Polygon { points } => MaskBins::Polygon(rasterize_polygon(
                    points,
                    frames.0,
                    frames.1,
                    n_time,
                    n_freq,
                    n_fft,
                    hop,
                    sample_rate,
                )),
                SelectionShape::Mask {
                    first_frame,
                    ranges,
                    sample_rate: source_rate,
                    hop_length: source_hop,
                    n_fft: source_fft,
                } => {
                    let rasterized = rasterize_stored_mask(
                        *first_frame,
                        ranges,
                        *source_rate,
                        *source_hop,
                        *source_fft,
                        n_time,
                        n_freq,
                        audio.sample_rate,
                        hop,
                        n_fft,
                    );
                    MaskBins::Polygon(rasterized.get(frames.0..=frames.1).unwrap_or(&[]).to_vec())
                }
            };
            Some(MaskRegion {
                action: region.action,
                frame_start: frames.0,
                frame_end: frames.1,
                bins,
                gain: region.gain,
            })
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n_fft);
    let inverse = planner.plan_fft_inverse(n_fft);
    // Channels are independent; blocks within a channel are independent too.
    // Each block overlap-adds into its own local buffers covering the samples it
    // touches, then the locals are summed. Per-channel results come back as
    // separate buffers because a parallel closure cannot share one `&mut` output.
    let block = frame_block(n_time);
    let block_count = n_time.div_ceil(block);

    let channel_buffers: Vec<Vec<f32>> = (0..channels)
        .into_par_iter()
        .map(|channel| {
            let partials: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..block_count)
                .into_par_iter()
                .map(|block_index| {
                    let frame_start = block_index * block;
                    let frame_end = (frame_start + block).min(n_time);
                    // Sample span this block writes into: the first frame's start
                    // through the last frame's start + n_fft.
                    let sample_start = frame_start * hop;
                    let sample_end = ((frame_end - 1) * hop + n_fft).min(input_frames);
                    let length = sample_end.saturating_sub(sample_start);
                    let mut accumulated = vec![0.0f32; length];
                    let mut weights = vec![0.0f32; length];
                    let mut spectrum = vec![Complex32::default(); n_fft];
                    let mut keep = vec![true; n_freq];
                    let mut gains = vec![1.0f32; n_freq];

                    for time_frame in frame_start..frame_end {
                        let start = time_frame * hop;
                        for index in 0..n_fft {
                            let sample = audio.samples[(start + index) * channels + channel];
                            spectrum[index] = Complex32::new(sample * window[index], 0.0);
                        }
                        forward.process(&mut spectrum);

                        // Build this frame's mask from only the regions active here.
                        for value in keep.iter_mut() {
                            *value = !has_solo;
                        }
                        for value in gains.iter_mut() {
                            *value = 1.0;
                        }
                        for region in mask_regions.iter() {
                            if time_frame < region.frame_start || time_frame > region.frame_end {
                                continue;
                            }
                            match region.action {
                                SelectionAction::Solo => {
                                    if has_solo {
                                        for (first, last) in region.bins_at(time_frame) {
                                            for slot in keep[*first..=*last].iter_mut() {
                                                *slot = true;
                                            }
                                        }
                                    }
                                }
                                SelectionAction::Gain => {
                                    if !has_solo {
                                        for (first, last) in region.bins_at(time_frame) {
                                            for slot in gains[*first..=*last].iter_mut() {
                                                *slot *= region.gain;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for bin in 0..n_freq {
                            if !keep[bin] {
                                spectrum[bin] = Complex32::default();
                                if bin > 0 && bin < n_fft / 2 {
                                    spectrum[n_fft - bin] = Complex32::default();
                                }
                            } else if (gains[bin] - 1.0).abs() > 1.0e-9 {
                                let gain = gains[bin];
                                spectrum[bin] *= gain;
                                if bin > 0 && bin < n_fft / 2 {
                                    spectrum[n_fft - bin] *= gain;
                                }
                            }
                        }

                        inverse.process(&mut spectrum);
                        let base = start - sample_start;
                        for index in 0..n_fft {
                            let slot = base + index;
                            if slot < length {
                                accumulated[slot] +=
                                    spectrum[index].re / n_fft as f32 * window[index];
                                weights[slot] += window[index] * window[index];
                            }
                        }
                    }
                    (sample_start, accumulated, weights)
                })
                .collect();

            // Merge the block-local overlap-add results. Sequential and linear in
            // signal length, negligible next to the FFT work done above.
            let mut accumulated = vec![0.0f32; input_frames];
            let mut weights = vec![0.0f32; input_frames];
            for (sample_start, local_accumulated, local_weights) in partials {
                for (offset, value) in local_accumulated.into_iter().enumerate() {
                    let index = sample_start + offset;
                    if index < input_frames {
                        accumulated[index] += value;
                    }
                }
                for (offset, value) in local_weights.into_iter().enumerate() {
                    let index = sample_start + offset;
                    if index < input_frames {
                        weights[index] += value;
                    }
                }
            }

            (0..input_frames)
                .map(|index| {
                    if weights[index] > 1.0e-8 {
                        accumulated[index] / weights[index]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    // Re-interleave the per-channel results into the output buffer.
    let mut output = vec![0.0f32; audio.samples.len()];
    for (index, frame) in output.chunks_exact_mut(channels).enumerate() {
        for (channel, value) in frame.iter_mut().enumerate() {
            *value = channel_buffers
                .get(channel)
                .and_then(|buffer| buffer.get(index))
                .copied()
                .unwrap_or(0.0);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_audio() -> AudioData {
        let sample_rate = 1000;
        let samples = (0..2000)
            .map(|index| (index as f32 * 0.1).sin())
            .collect::<Vec<_>>();
        AudioData {
            samples,
            sample_rate,
            channels: 1,
            duration: 2.0,
        }
    }

    #[test]
    fn zero_gain_and_solo_are_applied() {
        let audio = sample_audio();
        let mut gain = Selection::new(1, 0.5, 1.0, 0.0, 500.0, SelectionAction::Gain, true);
        gain.gain_db = -60.0;
        let zeroed = apply_time_domain(&audio, &[gain]);
        assert!(zeroed[500..1000].iter().all(|sample| *sample == 0.0));
        assert!(zeroed[0..400].iter().any(|sample| *sample != 0.0));

        let solo = Selection::new(2, 0.5, 1.0, 0.0, 500.0, SelectionAction::Solo, true);
        let soloed = apply_time_domain(&audio, &[solo]);
        assert!(soloed[0..500].iter().all(|sample| *sample == 0.0));
        assert!(soloed[500..1000].iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn gain_region_applies_db_converted_multiplier() {
        let audio = sample_audio();
        let mut gain = Selection::new(1, 0.5, 1.0, 0.0, 500.0, SelectionAction::Gain, true);
        gain.gain_db = 20.0 * 0.5_f32.log10();
        let output = apply_time_domain(&audio, &[gain]);
        for (index, sample) in output.iter().enumerate().take(1000).skip(500) {
            assert!((*sample - audio.samples[index] * 0.5).abs() < 1.0e-6);
        }
        assert_eq!(output[100], audio.samples[100]);
    }

    #[test]
    fn spectral_processing_preserves_length_and_finite_values() {
        let audio = sample_audio();
        let mut gain = Selection::new(1, 0.2, 1.5, 50.0, 200.0, SelectionAction::Gain, false);
        gain.gain_db = -60.0;
        let output = apply_spectral_masks(&audio, &[gain], 256, 64);
        assert_eq!(output.len(), audio.samples.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn polygon_mask_attenuates_only_bins_inside_sloped_region() {
        let sample_rate = 8_000;
        let frames = 40_000;
        let audio = AudioData {
            samples: (0..frames)
                .map(|index| {
                    let time = index as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * 700.0 * time).sin()
                        + (2.0 * std::f32::consts::PI * 1_800.0 * time).sin()
                })
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        };
        let polygon = Selection::polygon(
            99,
            vec![(0.2, 500.0), (1.4, 2_200.0), (1.4, 2_500.0), (0.2, 800.0)],
            SelectionAction::Gain,
        );
        let mut polygon = polygon;
        polygon.gain_db = -60.0;
        let output = apply_selections(&audio, &[polygon]);
        assert_eq!(output.len(), audio.samples.len());
        assert!(output.iter().all(|sample| sample.is_finite()));

        let original_rms = |samples: &[f32]| {
            (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32)
                .sqrt()
        };
        let inside = 7_200..9_600;
        assert!(original_rms(&output[inside.clone()]) < original_rms(&audio.samples[inside]));
        let outside = 16_000..24_000;
        assert!(
            (original_rms(&output[outside.clone()]) - original_rms(&audio.samples[outside])).abs()
                < 2.0e-2
        );
    }

    #[test]
    fn stored_mask_attenuates_only_its_time_frequency_region() {
        let sample_rate = 8_000;
        let frames = 16_000;
        let audio = AudioData {
            samples: (0..frames)
                .map(|index| {
                    (2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / sample_rate as f32).sin()
                })
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        };
        let mut mask = Selection::mask(
            7,
            4,
            vec![(255, 257); 9],
            512,
            sample_rate,
            2048,
            SelectionAction::Gain,
        );
        mask.gain_db = f32::NEG_INFINITY;
        let output = apply_selections(&audio, &[mask]);
        let rms = |samples: &[f32]| {
            (samples.iter().map(|value| value * value).sum::<f32>() / samples.len() as f32).sqrt()
        };
        assert!(rms(&output[4_500..6_000]) < rms(&audio.samples[4_500..6_000]) * 0.5);
        assert!((rms(&output[10_000..12_000]) - rms(&audio.samples[10_000..12_000])).abs() < 1e-3);
    }

    /// The precomputed frame/bin ranges must reproduce the current per-bin
    /// predicate evaluation. This reimplementation is kept as a reference oracle.
    fn reference_masks(
        audio: &AudioData,
        selections: &[Selection],
        n_fft: usize,
        hop: usize,
    ) -> Vec<f32> {
        let channels = audio.channels as usize;
        let input_frames = audio.frame_count();
        let n_freq = n_fft / 2 + 1;
        let n_time = (input_frames - n_fft) / hop + 1;
        let window = hann_window(n_fft);
        let has_solo = selections.iter().any(|s| s.action == SelectionAction::Solo);
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(n_fft);
        let inverse = planner.plan_fft_inverse(n_fft);
        let mut output = vec![0.0f32; audio.samples.len()];

        for channel in 0..channels {
            let mut accumulated = vec![0.0f32; input_frames];
            let mut window_weights = vec![0.0f32; input_frames];
            let mut spectrum = vec![Complex32::default(); n_fft];
            for time_frame in 0..n_time {
                let start = time_frame * hop;
                let time = time_frame as f32 * hop as f32 / audio.sample_rate.max(1) as f32;
                for index in 0..n_fft {
                    let sample = audio.samples[(start + index) * channels + channel];
                    spectrum[index] = Complex32::new(sample * window[index], 0.0);
                }
                forward.process(&mut spectrum);
                for bin in 0..n_freq {
                    let frequency = bin as f32 * audio.sample_rate as f32 / n_fft as f32;
                    let in_bounds = |selection: &Selection| {
                        time >= selection.time_start
                            && time <= selection.time_end
                            && (selection.full_band
                                || (frequency >= selection.freq_low
                                    && frequency <= selection.freq_high))
                    };
                    let in_solo = selections
                        .iter()
                        .filter(|s| s.action == SelectionAction::Solo)
                        .any(in_bounds);
                    let keep = if has_solo { in_solo } else { true };
                    if !keep {
                        spectrum[bin] = Complex32::default();
                        if bin > 0 && bin < n_fft / 2 {
                            spectrum[n_fft - bin] = Complex32::default();
                        }
                    } else {
                        let gain = if has_solo {
                            1.0
                        } else {
                            selections
                                .iter()
                                .filter(|s| s.action == SelectionAction::Gain)
                                .filter(|s| in_bounds(s))
                                .map(Selection::gain_linear)
                                .product::<f32>()
                        };
                        spectrum[bin] *= gain;
                        if bin > 0 && bin < n_fft / 2 {
                            spectrum[n_fft - bin] *= gain;
                        }
                    }
                }
                inverse.process(&mut spectrum);
                for index in 0..n_fft {
                    let sample_index = start + index;
                    accumulated[sample_index] += spectrum[index].re / n_fft as f32 * window[index];
                    window_weights[sample_index] += window[index] * window[index];
                }
            }
            for index in 0..input_frames {
                output[index * channels + channel] = if window_weights[index] > 1.0e-8 {
                    accumulated[index] / window_weights[index]
                } else {
                    0.0
                };
            }
        }
        output
    }

    fn noisy_audio(frames: usize, sample_rate: u32, channels: u16) -> AudioData {
        // Deterministic pseudo-noise plus two tones, so mask errors show up.
        let mut state = 12345u64;
        let samples: Vec<f32> = (0..frames * channels as usize)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as f32 / (1u64 << 31) as f32) * 0.2 - 0.1;
                let t = (i / channels as usize) as f32 / sample_rate as f32;
                noise
                    + (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.5
                    + (2.0 * std::f32::consts::PI * 2500.0 * t).sin() * 0.5
            })
            .collect();
        AudioData {
            samples,
            sample_rate,
            channels,
            duration: frames as f32 / sample_rate as f32,
        }
    }

    #[test]
    fn parallel_masks_match_sequential_reference_zero_gain() {
        let audio = noisy_audio(8_000, 8_000, 1);
        let selections = vec![Selection::new(
            1,
            0.2,
            0.7,
            200.0,
            900.0,
            SelectionAction::Gain,
            false,
        )];
        let got = apply_spectral_masks(&audio, &selections, 256, 64);
        let want = reference_masks(&audio, &selections, 256, 64);
        assert_eq!(got.len(), want.len());
        let max_err = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1.0e-4, "zero-gain band drift {max_err}");
    }

    #[test]
    fn solo_ignores_other_gain_regions() {
        let audio = noisy_audio(8_000, 8_000, 2);
        let selections = vec![
            Selection::new(1, 0.1, 0.6, 250.0, 1_500.0, SelectionAction::Solo, false),
            Selection::new(2, 0.3, 0.9, 100.0, 3_000.0, SelectionAction::Gain, false),
        ];
        let mut selections = selections;
        selections[1].gain_db = 20.0 * 1.5_f32.log10();
        let got = apply_spectral_masks(&audio, &selections, 256, 64);
        let solo_only = apply_spectral_masks(&audio, &selections[..1], 256, 64);
        assert_eq!(got, solo_only);
    }

    #[test]
    fn masks_match_reference_when_region_edges_land_on_frame_boundaries() {
        // hop=64 at 8 kHz -> one frame is exactly 8 ms; pick edges on that grid
        // so the inclusive boundary logic is exercised.
        let audio = noisy_audio(4_000, 8_000, 1);
        let mut selection = Selection::new(
            1,
            0.008 * 10.0,
            0.008 * 40.0,
            8_000.0 / 256.0 * 20.0,
            8_000.0 / 256.0 * 60.0,
            SelectionAction::Gain,
            false,
        );
        selection.gain_db = -60.0;
        let selections = vec![selection];
        let got = apply_spectral_masks(&audio, &selections, 256, 64);
        let want = reference_masks(&audio, &selections, 256, 64);
        let max_err = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1.0e-4, "boundary-aligned drift {max_err}");
    }

    #[test]
    fn region_outside_the_track_is_a_no_op() {
        let audio = noisy_audio(4_000, 8_000, 1);
        let selections = vec![Selection::new(
            1,
            5.0,
            9.0,
            100.0,
            500.0,
            SelectionAction::Gain,
            false,
        )];
        let got = apply_spectral_masks(&audio, &selections, 256, 64);
        let want = reference_masks(&audio, &selections, 256, 64);
        let max_err = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1.0e-5,
            "out-of-range region changed output: {max_err}"
        );
    }

    #[test]
    fn time_domain_parallel_matches_sequential() {
        let audio = noisy_audio(20_000, 44_100, 2);
        let mut zero_gain_region =
            Selection::new(1, 0.05, 0.2, 0.0, 20_000.0, SelectionAction::Gain, true);
        zero_gain_region.gain_db = -60.0;
        let selections = vec![
            zero_gain_region,
            Selection::new(2, 0.1, 0.35, 0.0, 20_000.0, SelectionAction::Gain, true),
        ];
        let mut selections = selections;
        selections[1].gain_db = 20.0 * 0.25_f32.log10();
        let got = apply_time_domain(&audio, &selections);

        // Sequential reference.
        let channels = 2usize;
        let mut want = audio.samples.clone();
        for (frame_index, frame) in want.chunks_exact_mut(channels).enumerate() {
            let time = frame_index as f32 / audio.sample_rate as f32;
            let gain_zero = time >= selections[0].time_start && time <= selections[0].time_end;
            let gain = if time >= selections[1].time_start && time <= selections[1].time_end {
                selections[1].gain_linear()
            } else {
                1.0
            };
            if gain_zero {
                frame.fill(0.0);
            } else {
                for sample in frame {
                    *sample *= gain;
                }
            }
        }
        let max_err = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1.0e-6, "time-domain drift {max_err}");
    }
}
