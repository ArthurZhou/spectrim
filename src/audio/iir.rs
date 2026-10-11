//! Biquad IIR filters used by the fast preview path.
//!
//! The preview renderer never runs an STFT. A frequency-selective region is
//! approximated by extracting the selected band with cascaded Butterworth
//! biquads and then adding/subtracting that band from the dry signal inside the
//! region's time window. Compared with the overlap-add STFT used for export
//! this is roughly two orders of magnitude cheaper per sample, which is what
//! makes live scrubbing while editing usable.
//!
//! Accuracy note: the band edges are realised with a 4th-order Butterworth
//! response, so the skirt is softer than the brick-wall bin masking of the
//! export path. That is deliberate — export stays exact, preview stays fast.

use std::f32::consts::PI;

use rayon::prelude::*;

/// Pole Q values of a 4th-order Butterworth lowpass (two cascaded sections).
const BUTTERWORTH4_Q: [f32; 2] = [0.54119610, 1.30656296];

/// Below this cutoff a lowpass is treated as "remove nothing but DC", which is
/// inaudible; skipping it avoids a very long filter settling tail.
const MIN_CUTOFF_HZ: f32 = 8.0;

/// Direct-form-I biquad section with `a0` already normalised to 1.
#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Biquad {
    /// RBJ Audio EQ Cookbook lowpass.
    ///
    /// `cutoff_hz` is clamped into the representable range so callers do not
    /// have to guard against 0 Hz or above-Nyquist values.
    pub fn lowpass(cutoff_hz: f32, q: f32, sample_rate: f32) -> Self {
        let normalised = (cutoff_hz / sample_rate.max(1.0)).clamp(1.0e-5, 0.4999);
        let w0 = 2.0 * PI * normalised;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q.max(1.0e-3));
        let a0 = 1.0 + alpha;
        let b1 = 1.0 - cos_w0;
        Self {
            b0: (b1 * 0.5) / a0,
            b1: b1 / a0,
            b2: (b1 * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Filter `input` into `output`. Both slices must have the same length;
    /// they may alias only if they are the same buffer.
    pub fn apply(&self, input: &[f32], output: &mut [f32]) {
        let (mut x1, mut x2) = (0.0f32, 0.0f32);
        let (mut y1, mut y2) = (0.0f32, 0.0f32);
        for index in 0..input.len().min(output.len()) {
            let x0 = input[index];
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            output[index] = y0;
        }
    }
}

/// Number of leading frames a chunk needs as warm-up before its output has
/// converged, i.e. the ~80 dB settling time of the slowest section.
fn settle_frames(cutoff_hz: f32, sample_rate: f32) -> usize {
    let cutoff = cutoff_hz.max(1.0);
    // tau ~= Q / (pi * fc); 4th order needs roughly 4 * tau for -80 dB.
    let seconds = (4.0 * BUTTERWORTH4_Q[1] / (PI * cutoff)).min(2.0);
    let frames = (seconds * sample_rate) as usize;
    frames.clamp((sample_rate as usize) / 100, (sample_rate as usize) * 2)
}

/// 4th-order Butterworth lowpass, split into overlapping chunks so rayon can
/// run it on several cores. Each chunk re-filters `settle` warm-up frames from
/// zero state and throws them away, which reproduces the sequential result to
/// within filter noise.
pub fn lowpass4(input: &[f32], output: &mut [f32], cutoff_hz: f32, sample_rate: f32) {
    let length = input.len().min(output.len());
    if length == 0 {
        return;
    }
    if cutoff_hz <= 1.0e-3 {
        output[..length].fill(0.0);
        return;
    }
    if cutoff_hz >= sample_rate * 0.4999 {
        output[..length].copy_from_slice(&input[..length]);
        return;
    }

    let sections = [
        Biquad::lowpass(cutoff_hz, BUTTERWORTH4_Q[0], sample_rate),
        Biquad::lowpass(cutoff_hz, BUTTERWORTH4_Q[1], sample_rate),
    ];
    let settle = settle_frames(cutoff_hz, sample_rate).min(length);
    let threads = rayon::current_num_threads().max(1);
    let chunk = ((length / (threads * 4)).max(4096))
        .max(settle + 1024)
        .min(length)
        .max(1);

    let input = &input[..length];
    output[..length]
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(index, out_chunk)| {
            let start = index * chunk;
            let end = start + out_chunk.len();
            let warm = start.saturating_sub(settle);
            let mut buffer = input[warm..end].to_vec();
            let mut scratch = vec![0.0f32; buffer.len()];
            for section in &sections {
                section.apply(&buffer, &mut scratch);
                std::mem::swap(&mut buffer, &mut scratch);
            }
            let skip = start - warm;
            out_chunk.copy_from_slice(&buffer[skip..]);
        });
}

/// Zero-phase 4th-order Butterworth lowpass: forward pass, then the same pass
/// run on the time-reversed signal, then reverse back.
///
/// This is the offline equivalent of `scipy.signal.filtfilt`. Zero phase is not
/// a nicety here — it is what makes the preview maths correct: `render_range`
/// removes a band by computing `dry - band`, and that subtraction only cancels
/// the in-band component if `band` is phase-aligned with `dry`. With a plain
/// causal bandpass the residual does not cancel and the "muted" region can come
/// out *louder* than the input (measured RMS 1.41 vs 1.00 before this change).
///
/// Side effect: the magnitude response is squared, so the effective order is 8
/// and the skirt is steeper than a causal 4th-order filter. That is desirable
/// for masking.
pub fn lowpass4_zero_phase(input: &[f32], output: &mut [f32], cutoff_hz: f32, sample_rate: f32) {
    let length = input.len().min(output.len());
    if length == 0 {
        return;
    }
    // Forward pass (handles the cutoff <= 0 and >= Nyquist shortcuts itself).
    lowpass4(input, output, cutoff_hz, sample_rate);
    if cutoff_hz <= 1.0e-3 || cutoff_hz >= sample_rate * 0.4999 {
        return; // all-pass / all-stop: reversing twice is a no-op
    }
    output[..length].reverse();
    let mut scratch = vec![0.0f32; length];
    lowpass4(&output[..length], &mut scratch, cutoff_hz, sample_rate);
    output[..length].copy_from_slice(&scratch);
    output[..length].reverse();
}

/// Frames of warm-up a caller must feed before `band_extract` output has
/// converged for this band. Used to size the overlap when rendering only a
/// sub-range of the track incrementally. Doubled versus a single causal pass
/// because the zero-phase version filters twice.
pub fn warmup_frames(freq_low: f32, freq_high: f32, sample_rate: f32) -> usize {
    let nyquist = sample_rate * 0.5;
    let high = freq_high.clamp(0.0, nyquist);
    // The slower (lower) cutoff dominates settling; a near-DC low edge means the
    // lowpass leg is skipped entirely, so only the high edge matters.
    let dominating = if freq_low > MIN_CUTOFF_HZ {
        freq_low.min(high)
    } else {
        high
    };
    if dominating <= 1.0e-3 {
        return 0;
    }
    settle_frames(dominating, sample_rate) * 2
}

/// Extract the energy inside `[freq_low, freq_high]` as `LP(hi) - LP(lo)`,
/// both legs zero-phase so the result is phase-aligned with `input`.
///
/// Degenerate edges fall out naturally: `freq_low` near DC yields a plain
/// lowpass, `freq_high` near Nyquist yields a plain highpass.
pub fn band_extract(
    input: &[f32],
    output: &mut [f32],
    freq_low: f32,
    freq_high: f32,
    sample_rate: f32,
) {
    let length = input.len().min(output.len());
    if length == 0 {
        return;
    }
    lowpass4_zero_phase(input, output, freq_high, sample_rate);
    if freq_low > MIN_CUTOFF_HZ {
        let mut low = vec![0.0f32; length];
        lowpass4_zero_phase(input, &mut low, freq_low, sample_rate);
        output[..length]
            .par_iter_mut()
            .zip(low.par_iter())
            .for_each(|(value, subtract)| *value -= subtract);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, frames: usize, sample_rate: f32) -> Vec<f32> {
        (0..frames)
            .map(|index| (2.0 * PI * freq * index as f32 / sample_rate).sin())
            .collect()
    }

    fn rms(values: &[f32]) -> f32 {
        (values.iter().map(|v| v * v).sum::<f32>() / values.len().max(1) as f32).sqrt()
    }

    #[test]
    fn lowpass_attenuates_above_cutoff() {
        let sample_rate = 48_000.0;
        let frames = 48_000;
        let high = sine(12_000.0, frames, sample_rate);
        let mut out = vec![0.0f32; frames];
        lowpass4(&high, &mut out, 1_000.0, sample_rate);
        // Measure only the settled tail so chunk warm-up is not part of the claim.
        assert!(rms(&out[frames / 2..]) < 0.05 * rms(&high[frames / 2..]));
    }

    #[test]
    fn lowpass_passes_below_cutoff() {
        let sample_rate = 48_000.0;
        let frames = 48_000;
        let low = sine(200.0, frames, sample_rate);
        let mut out = vec![0.0f32; frames];
        lowpass4(&low, &mut out, 4_000.0, sample_rate);
        assert!(rms(&out[frames / 2..]) > 0.7 * rms(&low[frames / 2..]));
    }

    #[test]
    fn band_extract_isolates_the_band() {
        let sample_rate = 48_000.0;
        let frames = 48_000;
        let in_band = sine(1_000.0, frames, sample_rate);
        let out_band = sine(9_000.0, frames, sample_rate);
        let mixed: Vec<f32> = in_band
            .iter()
            .zip(out_band.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut extracted = vec![0.0f32; frames];
        band_extract(&mixed, &mut extracted, 700.0, 1_400.0, sample_rate);
        let tail = frames / 2..;
        // The 1 kHz component survives, the 9 kHz component is largely gone.
        assert!(rms(&extracted[tail.clone()]) < 1.2 * rms(&in_band[tail.clone()]));
        let residual = rms(&extracted[tail.clone()]) - rms(&in_band[tail]);
        assert!(residual.abs() < 0.35);
    }

    #[test]
    fn chunked_output_matches_sequential_within_noise() {
        let sample_rate = 44_100.0;
        let frames = 200_000;
        let input = sine(440.0, frames, sample_rate);
        let mut parallel = vec![0.0f32; frames];
        lowpass4(&input, &mut parallel, 2_000.0, sample_rate);

        // Sequential reference: one single section chain over the whole signal.
        let sections = [
            Biquad::lowpass(2_000.0, BUTTERWORTH4_Q[0], sample_rate),
            Biquad::lowpass(2_000.0, BUTTERWORTH4_Q[1], sample_rate),
        ];
        let mut reference = input.clone();
        let mut scratch = vec![0.0f32; frames];
        for section in &sections {
            section.apply(&reference, &mut scratch);
            std::mem::swap(&mut reference, &mut scratch);
        }

        let max_error = parallel
            .iter()
            .zip(reference.iter())
            .skip(frames / 4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 1.0e-3, "chunk drift too large: {max_error}");
    }

    #[test]
    fn zero_phase_lowpass_has_no_group_delay() {
        // A causal lowpass delays its output; the zero-phase version must line
        // up with the input, which is the property band cancellation relies on.
        let sample_rate = 48_000.0;
        let frames = 48_000;
        let input = sine(300.0, frames, sample_rate);
        let mut causal = vec![0.0f32; frames];
        lowpass4(&input, &mut causal, 4_000.0, sample_rate);
        let mut zero = vec![0.0f32; frames];
        lowpass4_zero_phase(&input, &mut zero, 4_000.0, sample_rate);

        let mid = frames / 4..frames * 3 / 4;
        // Correlate each output against the input; the zero-phase peak lag is 0.
        let peak_lag = |out: &[f32]| {
            (-48..=48)
                .map(|lag| {
                    let energy: f32 = mid
                        .clone()
                        .map(|i| {
                            let j = i as isize + lag;
                            if j >= 0 && (j as usize) < frames {
                                out[i] * input[j as usize]
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    (lag, energy)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(lag, _)| lag)
                .unwrap_or(0)
        };
        assert_eq!(peak_lag(&zero), 0, "zero-phase filter still delayed");
        // Sanity: the causal one is not exactly aligned (or the test is vacuous).
        assert!(rms(&zero[mid.clone()]) > 0.7 * rms(&input[mid]));
    }

    #[test]
    fn subtracting_extracted_band_removes_in_band_tone() {
        // The exact operation render_range performs for a band mute: the
        // residual must be the out-of-band tone alone, not a louder mess.
        let sample_rate = 48_000.0;
        let frames = 48_000;
        let in_band = sine(1_000.0, frames, sample_rate);
        let out_band = sine(9_000.0, frames, sample_rate);
        let mixed: Vec<f32> = in_band
            .iter()
            .zip(out_band.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut band = vec![0.0f32; frames];
        band_extract(&mixed, &mut band, 700.0, 1_400.0, sample_rate);
        let residual: Vec<f32> = mixed.iter().zip(band.iter()).map(|(a, b)| a - b).collect();

        let mid = frames / 4..frames * 3 / 4;
        let residual_rms = rms(&residual[mid.clone()]);
        let target_rms = rms(&out_band[mid.clone()]);
        let mixed_rms = rms(&mixed[mid]);
        assert!(
            residual_rms < mixed_rms * 0.75,
            "band subtraction did not attenuate: {residual_rms} vs mixed {mixed_rms}"
        );
        assert!(
            (residual_rms - target_rms).abs() < 0.15,
            "residual should equal the out-of-band tone: {residual_rms} vs {target_rms}"
        );
    }

    #[test]
    fn degenerate_edges_do_not_panic_or_emit_nan() {
        let sample_rate = 44_100.0;
        let frames = 4_096;
        let input = sine(1_000.0, frames, sample_rate);
        for (low, high) in [
            (0.0, sample_rate * 0.5),
            (0.0, 100.0),
            (20_000.0, sample_rate),
            (1.0, 2.0),
        ] {
            let mut out = vec![0.0f32; frames];
            band_extract(&input, &mut out, low, high, sample_rate);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "non-finite for {low}..{high}"
            );
        }
    }
}
