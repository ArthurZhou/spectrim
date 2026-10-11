use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use super::types::{hann_window, AudioData, Selection, SelectionAction};

pub fn apply_selections(audio: &AudioData, selections: &[Selection]) -> Vec<f32> {
    if selections.is_empty() {
        return audio.samples.clone();
    }
    if selections.iter().all(|selection| selection.full_band) {
        apply_time_domain(audio, selections)
    } else {
        apply_spectral_masks(audio, selections, 2048, 512)
    }
}

pub fn apply_time_domain(audio: &AudioData, selections: &[Selection]) -> Vec<f32> {
    let has_solo = selections
        .iter()
        .any(|selection| selection.action == SelectionAction::Solo);
    let channels = audio.channels as usize;
    if channels == 0 {
        return audio.samples.clone();
    }
    let mut output = audio.samples.clone();
    for (frame_index, frame) in output.chunks_exact_mut(channels).enumerate() {
        let time = frame_index as f32 / audio.sample_rate.max(1) as f32;
        let in_solo = selections.iter().any(|selection| {
            selection.action == SelectionAction::Solo
                && time >= selection.time_start
                && time <= selection.time_end
        });
        let in_mute = selections.iter().any(|selection| {
            selection.action == SelectionAction::Mute
                && time >= selection.time_start
                && time <= selection.time_end
        });
        if (has_solo && !in_solo) || (!has_solo && in_mute) {
            frame.fill(0.0);
        } else {
            let gain = selections
                .iter()
                .filter(|selection| {
                    selection.action == SelectionAction::Gain
                        && time >= selection.time_start
                        && time <= selection.time_end
                })
                .map(|selection| selection.gain.clamp(0.0, 4.0))
                .product::<f32>();
            for sample in frame {
                *sample *= gain;
            }
        }
    }
    output
}

pub fn apply_spectral_masks(
    audio: &AudioData,
    selections: &[Selection],
    n_fft: usize,
    hop: usize,
) -> Vec<f32> {
    let channels = audio.channels as usize;
    if channels == 0 || n_fft < 2 || hop == 0 || audio.samples.is_empty() {
        return audio.samples.clone();
    }
    let input_frames = audio.frame_count();
    if input_frames < n_fft {
        return apply_time_domain(audio, selections);
    }
    let n_freq = n_fft / 2 + 1;
    let n_time = (input_frames - n_fft) / hop + 1;
    let window = hann_window(n_fft);
    let has_solo = selections
        .iter()
        .any(|selection| selection.action == SelectionAction::Solo);
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
                    .filter(|selection| selection.action == SelectionAction::Solo)
                    .any(in_bounds);
                let in_mute = selections
                    .iter()
                    .filter(|selection| selection.action == SelectionAction::Mute)
                    .any(in_bounds);
                let keep = if has_solo { in_solo } else { !in_mute };
                if !keep {
                    spectrum[bin] = Complex32::default();
                    if bin > 0 && bin < n_fft / 2 {
                        spectrum[n_fft - bin] = Complex32::default();
                    }
                } else {
                    let gain = selections
                        .iter()
                        .filter(|selection| selection.action == SelectionAction::Gain)
                        .filter(|selection| in_bounds(selection))
                        .map(|selection| selection.gain.clamp(0.0, 4.0))
                        .product::<f32>();
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
    fn time_domain_mute_and_solo_are_applied() {
        let audio = sample_audio();
        let mute = Selection::new(1, 0.5, 1.0, 0.0, 500.0, SelectionAction::Mute, true);
        let muted = apply_time_domain(&audio, &[mute]);
        assert!(muted[500..1000].iter().all(|sample| *sample == 0.0));
        assert!(muted[0..400].iter().any(|sample| *sample != 0.0));

        let solo = Selection::new(2, 0.5, 1.0, 0.0, 500.0, SelectionAction::Solo, true);
        let soloed = apply_time_domain(&audio, &[solo]);
        assert!(soloed[0..500].iter().all(|sample| *sample == 0.0));
        assert!(soloed[500..1000].iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn gain_region_applies_linear_multiplier() {
        let audio = sample_audio();
        let gain = Selection::new(1, 0.5, 1.0, 0.0, 500.0, SelectionAction::Gain, true);
        let output = apply_time_domain(&audio, &[gain]);
        for (index, sample) in output.iter().enumerate().take(1000).skip(500) {
            assert!((*sample - audio.samples[index] * 0.5).abs() < 1.0e-6);
        }
        assert_eq!(output[100], audio.samples[100]);
    }

    #[test]
    fn spectral_processing_preserves_length_and_finite_values() {
        let audio = sample_audio();
        let mute = Selection::new(1, 0.2, 1.5, 50.0, 200.0, SelectionAction::Mute, false);
        let output = apply_spectral_masks(&audio, &[mute], 256, 64);
        assert_eq!(output.len(), audio.samples.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
    }
}
