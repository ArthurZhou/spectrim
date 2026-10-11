use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use super::types::{hann_window, AudioData, Spectrogram};

pub fn compute(audio: &AudioData, n_fft: usize, hop: usize) -> Spectrogram {
    let n_freq = n_fft / 2 + 1;
    let mono = audio.to_mono();
    if n_fft == 0 || hop == 0 || mono.len() < n_fft {
        return Spectrogram {
            matrix: Vec::new(),
            n_freq,
            n_time: 0,
            n_fft,
            hop_length: hop,
            sample_rate: audio.sample_rate,
            db_min: 0.0,
            db_max: 0.0,
        };
    }

    let n_time = (mono.len() - n_fft) / hop + 1;
    let window = hann_window(n_fft);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut matrix = vec![0.0; n_freq * n_time];
    let mut frame = vec![Complex32::default(); n_fft];
    let mut magnitudes = vec![0.0; n_freq];

    for time in 0..n_time {
        let start = time * hop;
        for index in 0..n_fft {
            frame[index] = Complex32::new(mono[start + index] * window[index], 0.0);
        }
        fft.process(&mut frame);
        for frequency in 0..n_freq {
            let magnitude = frame[frequency].norm();
            let db = 20.0 * (magnitude + 1.0e-9).log10();
            magnitudes[frequency] = db;
            matrix[frequency * n_time + time] = db;
        }
    }

    let (db_min, db_max) = compute_db_range(&matrix);
    Spectrogram {
        matrix,
        n_freq,
        n_time,
        n_fft,
        hop_length: hop,
        sample_rate: audio.sample_rate,
        db_min,
        db_max,
    }
}

fn compute_db_range(matrix: &[f32]) -> (f32, f32) {
    if matrix.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = matrix.to_vec();
    sorted.sort_by(f32::total_cmp);
    let low = ((sorted.len() - 1) as f32 * 0.05).round() as usize;
    let high = ((sorted.len() - 1) as f32 * 0.95).round() as usize;
    (sorted[low], sorted[high])
}
