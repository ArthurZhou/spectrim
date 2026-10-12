use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::types::AudioData;

/// Original-rate audio for export plus the 44.1 kHz, stereo-limited editor copy.
pub struct DecodedAudio {
    pub source: AudioData,
    pub editor: AudioData,
}

#[derive(Debug)]
pub enum DecodeError {
    Io(std::io::Error),
    UnsupportedFormat(String),
    DecodeFailure(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::UnsupportedFormat(message) => write!(formatter, "Unsupported audio: {message}"),
            Self::DecodeFailure(message) => write!(formatter, "Audio decode failed: {message}"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<std::io::Error> for DecodeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn decode_file(path: &Path) -> Result<DecodedAudio, DecodeError> {
    decode_file_cancellable(path, &AtomicBool::new(false))
}

pub fn decode_file_cancellable(
    path: &Path,
    cancelled: &AtomicBool,
) -> Result<DecodedAudio, DecodeError> {
    let file = File::open(path)?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| DecodeError::UnsupportedFormat(error.to_string()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| {
            track.codec_params.codec != CODEC_TYPE_NULL
                && track.codec_params.sample_rate.is_some()
                && track.codec_params.channels.is_some()
        })
        .ok_or_else(|| DecodeError::UnsupportedFormat("no audio track found".to_owned()))?;
    let track_id = track.id;
    let estimated_frames = track.codec_params.n_frames;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| DecodeError::UnsupportedFormat(error.to_string()))?;

    let mut decoded_samples = Vec::<f32>::new();
    let mut source_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut source_channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(0);
    if let Some(frames) = estimated_frames {
        let estimate = usize::try_from(frames)
            .ok()
            .and_then(|frames| frames.checked_mul(source_channels));
        if let Some(sample_count) = estimate {
            // Avoid repeated Vec growth for formats that advertise their frame count.
            // Cap eager reservation at six hours; longer tracks still grow normally.
            let max_reserve_samples = 64 * 1024 * 1024 / std::mem::size_of::<f32>();
            decoded_samples.reserve(sample_count.min(max_reserve_samples));
        }
    }

    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Err(DecodeError::DecodeFailure("Loading cancelled".to_owned()));
        }
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(DecodeError::DecodeFailure(
                    "container requires decoder reset".to_owned(),
                ));
            }
            Err(error) => return Err(DecodeError::DecodeFailure(error.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(DecodeError::DecodeFailure(error.to_string())),
        };
        source_rate = decoded.spec().rate;
        source_channels = decoded.spec().channels.count();
        if source_rate == 0 || source_channels == 0 {
            return Err(DecodeError::DecodeFailure(
                "decoded stream has invalid audio parameters".to_owned(),
            ));
        }
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buffer.copy_interleaved_ref(decoded);
        let decoded = buffer.samples();
        decoded_samples.extend(decoded.iter().map(|sample| sample.clamp(-1.0, 1.0)));
    }

    if decoded_samples.is_empty() || source_rate == 0 || source_channels == 0 {
        return Err(DecodeError::DecodeFailure(
            "the selected audio track contains no decodable samples".to_owned(),
        ));
    }

    let source_channels = u16::try_from(source_channels).map_err(|_| {
        DecodeError::DecodeFailure("audio channel count exceeds WAV limits".to_owned())
    })?;
    let source_duration =
        decoded_samples.len() as f32 / source_channels as f32 / source_rate as f32;
    let source = AudioData {
        samples: decoded_samples,
        sample_rate: source_rate,
        channels: source_channels,
        duration: source_duration,
    };

    // Keep the source-rate PCM untouched for export. The editor uses a
    // stereo-limited 44.1 kHz copy for predictable preview/filter behavior.
    let editor_channels = source.channels.min(2);
    let editor_source_samples = if editor_channels == source.channels {
        source.samples.clone()
    } else {
        source
            .samples
            .chunks_exact(source.channels as usize)
            .flat_map(|frame| frame[..editor_channels as usize].iter().copied())
            .collect()
    };
    let samples = if source.sample_rate == 44_100 {
        editor_source_samples
    } else {
        resample_linear(
            &editor_source_samples,
            editor_channels as usize,
            source.sample_rate,
            44_100,
        )
    };
    let sample_rate = 44_100;
    let duration = samples.len() as f32 / editor_channels as f32 / sample_rate as f32;

    Ok(DecodedAudio {
        source,
        editor: AudioData {
            samples,
            sample_rate,
            channels: editor_channels,
            duration,
        },
    })
}

fn resample_linear(
    samples: &[f32],
    channels: usize,
    source_rate: u32,
    target_rate: u32,
) -> Vec<f32> {
    if channels == 0 || source_rate == 0 || target_rate == 0 || samples.is_empty() {
        return Vec::new();
    }
    let input_frames = samples.len() / channels;
    let output_frames = ((input_frames as u64 * target_rate as u64) / source_rate as u64) as usize;
    let mut output = vec![0.0; output_frames.saturating_mul(channels)];
    output
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(output_frame, output_samples)| {
            let position = output_frame as f64 * source_rate as f64 / target_rate as f64;
            let left = (position.floor() as usize).min(input_frames.saturating_sub(1));
            let right = (left + 1).min(input_frames.saturating_sub(1));
            let fraction = (position - left as f64) as f32;
            for channel in 0..channels {
                let a = samples[left * channels + channel];
                let b = samples[right * channels + channel];
                output_samples[channel] = a + (b - a) * fraction;
            }
        });
    output
}

#[cfg(test)]
mod tests {
    use super::resample_linear;

    #[test]
    fn editor_resampling_keeps_duration_and_stereo_layout() {
        let source_rate = 48_000;
        let target_rate = 44_100;
        let frames = source_rate as usize;
        let samples = (0..frames)
            .flat_map(|frame| {
                let sample = frame as f32 / frames as f32;
                [sample, -sample]
            })
            .collect::<Vec<_>>();

        let converted = resample_linear(&samples, 2, source_rate, target_rate);

        assert_eq!(converted.len() / 2, target_rate as usize);
        assert!(converted.chunks_exact(2).all(|frame| frame[0] == -frame[1]));
    }
}
