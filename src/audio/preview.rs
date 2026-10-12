//! Fast preview rendering for the "Edited playback" checkbox.
//!
//! The preview path deliberately avoids the STFT. The rule is:
//!   * export  -> exact overlap-add STFT bin masking (`process::apply_spectral_masks`)
//!   * preview -> cheap time-domain ops + IIR band extraction (this module)
//!
//! Preview must be cheap enough to re-render while the user is still dragging a
//! region, so it is also incremental: a `PreviewCache` keeps the last rendered
//! interleaved buffer plus the selection state it was rendered from, and on the
//! next edit only the time span actually touched is recomputed and spliced back
//! in.
//!
//! Two design points make the increment possible:
//!   1. A full-band region (solo / whole-band gain, including zero gain) is a
//!      pure time-domain multiply, so it can be re-evaluated for any frame range
//!      in isolation.
//!   2. A frequency-selective region only needs the *dry* audio filtered into
//!      its band; the band signal for the whole track is cached per region id,
//!      so moving a region's time window never re-runs the IIR filter.
//!
//! Preview semantics for frequency-selective regions are an approximation of
//! the export path: the band is extracted with a 4th-order Butterworth pair
//! instead of brick-wall FFT bins. Zero gain subtracts the band, other Gain
//! values add `band * (gain - 1)`, and Solo replaces the output with the band
//! inside its region. Any Solo region takes precedence over outside-region
//! Gain regions: output is silent outside the union of Solo regions. Export
//! stays exact.

use rayon::prelude::*;

use super::iir;
use super::process::apply_spectral_masks;
use super::types::{AudioData, Selection, SelectionAction, SelectionShape};

/// What a render call produced.
#[derive(Debug)]
pub struct PreviewOutcome {
    /// Frame span `[start, end)` that was recomputed; `None` when the cache was
    /// already up to date and nothing changed.
    pub changed_frames: Option<(usize, usize)>,
    /// True when the whole buffer was rebuilt (audio changed, or the change
    /// could not be localised to a span).
    pub full_rebuild: bool,
}

/// Cached preview result. `samples` is the fully rendered interleaved buffer for
/// `selections`; `band_signals` holds the extracted band for each
/// frequency-selective region, keyed by selection id.
pub struct PreviewCache {
    /// Rendered interleaved preview samples. Read this after `render`.
    pub samples: Vec<f32>,
    /// The selection set `samples` was rendered from.
    pub selections: Vec<Selection>,
    /// id -> per-channel extracted band (channel-major, full track length).
    band_signals: Vec<(u64, Vec<Vec<f32>>)>,
    /// Identity of the dry audio the cache was built from, to detect a reload.
    audio_signature: AudioSignature,
}

/// Cheap identity for "is this still the same decoded audio".
#[derive(Clone, Copy, PartialEq, Eq)]
struct AudioSignature {
    frames: usize,
    sample_rate: u32,
    channels: u16,
    checksum: u64,
}

impl AudioSignature {
    fn of(audio: &AudioData) -> Self {
        // Strided FNV over the samples: exact enough to catch a different file
        // or decode, cheap enough to run on every edit.
        let stride = (audio.samples.len() / 4096).max(1);
        let checksum = audio
            .samples
            .iter()
            .step_by(stride)
            .map(|s| s.to_bits() as u64)
            .fold(1469598103934665603u64, |acc, v| {
                (acc ^ v).wrapping_mul(1099511628211)
            });
        Self {
            frames: audio.frame_count(),
            sample_rate: audio.sample_rate,
            channels: audio.channels,
            checksum,
        }
    }
}

impl Default for PreviewCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A selection's time window pre-resolved to frame indices.
struct Resolved {
    action: SelectionAction,
    start: usize,
    end: usize,
    gain: f32,
}

fn resolve_time(selection: &Selection, sample_rate: f32) -> (usize, usize) {
    (
        (selection.time_start.max(0.0) * sample_rate).round() as usize,
        (selection.time_end.max(0.0) * sample_rate).round() as usize,
    )
}

/// Per-frame full-band factor: `None` means the frame is outside every solo
/// region when a full-band solo exists; otherwise it is the
/// cumulative gain multiplier.
///
/// `ignore_solo` is used for the band-solo path, where full-band solo gating
/// must not silence the band output.
fn full_band_factor(
    resolved: &[Resolved],
    has_solo: bool,
    frame_index: usize,
    ignore_solo: bool,
) -> Option<f32> {
    let mut gain = 1.0f32;
    let mut in_solo = false;
    for r in resolved {
        let inside = frame_index >= r.start && frame_index < r.end;
        match r.action {
            SelectionAction::Solo => in_solo |= inside,
            SelectionAction::Gain => {
                if inside {
                    gain *= r.gain;
                }
            }
        }
    }
    if ignore_solo {
        Some(gain)
    } else if has_solo {
        if in_solo {
            Some(1.0)
        } else {
            None
        }
    } else {
        Some(gain)
    }
}

fn block_size(span: usize, channels: usize) -> usize {
    let per_thread = span / (rayon::current_num_threads() * 8).max(1);
    per_thread.max(512).min(span.max(1)).max(1) * channels
}

/// Render frames `[start_frame, end_frame)` applying full-band regions only.
///
/// Pure time-domain, so any sub-range can be computed independently — this is
/// what the incremental splice relies on.
fn render_time_domain(
    audio: &AudioData,
    selections: &[Selection],
    start_frame: usize,
    end_frame: usize,
) -> Vec<f32> {
    let channels = audio.channels as usize;
    let sample_rate = audio.sample_rate.max(1) as f32;
    let end_frame = end_frame.min(audio.frame_count());
    let start_frame = start_frame.min(end_frame);
    let span = end_frame - start_frame;
    let mut output = vec![0.0f32; span * channels];
    if span == 0 || channels == 0 {
        return output;
    }

    let resolved: Vec<Resolved> = selections
        .iter()
        .map(|s| {
            let (start, end) = resolve_time(s, sample_rate);
            Resolved {
                action: s.action,
                start,
                end,
                gain: s.gain.clamp(0.0, 4.0),
            }
        })
        .collect();
    let has_solo = resolved.iter().any(|r| r.action == SelectionAction::Solo);

    let block_samples = block_size(span, channels);
    output
        .par_chunks_mut(block_samples)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let mut frame_index = start_frame + (block_index * block_samples) / channels.max(1);
            for frame in chunk.chunks_exact_mut(channels) {
                let factor = full_band_factor(&resolved, has_solo, frame_index, false);
                let src = &audio.samples[frame_index * channels..(frame_index + 1) * channels];
                match factor {
                    None => frame.fill(0.0),
                    Some(gain) => {
                        for (sample, source) in frame.iter_mut().zip(src) {
                            *sample = source * gain;
                        }
                    }
                }
                frame_index += 1;
            }
        });
    output
}

fn deinterleave(audio: &AudioData) -> Vec<Vec<f32>> {
    let channels = audio.channels as usize;
    let frames = audio.frame_count();
    let mut out = vec![vec![0.0f32; frames]; channels];
    for (c, channel_data) in out.iter_mut().enumerate() {
        let mut src = audio.samples[c..].chunks_exact(channels);
        for value in channel_data.iter_mut().take(frames) {
            *value = src.next().map_or(0.0, |frame| frame[0]);
        }
    }
    out
}

/// Extract the band signal for every frequency-selective selection, in
/// parallel across regions (each region's filter pass is itself parallel).
fn build_band_signals(audio: &AudioData, selections: &[Selection]) -> Vec<(u64, Vec<Vec<f32>>)> {
    let dry = deinterleave(audio);
    let sample_rate = audio.sample_rate as f32;
    let channels = audio.channels as usize;
    let frames = audio.frame_count();

    let targets: Vec<&Selection> = selections.iter().filter(|s| !s.full_band).collect();
    targets
        .into_par_iter()
        .map(|s| {
            let mut band = vec![vec![0.0f32; frames]; channels];
            for c in 0..channels {
                iir::band_extract(&dry[c], &mut band[c], s.freq_low, s.freq_high, sample_rate);
            }
            (s.id, band)
        })
        .collect()
}

/// Render frames `[start_frame, end_frame)` combining full-band time-domain
/// regions with frequency-selective band regions.
fn render_range(
    audio: &AudioData,
    selections: &[Selection],
    band_signals: &[(u64, Vec<Vec<f32>>)],
    start_frame: usize,
    end_frame: usize,
) -> Vec<f32> {
    let channels = audio.channels as usize;
    let frames = audio.frame_count();
    let sample_rate = audio.sample_rate.max(1) as f32;
    let end_frame = end_frame.min(frames);
    let start_frame = start_frame.min(end_frame);
    let span = end_frame - start_frame;
    if span == 0 || channels == 0 {
        return Vec::new();
    }

    // Full-band regions only — band regions are layered on top below.
    let full_band: Vec<Selection> = selections.iter().filter(|s| s.full_band).cloned().collect();
    let mut output = render_time_domain(audio, &full_band, start_frame, end_frame);

    struct BandRegion<'a> {
        action: SelectionAction,
        start: usize,
        end: usize,
        gain: f32,
        band: &'a [Vec<f32>],
    }
    let band_regions: Vec<BandRegion> = selections
        .iter()
        .filter(|s| !s.full_band)
        .filter_map(|s| {
            band_signals
                .iter()
                .find(|(id, _)| *id == s.id)
                .map(|(_, band)| {
                    let (start, end) = resolve_time(s, sample_rate);
                    BandRegion {
                        action: s.action,
                        start,
                        end,
                        gain: s.gain.clamp(0.0, 4.0),
                        band,
                    }
                })
        })
        .collect();
    if band_regions.is_empty() {
        return output;
    }
    let has_solo = selections
        .iter()
        .any(|selection| selection.action == SelectionAction::Solo);
    let full_band_solos: Vec<_> = selections
        .iter()
        .filter(|selection| selection.full_band && selection.action == SelectionAction::Solo)
        .map(|selection| resolve_time(selection, sample_rate))
        .collect();

    let block = block_size(span, channels);
    output
        .par_chunks_mut(block)
        .enumerate()
        .for_each(|(block_index, chunk)| {
            let mut frame_index = start_frame + (block_index * block) / channels;
            for frame in chunk.chunks_exact_mut(channels) {
                if has_solo {
                    // Solo regions form a union across full-band and band
                    // regions. Gain regions are ignored while any Solo exists.
                    let mut in_any_solo = false;
                    frame.fill(0.0);
                    let full_band_solo_active = full_band_solos
                        .iter()
                        .any(|(start, end)| frame_index >= *start && frame_index < *end);
                    if full_band_solo_active {
                        for (channel, value) in frame.iter_mut().enumerate() {
                            *value = audio.samples[frame_index * channels + channel];
                        }
                        in_any_solo = true;
                    } else {
                        for region in &band_regions {
                            if region.action == SelectionAction::Solo
                                && frame_index >= region.start
                                && frame_index < region.end
                            {
                                for c in 0..channels {
                                    frame[c] +=
                                        region.band[c].get(frame_index).copied().unwrap_or(0.0);
                                }
                                in_any_solo = true;
                            }
                        }
                    }
                    if !in_any_solo {
                        frame.fill(0.0);
                    }
                } else {
                    for region in &band_regions {
                        if frame_index < region.start || frame_index >= region.end {
                            continue;
                        }
                        for c in 0..channels {
                            let band_value =
                                region.band[c].get(frame_index).copied().unwrap_or(0.0);
                            match region.action {
                                // Remove the band from the (full-band processed) dry.
                                // Scale only the band: dry + band * (gain - 1).
                                SelectionAction::Gain => {
                                    frame[c] += band_value * (region.gain - 1.0)
                                }
                                SelectionAction::Solo => {
                                    unreachable!("solo handled by band-solo branch")
                                }
                            }
                        }
                    }
                }
                frame_index += 1;
            }
        });
    output
}

/// Render a whole preview from scratch (no cache).
pub fn render_full(audio: &AudioData, selections: &[Selection]) -> Vec<f32> {
    let enabled = selections
        .iter()
        .filter(|selection| selection.enabled)
        .cloned()
        .collect::<Vec<_>>();
    if enabled.len() != selections.len() {
        return render_full(audio, &enabled);
    }
    let channels = audio.channels as usize;
    if channels == 0 || audio.samples.is_empty() || selections.is_empty() {
        return audio.samples.clone();
    }
    if selections.iter().any(Selection::is_polygon) {
        return apply_spectral_masks(audio, selections, 2048, 512);
    }
    if selections.iter().all(|s| s.full_band) {
        return render_time_domain(audio, selections, 0, audio.frame_count());
    }
    let band_signals = build_band_signals(audio, selections);
    render_range(audio, selections, &band_signals, 0, audio.frame_count())
}

impl PreviewCache {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            selections: Vec::new(),
            band_signals: Vec::new(),
            audio_signature: AudioSignature {
                frames: 0,
                sample_rate: 0,
                channels: 0,
                checksum: 0,
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Bring `self.samples` up to date with `selections`, recomputing as little
    /// as possible. Read `self.samples` afterwards; the returned outcome says
    /// what was recomputed (if anything).
    pub fn render(&mut self, audio: &AudioData, selections: &[Selection]) -> PreviewOutcome {
        let enabled = selections
            .iter()
            .filter(|selection| selection.enabled)
            .cloned()
            .collect::<Vec<_>>();
        if enabled.len() != selections.len() {
            return self.render(audio, &enabled);
        }
        let signature = AudioSignature::of(audio);
        let channels = audio.channels as usize;
        let frames = audio.frame_count();

        if selections.iter().any(Selection::is_polygon)
            || self.selections.iter().any(Selection::is_polygon)
        {
            self.samples = render_full(audio, selections);
            self.selections = selections.to_vec();
            self.band_signals.clear();
            self.audio_signature = signature;
            return PreviewOutcome {
                changed_frames: Some((0, frames)),
                full_rebuild: true,
            };
        }

        // Different audio (or first call) -> full rebuild.
        if self.audio_signature != signature || self.samples.len() != frames * channels {
            self.audio_signature = signature;
            self.selections = selections.to_vec();
            self.band_signals = if selections.iter().all(|s| s.full_band) {
                Vec::new()
            } else {
                build_band_signals(audio, selections)
            };
            self.samples = render_full(audio, selections);
            return PreviewOutcome {
                changed_frames: Some((0, frames)),
                full_rebuild: true,
            };
        }

        // Keep polygon previews on the same exact STFT path as export. The
        // IIR cache represents a band rectangle and cannot follow polygon edges.
        if selections.iter().any(Selection::is_polygon)
            || self.selections.iter().any(Selection::is_polygon)
        {
            self.samples = render_full(audio, selections);
            self.selections = selections.to_vec();
            self.band_signals.clear();
            self.audio_signature = signature;
            return PreviewOutcome {
                changed_frames: Some((0, frames)),
                full_rebuild: true,
            };
        }

        let diff = diff_selections(&self.selections, selections);
        if diff.is_noop() {
            return PreviewOutcome {
                changed_frames: None,
                full_rebuild: false,
            };
        }

        let need_bands = selections.iter().any(|s| !s.full_band);
        if need_bands {
            self.refresh_bands(audio, selections, &diff);
        } else {
            self.band_signals.clear();
        }

        let (span_start, span_end) = self.compute_span(audio, selections, &diff);
        let span_start = span_start.min(frames);
        let span_end = span_end.clamp(span_start, frames);

        let rendered = if need_bands {
            render_range(audio, selections, &self.band_signals, span_start, span_end)
        } else {
            render_time_domain(audio, selections, span_start, span_end)
        };

        let start_sample = span_start * channels;
        let end_sample = span_end * channels;
        let full_rebuild = rendered.len() != end_sample - start_sample;
        if full_rebuild {
            // Span bookkeeping disagreed (should not happen); rebuild everything
            // so the cache is never left half-updated.
            self.samples = if need_bands {
                render_range(audio, selections, &self.band_signals, 0, frames)
            } else {
                render_time_domain(audio, selections, 0, frames)
            };
            self.selections = selections.to_vec();
            return PreviewOutcome {
                changed_frames: Some((0, frames)),
                full_rebuild: true,
            };
        }

        self.samples[start_sample..end_sample].copy_from_slice(&rendered);
        self.selections = selections.to_vec();
        PreviewOutcome {
            changed_frames: Some((span_start, span_end)),
            full_rebuild: false,
        }
    }

    /// Recompute band signals only for ids that are new or whose frequency edges
    /// moved; drop bands whose region was removed. Unchanged bands are kept, so
    /// dragging a region's time window costs zero filtering.
    fn refresh_bands(&mut self, audio: &AudioData, selections: &[Selection], diff: &SelectionDiff) {
        self.band_signals
            .retain(|(id, _)| !diff.removed.contains(id));

        let to_build: Vec<&Selection> = selections
            .iter()
            .filter(|s| !s.full_band)
            .filter(|s| {
                diff.band_changed.contains(&s.id)
                    || !self.band_signals.iter().any(|(id, _)| *id == s.id)
            })
            .collect();
        if to_build.is_empty() {
            return;
        }

        let sample_rate = audio.sample_rate as f32;
        let channels = audio.channels as usize;
        let frames = audio.frame_count();
        let dry = deinterleave(audio);

        let built: Vec<(u64, Vec<Vec<f32>>)> = to_build
            .into_par_iter()
            .map(|s| {
                let mut band = vec![vec![0.0f32; frames]; channels];
                for c in 0..channels {
                    iir::band_extract(&dry[c], &mut band[c], s.freq_low, s.freq_high, sample_rate);
                }
                (s.id, band)
            })
            .collect();

        for (id, band) in built {
            if let Some(slot) = self
                .band_signals
                .iter_mut()
                .find(|(existing, _)| *existing == id)
            {
                slot.1 = band;
            } else {
                self.band_signals.push((id, band));
            }
        }
    }

    /// Union of the time windows of everything that changed, in frames, plus a
    /// small guard band. When nothing can be localised, the whole track.
    fn compute_span(
        &self,
        audio: &AudioData,
        selections: &[Selection],
        diff: &SelectionDiff,
    ) -> (usize, usize) {
        let sample_rate = audio.sample_rate.max(1) as f32;
        let frames = audio.frame_count();

        if selections.iter().any(Selection::is_polygon)
            || self.selections.iter().any(Selection::is_polygon)
        {
            return (0, frames);
        }

        let mut start = f32::INFINITY;
        let mut end = f32::NEG_INFINITY;
        let mut consider = |s: &Selection| {
            start = start.min(s.time_start);
            end = end.max(s.time_end);
        };

        for s in &self.selections {
            if diff.removed.contains(&s.id) || diff.changed.contains(&s.id) {
                consider(s);
            }
        }
        for s in &diff.added_regions {
            consider(s);
        }
        // A region whose action flipped to/from full-band changes rendering
        // globally in solo scenarios; widen to the whole track to stay safe when
        // any solo semantics are involved.
        let solo_semantics_changed = diff.changed.iter().any(|id| {
            self.selections
                .iter()
                .find(|s| &s.id == id)
                .map(|s| s.action == SelectionAction::Solo)
                .unwrap_or(false)
                || selections
                    .iter()
                    .find(|s| &s.id == id)
                    .is_some_and(|s| s.action == SelectionAction::Solo)
        }) || diff
            .added_regions
            .iter()
            .any(|s| s.action == SelectionAction::Solo)
            || diff.removed.iter().any(|id| {
                self.selections
                    .iter()
                    .any(|s| &s.id == id && s.action == SelectionAction::Solo)
            });
        // Any edit to a Solo region changes the keep/silence decision outside
        // its time/frequency bounds too. In particular, removing the last Solo
        // restores audio over the entire track, so a local splice is incorrect.
        if solo_semantics_changed {
            return (0, frames);
        }

        if !start.is_finite() || !end.is_finite() {
            return (0, frames);
        }
        // Guard band covers splice edges; the IIR warm-up overlap inside
        // band_extract already handles filter settling for the band signals.
        let guard = 0.02;
        let start_frame = (((start - guard).max(0.0)) * sample_rate).round() as usize;
        let end_frame =
            (((end + guard).min(audio.duration.max(end))) * sample_rate).round() as usize;
        (
            start_frame.min(frames),
            end_frame.clamp(start_frame, frames),
        )
    }
}

/// Result of comparing two selection sets by id.
#[derive(Default)]
struct SelectionDiff {
    /// ids present in old but not new
    removed: Vec<u64>,
    /// ids whose band edges, time window, gain or action changed
    changed: Vec<u64>,
    /// subset of `changed` whose freq_low/freq_high moved (band signal must be
    /// rebuilt; a pure time-window move must NOT rebuild)
    band_changed: Vec<u64>,
    /// full new structs for ids present in new but not old
    added_regions: Vec<Selection>,
}

impl SelectionDiff {
    fn is_noop(&self) -> bool {
        self.removed.is_empty() && self.changed.is_empty() && self.added_regions.is_empty()
    }
}

fn diff_selections(old: &[Selection], new: &[Selection]) -> SelectionDiff {
    let mut diff = SelectionDiff::default();
    for s in new {
        match old.iter().find(|o| o.id == s.id) {
            None => diff.added_regions.push(s.clone()),
            Some(o) => {
                let band_changed = (o.freq_low - s.freq_low).abs() > 1.0e-3
                    || (o.freq_high - s.freq_high).abs() > 1.0e-3;
                let time_changed = (o.time_start - s.time_start).abs() > 1.0e-4
                    || (o.time_end - s.time_end).abs() > 1.0e-4;
                let gain_changed = (o.gain - s.gain).abs() > 1.0e-4;
                let action_changed = o.action != s.action;
                let full_band_changed = o.full_band != s.full_band;
                let shape_changed = o.shape != s.shape;
                if band_changed
                    || time_changed
                    || gain_changed
                    || action_changed
                    || full_band_changed
                    || shape_changed
                {
                    diff.changed.push(s.id);
                }
                if band_changed || full_band_changed || shape_changed {
                    diff.band_changed.push(s.id);
                }
            }
        }
    }
    for o in old {
        if !new.iter().any(|s| s.id == o.id) {
            diff.removed.push(o.id);
        }
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frames: usize, freq: f32, sample_rate: u32) -> AudioData {
        AudioData {
            samples: (0..frames)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        }
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
    }

    #[test]
    fn full_band_zero_gain_preview_is_silent_in_window() {
        let audio = tone(48_000, 440.0, 48_000);
        let mut sel = Selection::new(1, 0.25, 0.5, 0.0, 24_000.0, SelectionAction::Gain, true);
        sel.gain = 0.0;
        let preview = render_full(&audio, std::slice::from_ref(&sel));
        assert!(preview[12_000..24_000].iter().all(|v| v.abs() < 1.0e-6));
        assert!(preview[1_000..11_000].iter().any(|v| v.abs() > 1.0e-3));
    }

    #[test]
    fn polygon_preview_uses_exact_stft_mask_and_disabled_regions_are_ignored() {
        let audio = tone(48_000, 440.0, 48_000);
        let mut polygon = Selection::polygon(
            31,
            vec![(0.2, 300.0), (0.8, 900.0), (0.9, 1_200.0), (0.3, 700.0)],
            SelectionAction::Gain,
        );
        polygon.gain = 0.0;
        let exact = apply_spectral_masks(&audio, std::slice::from_ref(&polygon), 2048, 512);
        assert_eq!(render_full(&audio, std::slice::from_ref(&polygon)), exact);

        polygon.enabled = false;
        assert_eq!(render_full(&audio, &[polygon]), audio.samples);
    }

    #[test]
    fn incremental_render_matches_fresh_full_render() {
        let audio = tone(96_000, 440.0, 48_000);
        let mut sel_a = Selection::new(1, 0.5, 1.0, 0.0, 24_000.0, SelectionAction::Gain, true);
        sel_a.gain = 0.0;

        let mut cache = PreviewCache::new();
        let out_a = cache.render(&audio, std::slice::from_ref(&sel_a));
        assert!(out_a.full_rebuild);

        // Shrink the window; the incremental result must equal a fresh render.
        let mut sel_b = sel_a.clone();
        sel_b.time_end = 0.75;
        let out_b = cache.render(&audio, std::slice::from_ref(&sel_b));
        assert!(!out_b.full_rebuild);
        assert!(out_b.changed_frames.is_some());
        let incremental = cache.samples.clone();

        let mut fresh = PreviewCache::new();
        fresh.render(&audio, std::slice::from_ref(&sel_b));

        assert_eq!(incremental.len(), fresh.samples.len());
        let max_err = incremental
            .iter()
            .zip(fresh.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1.0e-5, "incremental drift {max_err}");

        // And it must actually differ from the previous state.
        let unchanged = cache.render(&audio, std::slice::from_ref(&sel_b));
        assert!(unchanged.changed_frames.is_none(), "no-op not detected");
    }

    #[test]
    fn removing_last_solo_rebuilds_audio_outside_old_region() {
        let audio = tone(96_000, 440.0, 48_000);
        let solo = Selection::new(7, 0.5, 0.75, 0.0, 24_000.0, SelectionAction::Solo, true);
        let mut cache = PreviewCache::new();
        cache.render(&audio, std::slice::from_ref(&solo));
        assert!(cache.samples[..20_000].iter().all(|sample| *sample == 0.0));

        let outcome = cache.render(&audio, &[]);
        assert_eq!(outcome.changed_frames, Some((0, audio.frame_count())));
        assert!(cache.samples[..20_000]
            .iter()
            .any(|sample| sample.abs() > 1.0e-3));
        assert_eq!(cache.samples, audio.samples);
    }

    #[test]
    fn changing_gain_to_solo_rebuilds_the_entire_preview() {
        let audio = tone(96_000, 440.0, 48_000);
        let mut selection =
            Selection::new(8, 0.5, 0.75, 0.0, 24_000.0, SelectionAction::Gain, true);
        selection.gain = 0.0;
        let mut cache = PreviewCache::new();
        cache.render(&audio, std::slice::from_ref(&selection));

        selection.action = SelectionAction::Solo;
        let outcome = cache.render(&audio, std::slice::from_ref(&selection));
        assert_eq!(outcome.changed_frames, Some((0, audio.frame_count())));
        assert!(cache.samples[..20_000].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn full_band_toggle_updates_incremental_preview_to_match_fresh_render() {
        let audio = tone(96_000, 440.0, 48_000);
        let mut selection = Selection::new(9, 0.4, 0.8, 300.0, 900.0, SelectionAction::Gain, false);
        selection.gain = 0.0;
        let mut cache = PreviewCache::new();
        cache.render(&audio, std::slice::from_ref(&selection));

        selection.full_band = true;
        selection.freq_low = 0.0;
        selection.freq_high = 24_000.0;
        let outcome = cache.render(&audio, std::slice::from_ref(&selection));
        assert!(outcome.changed_frames.is_some());

        let mut fresh = PreviewCache::new();
        fresh.render(&audio, std::slice::from_ref(&selection));
        assert_eq!(cache.samples, fresh.samples);
    }

    #[test]
    fn frequency_edge_edit_refreshes_iir_band_cache() {
        let sample_rate = 48_000;
        let audio = AudioData {
            samples: (0..96_000)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                        + (2.0 * std::f32::consts::PI * 4_000.0 * t).sin()
                })
                .collect(),
            sample_rate,
            channels: 1,
            duration: 2.0,
        };
        let mut selection =
            Selection::new(11, 0.2, 1.8, 500.0, 1_000.0, SelectionAction::Gain, false);
        selection.gain = 0.0;
        let mut cache = PreviewCache::new();
        cache.render(&audio, std::slice::from_ref(&selection));

        selection.freq_low = 3_500.0;
        selection.freq_high = 4_500.0;
        let outcome = cache.render(&audio, std::slice::from_ref(&selection));
        assert!(outcome.changed_frames.is_some());

        let mut fresh = PreviewCache::new();
        fresh.render(&audio, std::slice::from_ref(&selection));
        assert_eq!(cache.samples, fresh.samples);
    }

    #[test]
    fn zero_gain_band_attenuates_without_double_silencing() {
        let sample_rate = 48_000;
        let frames = 96_000;
        let audio = AudioData {
            samples: (0..frames)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                        + (2.0 * std::f32::consts::PI * 9000.0 * t).sin()
                })
                .collect(),
            sample_rate,
            channels: 1,
            duration: frames as f32 / sample_rate as f32,
        };
        // Set gain to zero for 700-1400 Hz only between 0.5 and 1.5 s; outside that window the
        // 9 kHz tone AND the 1 kHz tone must both survive at full level.
        let mut sel = Selection::new(1, 0.5, 1.5, 700.0, 1400.0, SelectionAction::Gain, false);
        sel.gain = 0.0;
        let out = render_full(&audio, std::slice::from_ref(&sel));

        let before = 4800..19_200; // 0.1-0.4 s, outside the window
        let inside = 33_600..62_400; // 0.7-1.3 s, settled inside the window
        let out_before = rms(&out[before.clone()]);
        let dry_before = rms(&audio.samples[before]);
        assert!(
            (out_before - dry_before).abs() < 1.0e-3,
            "outside window altered: {out_before} vs {dry_before}"
        );
        let out_inside = rms(&out[inside.clone()]);
        let dry_inside = rms(&audio.samples[inside]);
        // Removing one of two equal-amplitude tones drops RMS by ~sqrt(2).
        assert!(
            out_inside < dry_inside * 0.85,
            "zero-gain band did not attenuate: {out_inside} vs {dry_inside}"
        );
        assert!(
            out_inside > dry_inside * 0.45,
            "zero-gain band over-attenuated (double silence?): {out_inside} vs {dry_inside}"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn moving_time_window_does_not_rebuild_bands() {
        let sample_rate = 48_000;
        let audio = AudioData {
            samples: (0..96_000)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                })
                .collect(),
            sample_rate,
            channels: 1,
            duration: 2.0,
        };
        let mut sel = Selection::new(1, 0.25, 0.75, 700.0, 1400.0, SelectionAction::Gain, false);
        sel.gain = 0.0;
        let mut cache = PreviewCache::new();
        cache.render(&audio, std::slice::from_ref(&sel));
        assert_eq!(cache.band_signals.len(), 1);

        // Slide the window: band definition unchanged, so the cached band
        // signal pointer content must be reused (we check the id survives and
        // no rebuild flag is raised).
        let mut moved = sel.clone();
        moved.time_start = 1.0;
        moved.time_end = 1.5;
        let out = cache.render(&audio, std::slice::from_ref(&moved));
        assert!(!out.full_rebuild);
        assert_eq!(cache.band_signals.len(), 1);
        assert_eq!(cache.band_signals[0].0, 1);
    }

    #[test]
    fn cache_rebuilds_on_audio_change() {
        let audio1 = tone(48_000, 440.0, 48_000);
        let audio2 = tone(48_000, 880.0, 48_000);
        let mut sel = Selection::new(1, 0.25, 0.5, 0.0, 24_000.0, SelectionAction::Gain, true);
        sel.gain = 0.0;
        let mut cache = PreviewCache::new();
        cache.render(&audio1, std::slice::from_ref(&sel));
        let out = cache.render(&audio2, std::slice::from_ref(&sel));
        assert!(out.full_rebuild);
        assert!(cache.samples[12_000..24_000]
            .iter()
            .all(|v| v.abs() < 1.0e-6));
    }

    #[test]
    fn stereo_preview_keeps_channels_independent() {
        let sample_rate = 48_000;
        let frames = 48_000;
        // Left = 1 kHz, right = silence.
        let mut samples = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sample_rate as f32;
            samples[i * 2] = (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
        }
        let audio = AudioData {
            samples,
            sample_rate,
            channels: 2,
            duration: 1.0,
        };
        let mut sel = Selection::new(1, 0.25, 0.75, 700.0, 1400.0, SelectionAction::Gain, false);
        sel.gain = 0.0;
        let out = render_full(&audio, std::slice::from_ref(&sel));
        // Right channel must stay silent everywhere.
        assert!(out.chunks_exact(2).all(|frame| frame[1].abs() < 1.0e-5));
        // Left channel must be attenuated inside the window.
        let inside: Vec<f32> = out[33_600..43_200].iter().step_by(2).copied().collect();
        assert!(rms(&inside) < 0.2);
    }
}
