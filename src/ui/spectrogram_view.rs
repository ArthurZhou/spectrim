use eframe::egui;

use crate::audio::types::{AudioData, Selection, SelectionAction, Spectrogram};

#[derive(Default)]
pub struct SpectrogramView {
    drag_origin: Option<egui::Pos2>,
}

pub struct ViewEvents {
    pub created: Option<(f32, f32, f32, f32, bool)>,
    pub deleted: Option<u64>,
}

impl SpectrogramView {
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        texture: Option<&egui::TextureHandle>,
        spectrogram: &Spectrogram,
        audio: &AudioData,
        selections: &[Selection],
        playhead: Option<f32>,
    ) -> ViewEvents {
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        if let Some(texture) = texture {
            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter()
                .rect_filled(rect, 0.0, egui::Color32::from_gray(24));
        }

        let duration = audio.duration.max(0.0001);
        let nyquist = audio.sample_rate as f32 / 2.0;
        let point_to_values = |point: egui::Pos2| {
            let x = ((point.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let y = (1.0 - (point.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
            (x * duration, y * nyquist)
        };

        for selection in selections {
            let left =
                rect.left() + rect.width() * (selection.time_start / duration).clamp(0.0, 1.0);
            let right =
                rect.left() + rect.width() * (selection.time_end / duration).clamp(0.0, 1.0);
            let low_y = rect.bottom()
                - rect.height() * (selection.freq_low / nyquist.max(1.0)).clamp(0.0, 1.0);
            let high_y = rect.bottom()
                - rect.height() * (selection.freq_high / nyquist.max(1.0)).clamp(0.0, 1.0);
            let selection_rect =
                egui::Rect::from_min_max(egui::pos2(left, high_y), egui::pos2(right, low_y));
            let color = match selection.action {
                SelectionAction::Mute => egui::Color32::from_rgb(95, 105, 125),
                SelectionAction::Solo => egui::Color32::from_rgb(40, 190, 185),
            };
            ui.painter()
                .rect_filled(selection_rect, 0.0, color.gamma_multiply(0.28));
            ui.painter().rect_stroke(
                selection_rect,
                0.0,
                egui::Stroke::new(2.0_f32, color),
                egui::StrokeKind::Inside,
            );
        }

        if let Some(time) = playhead {
            let x = rect.left() + rect.width() * (time / duration).clamp(0.0, 1.0);
            ui.painter().line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.5_f32, egui::Color32::WHITE),
            );
        }

        if response.drag_started() {
            self.drag_origin = response.interact_pointer_pos();
        }
        if response.dragged() {
            if let (Some(start), Some(end)) = (self.drag_origin, response.interact_pointer_pos()) {
                ui.painter().rect_filled(
                    egui::Rect::from_two_pos(start, end),
                    0.0,
                    egui::Color32::YELLOW.gamma_multiply(0.25),
                );
                ui.painter().rect_stroke(
                    egui::Rect::from_two_pos(start, end),
                    0.0,
                    egui::Stroke::new(1.5_f32, egui::Color32::YELLOW),
                    egui::StrokeKind::Inside,
                );
            }
        }

        let mut created = None;
        if response.drag_stopped() {
            if let (Some(start), Some(end)) =
                (self.drag_origin.take(), response.interact_pointer_pos())
            {
                if (start.x - end.x).abs() > 4.0 && (start.y - end.y).abs() > 4.0 {
                    let (t1, f1) = point_to_values(start);
                    let (t2, f2) = point_to_values(end);
                    let full_band = (start.y - rect.top()).abs() < 4.0
                        && (end.y - rect.bottom()).abs() < 4.0
                        || (end.y - rect.top()).abs() < 4.0
                            && (start.y - rect.bottom()).abs() < 4.0;
                    created = Some((t1.min(t2), t1.max(t2), f1.min(f2), f1.max(f2), full_band));
                }
            }
        }

        let deleted = if response.secondary_clicked() {
            response.interact_pointer_pos().and_then(|point| {
                selections
                    .iter()
                    .rev()
                    .find(|selection| {
                        let left = rect.left()
                            + rect.width() * (selection.time_start / duration).clamp(0.0, 1.0);
                        let right = rect.left()
                            + rect.width() * (selection.time_end / duration).clamp(0.0, 1.0);
                        let top = rect.bottom()
                            - rect.height()
                                * (selection.freq_high / nyquist.max(1.0)).clamp(0.0, 1.0);
                        let bottom = rect.bottom()
                            - rect.height()
                                * (selection.freq_low / nyquist.max(1.0)).clamp(0.0, 1.0);
                        point.x >= left && point.x <= right && point.y >= top && point.y <= bottom
                    })
                    .map(|selection| selection.id)
            })
        } else {
            None
        };

        let _ = spectrogram;
        ViewEvents { created, deleted }
    }
}

pub fn color_image(spectrogram: &Spectrogram) -> egui::ColorImage {
    let max_width = 8192usize;
    let width = spectrogram.n_time.min(max_width).max(1);
    let height = spectrogram.n_freq.max(1);
    let mut pixels = Vec::with_capacity(width * height);
    let range = (spectrogram.db_max - spectrogram.db_min).max(1.0e-6);
    for y in 0..height {
        let frequency = height
            .saturating_sub(1)
            .saturating_sub(y)
            .min(spectrogram.n_freq.saturating_sub(1));
        for x in 0..width {
            let start = x * spectrogram.n_time / width;
            let end = ((x + 1) * spectrogram.n_time / width)
                .max(start + 1)
                .min(spectrogram.n_time);
            let value = if start < end {
                (start..end)
                    .map(|time| spectrogram.at(frequency, time))
                    .fold(f32::NEG_INFINITY, f32::max)
            } else {
                spectrogram.db_min
            };
            let normalized = ((value - spectrogram.db_min) / range).clamp(0.0, 1.0);
            pixels.push(viridis(normalized));
        }
    }
    egui::ColorImage::new([width, height], pixels)
}

fn viridis(value: f32) -> egui::Color32 {
    // Compact approximation of the Viridis perceptual palette.
    let stops = [
        (0.0, [68, 1, 84]),
        (0.25, [59, 82, 139]),
        (0.5, [33, 145, 140]),
        (0.75, [94, 201, 98]),
        (1.0, [253, 231, 37]),
    ];
    let pair = stops
        .windows(2)
        .find(|pair| value <= pair[1].0)
        .unwrap_or(&stops[3..5]);
    let fraction = ((value - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0.0, 1.0);
    let channel = |index: usize| {
        (pair[0].1[index] as f32 * (1.0 - fraction) + pair[1].1[index] as f32 * fraction).round()
            as u8
    };
    egui::Color32::from_rgb(channel(0), channel(1), channel(2))
}
