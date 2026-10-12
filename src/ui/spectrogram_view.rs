use eframe::egui;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio::spectrogram::SpectrogramPatch;
use crate::audio::types::{AudioData, Selection, SelectionAction, SelectionShape, Spectrogram};

/// Half-width of the playhead notch triangle. Doubles as its grab radius, so
/// the visual affordance and the hit area always agree.
const PLAYHEAD_GRAB_X: f32 = 7.0;
/// Height of the notch triangle.
const PLAYHEAD_NOTCH_H: f32 = 11.0;
/// Extra slack around the notch when hit-testing, so the thin line itself is
/// still easy to grab.
const PLAYHEAD_HIT_SLACK: f32 = 3.0;

#[derive(Default)]
pub struct SpectrogramView {
    drag: Option<DragMode>,
    time_range: Option<(f32, f32)>,
    frequency_range: Option<(f32, f32)>,
    detail: Option<DetailTexture>,
    polygon_tool: bool,
    polygon_points: Vec<(f32, f32)>,
}

struct DetailTexture {
    texture: egui::TextureHandle,
    time_start: f32,
    time_end: f32,
}

#[derive(Clone, Copy)]
enum DragMode {
    Create(egui::Pos2),
    Move {
        id: u64,
        last: egui::Pos2,
    },
    Resize {
        id: u64,
        corner: ResizeCorner,
        last: egui::Pos2,
    },
    Playhead,
    Pan {
        last: egui::Pos2,
    },
    PolygonVertex {
        id: u64,
        index: usize,
    },
}

#[derive(Clone, Copy)]
enum ResizeCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

pub struct ViewEvents {
    pub created: Option<(f32, f32, f32, f32, bool)>,
    pub deleted: Option<u64>,
    pub selected: Option<Option<u64>>,
    pub seek_to: Option<f32>,
    pub selections_changed: bool,
    pub drag_started: bool,
    pub drag_stopped: bool,
    pub pointer_coordinates: Option<(f32, f32)>,
    pub visible_ranges: (f32, f32, f32, f32),
    pub view_changed: bool,
    pub created_polygon: Option<Vec<(f32, f32)>>,
}

impl SpectrogramView {
    pub fn set_detail(&mut self, texture: egui::TextureHandle, time_start: f32, time_end: f32) {
        self.detail = Some(DetailTexture {
            texture,
            time_start,
            time_end,
        });
    }

    pub fn clear_detail(&mut self) {
        self.detail = None;
    }

    pub fn set_polygon_tool(&mut self, active: bool) {
        if self.polygon_tool != active {
            self.polygon_tool = active;
            self.polygon_points.clear();
            self.drag = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        texture: Option<&egui::TextureHandle>,
        spectrogram: &Spectrogram,
        audio: &AudioData,
        selections: &mut [Selection],
        selected_id: Option<u64>,
        playhead: f32,
        polygon_tool: bool,
    ) -> ViewEvents {
        let outer = ui.available_rect_before_wrap();
        let plot = egui::Rect::from_min_max(
            egui::pos2(outer.left() + 58.0, outer.top() + 8.0),
            egui::pos2(outer.right() - 12.0, outer.bottom() - 32.0),
        );
        let _ = ui.allocate_rect(outer, egui::Sense::hover());
        let response = ui.interact(
            plot,
            ui.id().with("spectrogram_plot"),
            egui::Sense::click_and_drag(),
        );

        let duration = audio.duration.max(0.001);
        let nyquist = (audio.sample_rate as f32 * 0.5).max(1.0);
        self.set_polygon_tool(polygon_tool);
        let (time_start, time_end) = self.time_range.unwrap_or((0.0, duration));
        let (freq_low, freq_high) = self.frequency_range.unwrap_or((0.0, nyquist));
        let time_span = (time_end - time_start).max(0.001);
        let freq_span = (freq_high - freq_low).max(1.0);
        let time_to_x = |time: f32| plot.left() + plot.width() * (time - time_start) / time_span;
        let hz_to_y = |hz: f32| plot.bottom() - plot.height() * (hz - freq_low) / freq_span;
        let point_to_values = |point: egui::Pos2| {
            let time = time_start + (point.x - plot.left()) / plot.width() * time_span;
            let hz = freq_low + (plot.bottom() - point.y) / plot.height() * freq_span;
            (time.clamp(0.0, duration), hz.clamp(0.0, nyquist))
        };

        ui.painter()
            .rect_filled(plot, 0.0, egui::Color32::from_gray(20));
        if let Some(texture) = texture {
            // Color image rows are high-to-low frequency, so UV y is mapped high-to-low as well.
            let uv = egui::Rect::from_min_max(
                egui::pos2(time_start / duration, 1.0 - freq_high / nyquist),
                egui::pos2(time_end / duration, 1.0 - freq_low / nyquist),
            );
            ui.painter()
                .image(texture.id(), plot, uv, egui::Color32::WHITE);
        }
        if let Some(detail) = &self.detail {
            let visible_start = time_start.max(detail.time_start);
            let visible_end = time_end.min(detail.time_end);
            if visible_end > visible_start {
                let target = egui::Rect::from_min_max(
                    egui::pos2(time_to_x(visible_start), plot.top()),
                    egui::pos2(time_to_x(visible_end), plot.bottom()),
                );
                let local_duration = (detail.time_end - detail.time_start).max(f32::EPSILON);
                let uv = egui::Rect::from_min_max(
                    egui::pos2(
                        (visible_start - detail.time_start) / local_duration,
                        1.0 - freq_high / nyquist,
                    ),
                    egui::pos2(
                        (visible_end - detail.time_start) / local_duration,
                        1.0 - freq_low / nyquist,
                    ),
                );
                ui.painter()
                    .image(detail.texture.id(), target, uv, egui::Color32::WHITE);
            }
        }

        let mut shapes = Vec::new();
        let time_ticks = 5;
        for index in 0..=time_ticks {
            let fraction = index as f32 / time_ticks as f32;
            let time = time_start + fraction * time_span;
            let x = plot.left() + fraction * plot.width();
            shapes.push(egui::Shape::line_segment(
                [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
                egui::Stroke::new(0.5_f32, egui::Color32::from_white_alpha(70)),
            ));
            ui.painter().text(
                egui::pos2(x, plot.bottom() + 5.0),
                egui::Align2::CENTER_TOP,
                format_time(time),
                egui::FontId::monospace(10.0),
                egui::Color32::LIGHT_GRAY,
            );
        }
        for index in 0..=5 {
            let fraction = index as f32 / 5.0;
            let hz = freq_low + (1.0 - fraction) * freq_span;
            let y = plot.top() + fraction * plot.height();
            shapes.push(egui::Shape::line_segment(
                [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
                egui::Stroke::new(0.5_f32, egui::Color32::from_white_alpha(70)),
            ));
            ui.painter().text(
                egui::pos2(plot.left() - 6.0, y),
                egui::Align2::RIGHT_CENTER,
                format_frequency(hz),
                egui::FontId::monospace(10.0),
                egui::Color32::LIGHT_GRAY,
            );
        }
        ui.painter().extend(shapes);

        for selection in selections.iter() {
            let selection_alpha = if selection.enabled { 1.0 } else { 0.35 };
            if let SelectionShape::Polygon { points } = &selection.shape {
                let screen_points = points
                    .iter()
                    .map(|(time, hz)| egui::pos2(time_to_x(*time), hz_to_y(*hz)))
                    .collect::<Vec<_>>();
                let color = match selection.action {
                    SelectionAction::Solo => egui::Color32::from_rgb(40, 190, 185),
                    SelectionAction::Gain => egui::Color32::from_rgb(250, 170, 50),
                };
                draw_polygon_fill(
                    ui.painter(),
                    &screen_points,
                    color.gamma_multiply(selection_alpha),
                );
                let stroke = egui::Stroke::new(
                    if selected_id == Some(selection.id) {
                        2.5
                    } else {
                        1.5
                    },
                    color.gamma_multiply(selection_alpha),
                );
                if screen_points.len() >= 2 {
                    for pair in screen_points.windows(2) {
                        ui.painter().line_segment([pair[0], pair[1]], stroke);
                    }
                    if let (Some(first), Some(last)) = (screen_points.first(), screen_points.last())
                    {
                        ui.painter().line_segment([*last, *first], stroke);
                    }
                }
                for (index, point) in screen_points.iter().enumerate() {
                    ui.painter().circle_filled(
                        *point,
                        if index == 0 { 5.0 } else { 3.5 },
                        if index == 0 {
                            egui::Color32::WHITE
                        } else {
                            stroke.color
                        },
                    );
                    if selected_id == Some(selection.id) {
                        ui.painter().circle_stroke(
                            *point,
                            7.0,
                            egui::Stroke::new(1.0, egui::Color32::WHITE),
                        );
                    }
                }
                continue;
            }
            let left = time_to_x(selection.time_start).clamp(plot.left(), plot.right());
            let right = time_to_x(selection.time_end).clamp(plot.left(), plot.right());
            let (selection_low, selection_high) = if selection.full_band {
                (0.0, nyquist)
            } else {
                (selection.freq_low, selection.freq_high)
            };
            let top = hz_to_y(selection_high).clamp(plot.top(), plot.bottom());
            let bottom = hz_to_y(selection_low).clamp(plot.top(), plot.bottom());
            let rect = egui::Rect::from_min_max(egui::pos2(left, top), egui::pos2(right, bottom));
            let color = match selection.action {
                SelectionAction::Solo => egui::Color32::from_rgb(40, 190, 185),
                SelectionAction::Gain => egui::Color32::from_rgb(250, 170, 50),
            };
            ui.painter()
                .rect_filled(rect, 0.0, color.gamma_multiply(0.25 * selection_alpha));
            ui.painter().rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(
                    if selected_id == Some(selection.id) {
                        2.5_f32
                    } else {
                        1.5_f32
                    },
                    color.gamma_multiply(selection_alpha),
                ),
                egui::StrokeKind::Inside,
            );
            if selected_id == Some(selection.id) {
                for handle in [
                    egui::pos2(left, top),
                    egui::pos2(right, top),
                    egui::pos2(left, bottom),
                    egui::pos2(right, bottom),
                ] {
                    ui.painter()
                        .circle_filled(handle, 4.0, egui::Color32::WHITE);
                }
            }
        }

        // --- Playhead -----------------------------------------------------
        // Always drawn, with grabbable notch handles at both ends.
        //
        // `playhead_raw_x` is the true (unclamped) screen position and
        // `playhead_in_view` says whether it falls inside the visible time
        // window. The previous code only kept the *clamped* x, which had two
        // problems: the handle vanished as soon as you zoomed away from it, and
        // the drag hit-test compared against the clamped edge, so the entire
        // left/right edge of the plot behaved like the playhead and swallowed
        // region-creation drags. Keeping both fixes the hit-test and lets us
        // draw an off-screen indicator instead of losing the handle.
        let playhead_time = playhead.clamp(0.0, duration);
        let playhead_raw_x = time_to_x(playhead_time);
        let playhead_in_view = playhead_raw_x >= plot.left() && playhead_raw_x <= plot.right();
        let playhead_x = playhead_raw_x.clamp(plot.left(), plot.right());
        let dragging_playhead = matches!(self.drag, Some(DragMode::Playhead));
        let notch_fill = if dragging_playhead {
            egui::Color32::from_rgb(255, 228, 130)
        } else {
            egui::Color32::from_rgb(255, 186, 66)
        };
        let notch_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_black_alpha(170));
        let painter = ui.painter();

        if playhead_in_view {
            // Dark outline under the white line keeps it legible over the bright
            // yellow/green part of the viridis colormap.
            painter.line_segment(
                [
                    egui::pos2(playhead_x, plot.top()),
                    egui::pos2(playhead_x, plot.bottom()),
                ],
                egui::Stroke::new(4.0_f32, egui::Color32::from_black_alpha(110)),
            );
            painter.line_segment(
                [
                    egui::pos2(playhead_x, plot.top()),
                    egui::pos2(playhead_x, plot.bottom()),
                ],
                egui::Stroke::new(1.5_f32, egui::Color32::WHITE),
            );
            // Top notch pointing down into the plot.
            painter.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(playhead_x - PLAYHEAD_GRAB_X, plot.top()),
                    egui::pos2(playhead_x + PLAYHEAD_GRAB_X, plot.top()),
                    egui::pos2(playhead_x, plot.top() + PLAYHEAD_NOTCH_H),
                ],
                notch_fill,
                notch_stroke,
            ));
            // Bottom notch pointing up, so the handle is reachable from either
            // end of a tall plot.
            painter.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(playhead_x - PLAYHEAD_GRAB_X, plot.bottom()),
                    egui::pos2(playhead_x + PLAYHEAD_GRAB_X, plot.bottom()),
                    egui::pos2(playhead_x, plot.bottom() - PLAYHEAD_NOTCH_H),
                ],
                notch_fill,
                notch_stroke,
            ));
            draw_playhead_label(painter, playhead_time, playhead_x, notch_fill, plot, true);
        } else {
            // Off-screen: park an arrow at the near edge pointing toward the
            // playhead so the position is never invisible.
            let on_left = playhead_raw_x < plot.left();
            let edge_x = if on_left { plot.left() } else { plot.right() };
            // Base sits inward from the edge; the apex touches the edge.
            let inward = if on_left { 1.0 } else { -1.0 };
            let mid_y = plot.center().y;
            painter.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(edge_x, mid_y),
                    egui::pos2(edge_x + inward * (PLAYHEAD_NOTCH_H + 3.0), mid_y - 8.0),
                    egui::pos2(edge_x + inward * (PLAYHEAD_NOTCH_H + 3.0), mid_y + 8.0),
                ],
                notch_fill,
                notch_stroke,
            ));
            draw_playhead_label(
                painter,
                playhead_time,
                edge_x + inward * (PLAYHEAD_NOTCH_H + 9.0),
                notch_fill,
                plot,
                !on_left,
            );
        }

        let mut events = ViewEvents {
            created: None,
            deleted: None,
            selected: None,
            seek_to: None,
            selections_changed: false,
            drag_started: false,
            drag_stopped: false,
            pointer_coordinates: None,
            visible_ranges: (time_start, time_end, freq_low, freq_high),
            view_changed: false,
            created_polygon: None,
        };

        let hover_pointer = ui
            .input(|input| input.pointer.hover_pos())
            .filter(|p| plot.contains(*p));
        if self.polygon_tool {
            if !self.polygon_points.is_empty() {
                let screen_points = self
                    .polygon_points
                    .iter()
                    .map(|(time, hz)| egui::pos2(time_to_x(*time), hz_to_y(*hz)))
                    .collect::<Vec<_>>();
                for pair in screen_points.windows(2) {
                    ui.painter().line_segment(
                        [pair[0], pair[1]],
                        egui::Stroke::new(2.0, egui::Color32::YELLOW),
                    );
                }
                for (index, point) in screen_points.iter().enumerate() {
                    ui.painter().circle_filled(
                        *point,
                        if index == 0 { 6.0 } else { 4.0 },
                        if index == 0 {
                            egui::Color32::WHITE
                        } else {
                            egui::Color32::YELLOW
                        },
                    );
                }
                if let (Some(cursor), Some(last)) = (hover_pointer, screen_points.last()) {
                    let near_first = screen_points
                        .first()
                        .is_some_and(|first| cursor.distance(*first) <= 10.0)
                        && screen_points.len() >= 3;
                    draw_dashed_line(
                        ui.painter(),
                        *last,
                        cursor,
                        if near_first {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::YELLOW
                        },
                    );
                }
            }

            // Use the raw wheel delta and test the plot's actual hover position.
            // `response.hovered()` can be false when another overlapping egui
            // response (for example the enclosing panel) owns the pointer event.
            if let Some(pointer) = ui
                .input(|input| input.pointer.hover_pos())
                .filter(|position| plot.contains(*position))
            {
                let scroll_y = ui.input(|input| input.raw_scroll_delta.y);
                if scroll_y.abs() > f32::EPSILON {
                    let (anchor_time, anchor_hz) = point_to_values(pointer);
                    // Wheel up zooms in; wheel down zooms out. Exponential scaling
                    // behaves consistently for both mouse wheels and touchpads.
                    let scale = (-scroll_y * 0.003).exp().clamp(0.5, 2.0);
                    let zoom_frequency = ui.input(|input| input.modifiers.ctrl);
                    if zoom_frequency {
                        let new_span = (freq_span * scale).clamp(100.0_f32.min(nyquist), nyquist);
                        let fraction =
                            ((plot.bottom() - pointer.y) / plot.height()).clamp(0.0, 1.0);
                        let low = (anchor_hz - fraction * new_span)
                            .clamp(0.0, (nyquist - new_span).max(0.0));
                        self.frequency_range = Some((low, low + new_span));
                    } else {
                        let new_span = (time_span * scale).clamp(0.05_f32.min(duration), duration);
                        let fraction = ((pointer.x - plot.left()) / plot.width()).clamp(0.0, 1.0);
                        let start = (anchor_time - fraction * new_span)
                            .clamp(0.0, (duration - new_span).max(0.0));
                        self.time_range = Some((start, start + new_span));
                    }
                    events.view_changed = true;
                    ui.ctx().request_repaint();
                }
            }

            let enter = ui.input(|input| input.key_pressed(egui::Key::Enter));
            let escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
            let right_clicked = response.clicked_by(egui::PointerButton::Secondary);
            if escape || right_clicked {
                self.polygon_points.clear();
            } else {
                let double_clicked = response.double_clicked_by(egui::PointerButton::Primary);
                let clicked = response.clicked_by(egui::PointerButton::Primary);
                let close_near_first = clicked
                    && self.polygon_points.len() >= 3
                    && hover_pointer.is_some_and(|cursor| {
                        let first = egui::pos2(
                            time_to_x(self.polygon_points[0].0),
                            hz_to_y(self.polygon_points[0].1),
                        );
                        cursor.distance(first) <= 10.0
                    });
                if double_clicked && self.polygon_points.len() == 2 {
                    if let Some(position) = hover_pointer {
                        let point = point_to_values(position);
                        let duplicate = self.polygon_points.last().is_some_and(|previous| {
                            let dx = (point.0 - previous.0) * plot.width() / time_span;
                            let dy = (point.1 - previous.1) * plot.height() / freq_span;
                            dx.hypot(dy) < 3.0
                        });
                        if !duplicate {
                            self.polygon_points.push(point);
                        }
                    }
                }
                if enter || double_clicked || close_near_first {
                    if self.polygon_points.len() >= 3
                        && polygon_area_pixels(&self.polygon_points, &time_to_x, &hz_to_y) >= 16.0
                    {
                        events.created_polygon = Some(std::mem::take(&mut self.polygon_points));
                    } else if double_clicked || enter {
                        self.polygon_points.clear();
                    }
                } else if clicked {
                    if let Some(position) = hover_pointer {
                        let point = point_to_values(position);
                        let duplicate = self.polygon_points.iter().any(|previous| {
                            let dx = (point.0 - previous.0) * plot.width() / time_span;
                            let dy = (point.1 - previous.1) * plot.height() / freq_span;
                            dx.hypot(dy) < 3.0
                        });
                        if !duplicate {
                            self.polygon_points.push(point);
                        }
                    }
                }
            }
        }

        // --- Standard Interaction / Pan -------------------------------------
        if let Some(position) = hover_pointer {
            let (time, hz) = point_to_values(position);
            events.pointer_coordinates = Some((time, hz));
            response
                .clone()
                .on_hover_text(format!("Time: {:.3} s\nFrequency: {:.1} Hz", time, hz));
            // Advertise that the playhead can be grabbed, and do so on the
            // unclamped position so the plot edges stay free for creating
            // regions.
            if dragging_playhead
                || (playhead_in_view
                    && (position.x - playhead_raw_x).abs() <= PLAYHEAD_GRAB_X + PLAYHEAD_HIT_SLACK)
            {
                ui.ctx()
                    .output_mut(|output| output.cursor_icon = egui::CursorIcon::ResizeColumn);
            }
        }
        // Keep drag gestures alive if the pointer is released just outside the
        // plot, while only publishing coordinate readouts for points in it.
        let pointer = response.interact_pointer_pos();

        if !self.polygon_tool {
            // Standard zooming and panning are disabled when polygon_tool is active
            // (as it replaces them with polygon-specific logic above).

            // Use the raw wheel delta and test the plot's actual hover position.
            if let Some(pointer) = ui
                .input(|input| input.pointer.hover_pos())
                .filter(|position| plot.contains(*position))
            {
                let scroll_y = ui.input(|input| input.raw_scroll_delta.y);
                if scroll_y.abs() > f32::EPSILON {
                    let (anchor_time, anchor_hz) = point_to_values(pointer);
                    let scale = (-scroll_y * 0.003).exp().clamp(0.5, 2.0);
                    let zoom_frequency = ui.input(|input| input.modifiers.ctrl);
                    if zoom_frequency {
                        let new_span = (freq_span * scale).clamp(100.0_f32.min(nyquist), nyquist);
                        let fraction =
                            ((plot.bottom() - pointer.y) / plot.height()).clamp(0.0, 1.0);
                        let low = (anchor_hz - fraction * new_span)
                            .clamp(0.0, (nyquist - new_span).max(0.0));
                        self.frequency_range = Some((low, low + new_span));
                    } else {
                        let new_span = (time_span * scale).clamp(0.05_f32.min(duration), duration);
                        let fraction = ((pointer.x - plot.left()) / plot.width()).clamp(0.0, 1.0);
                        let start = (anchor_time - fraction * new_span)
                            .clamp(0.0, (duration - new_span).max(0.0));
                        self.time_range = Some((start, start + new_span));
                    }
                    events.view_changed = true;
                    ui.ctx().request_repaint();
                }
            }

            // Pan the visible time/frequency window with right-button drag. Read
            // raw pointer state because this plot response is configured for the
            // primary-button region gestures only.
            let (secondary_pointer, secondary_pressed, secondary_down, secondary_released) = ui
                .input(|input| {
                    (
                        input.pointer.hover_pos(),
                        input.pointer.button_pressed(egui::PointerButton::Secondary),
                        input.pointer.button_down(egui::PointerButton::Secondary),
                        input
                            .pointer
                            .button_released(egui::PointerButton::Secondary),
                    )
                });
            if secondary_pressed
                && secondary_pointer.is_some_and(|position| plot.contains(position))
            {
                if let Some(position) = secondary_pointer {
                    self.drag = Some(DragMode::Pan { last: position });
                }
            }
            if secondary_down {
                if let (Some(DragMode::Pan { last }), Some(position)) =
                    (self.drag, secondary_pointer)
                {
                    let time_delta = -(position.x - last.x) / plot.width() * time_span;
                    let frequency_delta = (position.y - last.y) / plot.height() * freq_span;
                    let new_time_start =
                        (time_start + time_delta).clamp(0.0, (duration - time_span).max(0.0));
                    let new_frequency_low =
                        (freq_low + frequency_delta).clamp(0.0, (nyquist - freq_span).max(0.0));
                    self.time_range = Some((new_time_start, new_time_start + time_span));
                    self.frequency_range = Some((new_frequency_low, new_frequency_low + freq_span));
                    events.view_changed = true;
                    self.drag = Some(DragMode::Pan { last: position });
                    ui.ctx().request_repaint();
                }
            }
            if secondary_released && matches!(self.drag, Some(DragMode::Pan { .. })) {
                self.drag = None;
            }
        }

        if !self.polygon_tool && response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(position) = pointer {
                events.drag_started = true;
                if !self.polygon_tool {
                    let vertex_for = |selection: &Selection| match &selection.shape {
                        SelectionShape::Polygon { points } => points
                            .iter()
                            .enumerate()
                            .map(|(index, (time, hz))| {
                                (index, egui::pos2(time_to_x(*time), hz_to_y(*hz)))
                            })
                            .find(|(_, point)| point.distance(position) <= 9.0)
                            .map(|(index, _)| (selection.id, index)),
                        SelectionShape::Rect => None,
                    };
                    let vertex = selected_id
                        .and_then(|id| selections.iter().find(|selection| selection.id == id))
                        .and_then(vertex_for)
                        .or_else(|| selections.iter().rev().find_map(vertex_for));
                    if let Some((id, index)) = vertex {
                        self.drag = Some(DragMode::PolygonVertex { id, index });
                        events.selected = Some(Some(id));
                    }
                }
                if matches!(self.drag, Some(DragMode::PolygonVertex { .. })) {
                    // Vertex drag has priority over playhead/rectangle operations.
                } else if self.polygon_tool {
                    // Polygon clicks are handled by the multi-click state machine below.
                } else if playhead_in_view
                    && (position.x - playhead_raw_x).abs() <= PLAYHEAD_GRAB_X + PLAYHEAD_HIT_SLACK
                {
                    self.drag = Some(DragMode::Playhead);
                    events.seek_to = Some(point_to_values(position).0);
                } else if let Some(selected) = selected_id
                    .and_then(|id| selections.iter().find(|selection| selection.id == id))
                {
                    if let SelectionShape::Polygon { points } = &selected.shape {
                        if point_in_polygon(point_to_values(position), points) {
                            events.selected = Some(Some(selected.id));
                        } else {
                            self.drag = Some(DragMode::Create(position));
                        }
                    } else {
                        let x0 = time_to_x(selected.time_start);
                        let x1 = time_to_x(selected.time_end);
                        let y0 = hz_to_y(selected.freq_high);
                        let y1 = hz_to_y(selected.freq_low);
                        let corner = [
                            (egui::pos2(x0, y0), ResizeCorner::TopLeft),
                            (egui::pos2(x1, y0), ResizeCorner::TopRight),
                            (egui::pos2(x0, y1), ResizeCorner::BottomLeft),
                            (egui::pos2(x1, y1), ResizeCorner::BottomRight),
                        ]
                        .into_iter()
                        .find(|(handle, _)| handle.distance(position) <= 9.0)
                        .map(|(_, corner)| corner);
                        if let Some(corner) = corner {
                            events.selected = Some(Some(selected.id));
                            self.drag = Some(DragMode::Resize {
                                id: selected.id,
                                corner,
                                last: position,
                            });
                        } else {
                            let inside_selected = position.x >= x0
                                && position.x <= time_to_x(selected.time_end)
                                && position.y >= y0
                                && position.y <= y1;
                            if inside_selected {
                                events.selected = Some(Some(selected.id));
                                self.drag = Some(DragMode::Move {
                                    id: selected.id,
                                    last: position,
                                });
                            } else {
                                self.drag = Some(DragMode::Create(position));
                            }
                        }
                    }
                } else if let Some(selection) = selections
                    .iter()
                    .rev()
                    .find(|selection| point_in_selection(point_to_values(position), selection))
                {
                    events.selected = Some(Some(selection.id));
                    if !selection.is_polygon() {
                        self.drag = Some(DragMode::Move {
                            id: selection.id,
                            last: position,
                        });
                    }
                } else {
                    self.drag = Some(DragMode::Create(position));
                }
            }
        }

        if self.polygon_tool {
            let enter = ui.input(|input| input.key_pressed(egui::Key::Enter));
            let escape = ui.input(|input| input.key_pressed(egui::Key::Escape));
            let right_clicked = response.clicked_by(egui::PointerButton::Secondary);
            if escape || right_clicked {
                self.polygon_points.clear();
            } else {
                let double_clicked = response.double_clicked_by(egui::PointerButton::Primary);
                let clicked = response.clicked_by(egui::PointerButton::Primary);
                let close_near_first = clicked
                    && self.polygon_points.len() >= 3
                    && hover_pointer.is_some_and(|cursor| {
                        let first = egui::pos2(
                            time_to_x(self.polygon_points[0].0),
                            hz_to_y(self.polygon_points[0].1),
                        );
                        cursor.distance(first) <= 10.0
                    });
                if enter || double_clicked || close_near_first {
                    if self.polygon_points.len() >= 3
                        && polygon_area(&self.polygon_points).abs() > 1.0e-6
                    {
                        events.created_polygon = Some(std::mem::take(&mut self.polygon_points));
                    } else if double_clicked || enter {
                        self.polygon_points.clear();
                    }
                } else if clicked {
                    if let Some(position) = hover_pointer {
                        let point = point_to_values(position);
                        let duplicate = self.polygon_points.last().is_some_and(|previous| {
                            let dx = (point.0 - previous.0) * plot.width() / time_span;
                            let dy = (point.1 - previous.1) * plot.height() / freq_span;
                            dx.hypot(dy) < 3.0
                        });
                        if !duplicate {
                            self.polygon_points.push(point);
                        }
                    }
                }
            }
        }

        if !self.polygon_tool && response.dragged_by(egui::PointerButton::Primary) {
            if let Some(position) = pointer {
                match self.drag {
                    Some(DragMode::PolygonVertex { id, index }) => {
                        if let Some(selection) = selections.iter_mut().find(|s| s.id == id) {
                            if let SelectionShape::Polygon { points } = &mut selection.shape {
                                if let Some(point) = points.get_mut(index) {
                                    *point = point_to_values(position);
                                    selection.refresh_bounds();
                                    events.selections_changed = true;
                                }
                            }
                        }
                    }
                    Some(DragMode::Playhead) => events.seek_to = Some(point_to_values(position).0),
                    Some(DragMode::Move { id, last }) => {
                        let dt = (position.x - last.x) / plot.width() * time_span;
                        let df = -(position.y - last.y) / plot.height() * freq_span;
                        if let Some(selection) = selections.iter_mut().find(|s| s.id == id) {
                            let time_delta =
                                dt.clamp(-selection.time_start, duration - selection.time_end);
                            let frequency_delta = if selection.full_band {
                                0.0
                            } else {
                                df.clamp(-selection.freq_low, nyquist - selection.freq_high)
                            };
                            selection.time_start += time_delta;
                            selection.time_end += time_delta;
                            selection.freq_low += frequency_delta;
                            selection.freq_high += frequency_delta;
                            if selection.full_band {
                                selection.freq_low = 0.0;
                                selection.freq_high = nyquist;
                            }
                            events.selections_changed = time_delta != 0.0 || frequency_delta != 0.0;
                        }
                        self.drag = Some(DragMode::Move { id, last: position });
                    }
                    Some(DragMode::Resize { id, corner, last }) => {
                        let delta_time = (position.x - last.x) / plot.width() * time_span;
                        let delta_frequency = -(position.y - last.y) / plot.height() * freq_span;
                        if let Some(selection) = selections.iter_mut().find(|s| s.id == id) {
                            match corner {
                                ResizeCorner::TopLeft | ResizeCorner::BottomLeft => {
                                    let minimum = (selection.time_end - 0.001).max(0.0);
                                    selection.time_start =
                                        (selection.time_start + delta_time).clamp(0.0, minimum);
                                }
                                ResizeCorner::TopRight | ResizeCorner::BottomRight => {
                                    let minimum = (selection.time_start + 0.001).min(duration);
                                    selection.time_end =
                                        (selection.time_end + delta_time).clamp(minimum, duration);
                                }
                            }
                            if selection.full_band {
                                selection.freq_low = 0.0;
                                selection.freq_high = nyquist;
                            } else {
                                match corner {
                                    ResizeCorner::TopLeft | ResizeCorner::TopRight => {
                                        let minimum = (selection.freq_low + 1.0).min(nyquist);
                                        selection.freq_high = (selection.freq_high
                                            + delta_frequency)
                                            .clamp(minimum, nyquist);
                                    }
                                    ResizeCorner::BottomLeft | ResizeCorner::BottomRight => {
                                        let maximum = (selection.freq_high - 1.0).max(0.0);
                                        selection.freq_low = (selection.freq_low + delta_frequency)
                                            .clamp(0.0, maximum);
                                    }
                                }
                            }
                            events.selections_changed = delta_time != 0.0
                                || (!selection.full_band && delta_frequency != 0.0);
                        }
                        self.drag = Some(DragMode::Resize {
                            id,
                            corner,
                            last: position,
                        });
                    }
                    Some(DragMode::Create(start)) => {
                        ui.painter().rect_stroke(
                            egui::Rect::from_two_pos(start, position),
                            0.0,
                            egui::Stroke::new(1.5_f32, egui::Color32::YELLOW),
                            egui::StrokeKind::Inside,
                        );
                    }
                    _ => {}
                }
            }
        }

        if !self.polygon_tool && response.drag_stopped() {
            events.drag_stopped = true;
            if let (Some(DragMode::Create(start)), Some(end)) = (self.drag, pointer) {
                if (start.x - end.x).abs() > 4.0 && (start.y - end.y).abs() > 4.0 {
                    let (t1, f1) = point_to_values(start);
                    let (t2, f2) = point_to_values(end);
                    let full_band = ((start.y - plot.top()).abs() < 5.0
                        && (end.y - plot.bottom()).abs() < 5.0)
                        || ((end.y - plot.top()).abs() < 5.0
                            && (start.y - plot.bottom()).abs() < 5.0);
                    events.created = Some((
                        t1.min(t2),
                        t1.max(t2),
                        if full_band { 0.0 } else { f1.min(f2) },
                        if full_band { nyquist } else { f1.max(f2) },
                        full_band,
                    ));
                }
            }
            self.drag = None;
        }

        if response.secondary_clicked() {
            if !self.polygon_tool {
                events.deleted = pointer.and_then(|position| {
                    selections
                        .iter()
                        .rev()
                        .find(|selection| point_in_selection(point_to_values(position), selection))
                        .map(|selection| selection.id)
                });
            }
        }

        events.visible_ranges = (
            self.time_range.map_or(0.0, |range| range.0),
            self.time_range.map_or(duration, |range| range.1),
            self.frequency_range.map_or(0.0, |range| range.0),
            self.frequency_range.map_or(nyquist, |range| range.1),
        );
        let _ = spectrogram;
        events
    }
}

/// Draw the playhead's time readout next to its notch.
///
/// `anchor_x` is the handle position and `anchor_left_of_label` says whether the
/// label should extend to the right of it (it flips near the plot's right edge so
/// the readout is never clipped off-screen).
fn draw_playhead_label(
    painter: &egui::Painter,
    time: f32,
    anchor_x: f32,
    color: egui::Color32,
    plot: egui::Rect,
    anchor_left_of_label: bool,
) {
    let text = format!("{time:.2}s");
    let font = egui::FontId::monospace(10.0);
    let galley = painter.layout_no_wrap(text.clone(), font, color);
    let padding = 3.0_f32;
    let width = galley.size().x + padding * 2.0;
    let height = galley.size().y + padding;
    let gap = 4.0_f32;

    let mut left = if anchor_left_of_label {
        anchor_x + gap
    } else {
        anchor_x - gap - width
    };
    // Keep the box inside the plot; flip to the other side if it would overflow.
    if left + width > plot.right() {
        left = anchor_x - gap - width;
    }
    if left < plot.left() {
        left = anchor_x + gap;
    }
    left = left.clamp(plot.left(), (plot.right() - width).max(plot.left()));

    let rect = egui::Rect::from_min_size(
        egui::pos2(left, plot.top() + 2.0),
        egui::vec2(width, height),
    );
    painter.rect_filled(rect, 3.0, egui::Color32::from_black_alpha(175));
    painter.galley(rect.center() - galley.size() * 0.5, galley, color);
}

fn format_time(time: f32) -> String {
    let seconds = time.max(0.0) as u32;
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

fn format_frequency(hz: f32) -> String {
    if hz >= 1000.0 {
        format!("{:.1}k", hz / 1000.0)
    } else {
        format!("{hz:.0}")
    }
}

fn polygon_area(points: &[(f32, f32)]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    (0..points.len())
        .map(|index| {
            let (x1, y1) = points[index];
            let (x2, y2) = points[(index + 1) % points.len()];
            x1 * y2 - x2 * y1
        })
        .sum::<f32>()
        * 0.5
}

fn polygon_area_pixels(
    points: &[(f32, f32)],
    time_to_x: &impl Fn(f32) -> f32,
    hz_to_y: &impl Fn(f32) -> f32,
) -> f32 {
    let screen = points
        .iter()
        .map(|(time, hz)| (time_to_x(*time), hz_to_y(*hz)))
        .collect::<Vec<_>>();
    polygon_area(&screen).abs()
}

fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let (x, y) = point;
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let (xi, yi) = polygon[current];
        let (xj, yj) = polygon[previous];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn point_in_selection(point: (f32, f32), selection: &Selection) -> bool {
    match &selection.shape {
        SelectionShape::Rect => {
            point.0 >= selection.time_start
                && point.0 <= selection.time_end
                && point.1 >= selection.freq_low
                && point.1 <= selection.freq_high
        }
        SelectionShape::Polygon { points } => point_in_polygon(point, points),
    }
}

fn draw_dashed_line(
    painter: &egui::Painter,
    start: egui::Pos2,
    end: egui::Pos2,
    color: egui::Color32,
) {
    let delta = end - start;
    let length = delta.length();
    if length <= f32::EPSILON {
        return;
    }
    let direction = delta / length;
    let dash = 6.0;
    let gap = 4.0;
    let mut offset = 0.0;
    while offset < length {
        let finish = (offset + dash).min(length);
        painter.line_segment(
            [start + direction * offset, start + direction * finish],
            egui::Stroke::new(1.5, color),
        );
        offset += dash + gap;
    }
}

fn draw_polygon_fill(painter: &egui::Painter, points: &[egui::Pos2], color: egui::Color32) {
    if points.len() < 3 {
        return;
    }
    // Ear-clipping triangulation supports concave simple polygons unlike
    // Shape::convex_polygon. Selection validation rejects tiny polygons.
    let signed_area = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let next = points[(index + 1) % points.len()];
            point.x * next.y - next.x * point.y
        })
        .sum::<f32>();
    if signed_area.abs() < 1.0e-4 {
        return;
    }
    let orientation = signed_area.signum();
    let mut vertices = (0..points.len()).collect::<Vec<_>>();
    let mut triangles = Vec::new();
    let cross = |a: egui::Pos2, b: egui::Pos2, c: egui::Pos2| {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    };
    let inside_triangle = |p: egui::Pos2, a: egui::Pos2, b: egui::Pos2, c: egui::Pos2| {
        let ab = cross(a, b, p);
        let bc = cross(b, c, p);
        let ca = cross(c, a, p);
        (ab >= 0.0 && bc >= 0.0 && ca >= 0.0) || (ab <= 0.0 && bc <= 0.0 && ca <= 0.0)
    };
    let mut attempts = 0;
    while vertices.len() > 3 && attempts < points.len() * points.len() {
        attempts += 1;
        let mut ear_found = false;
        for index in 0..vertices.len() {
            let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
            let current = vertices[index];
            let next = vertices[(index + 1) % vertices.len()];
            if cross(points[previous], points[current], points[next]) * orientation <= 1.0e-5 {
                continue;
            }
            if vertices.iter().copied().any(|candidate| {
                candidate != previous
                    && candidate != current
                    && candidate != next
                    && inside_triangle(
                        points[candidate],
                        points[previous],
                        points[current],
                        points[next],
                    )
            }) {
                continue;
            }
            triangles.push([previous, current, next]);
            vertices.remove(index);
            ear_found = true;
            break;
        }
        if !ear_found {
            return;
        }
    }
    if vertices.len() == 3 {
        triangles.push([vertices[0], vertices[1], vertices[2]]);
    }
    for triangle in triangles {
        painter.add(egui::Shape::convex_polygon(
            triangle.into_iter().map(|index| points[index]).collect(),
            color.gamma_multiply(0.25),
            egui::Stroke::NONE,
        ));
    }
}

pub fn color_image(spectrogram: &Spectrogram) -> egui::ColorImage {
    color_image_inner(spectrogram, None).expect("non-cancellable image conversion returns a result")
}

pub fn color_image_cancellable(
    spectrogram: &Spectrogram,
    cancelled: &AtomicBool,
) -> Option<egui::ColorImage> {
    color_image_inner(spectrogram, Some(cancelled))
}

fn color_image_inner(
    spectrogram: &Spectrogram,
    cancelled: Option<&AtomicBool>,
) -> Option<egui::ColorImage> {
    let max_width = 8192usize;
    let width = spectrogram.n_time.min(max_width).max(1);
    let height = spectrogram.n_freq.max(1);
    let range = (spectrogram.db_max - spectrogram.db_min).max(1.0e-6);
    let time_ranges: Vec<(usize, usize)> = (0..width)
        .map(|x| {
            let start = x * spectrogram.n_time / width;
            let end = ((x + 1) * spectrogram.n_time / width)
                .max(start + 1)
                .min(spectrogram.n_time);
            (start, end)
        })
        .collect();
    let mut pixels = vec![egui::Color32::BLACK; width * height];
    pixels
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            if cancelled.is_some_and(|token| token.load(Ordering::Relaxed)) {
                return;
            }
            let frequency = height
                .saturating_sub(1)
                .saturating_sub(y)
                .min(spectrogram.n_freq.saturating_sub(1));
            for (x, pixel) in row.iter_mut().enumerate() {
                let (start, end) = time_ranges[x];
                if start < end {
                    let row_start = frequency * spectrogram.n_time;
                    let slice = &spectrogram.matrix[row_start + start..row_start + end];
                    if slice.iter().all(|value| *value == f32::NEG_INFINITY) {
                        *pixel = egui::Color32::BLACK;
                    } else {
                        let value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        *pixel = viridis(((value - spectrogram.db_min) / range).clamp(0.0, 1.0));
                    }
                }
            }
        });
    if cancelled.is_some_and(|token| token.load(Ordering::Relaxed)) {
        None
    } else {
        Some(egui::ColorImage::new([width, height], pixels))
    }
}

/// Convert changed STFT columns into a texture patch, retaining the original
/// spectrogram's fixed dB range so pixel colors match unchanged columns.
#[allow(dead_code)]
pub fn color_image_patch(
    patch: &SpectrogramPatch,
    n_total_time: usize,
    db_min: f32,
    db_max: f32,
) -> Option<(usize, egui::ColorImage)> {
    if patch.n_time == 0 || patch.n_freq == 0 || n_total_time == 0 {
        return None;
    }
    let width = n_total_time.min(8192).max(1);
    let first_x = patch.first_time_frame * width / n_total_time;
    let last_x = ((patch.first_time_frame + patch.n_time) * width).div_ceil(n_total_time);
    let first_x = first_x.min(width);
    let last_x = last_x.min(width).max(first_x);
    if first_x == last_x {
        return None;
    }

    let height = patch.n_freq;
    let patch_width = last_x - first_x;
    let patch_first_time = first_x * n_total_time / width;
    let patch_last_time = (last_x * n_total_time).div_ceil(width).min(n_total_time);
    if patch_first_time < patch.first_time_frame
        || patch_last_time > patch.first_time_frame + patch.n_time
    {
        return None;
    }
    let range = (db_max - db_min).max(1.0e-6);
    let mut pixels = vec![egui::Color32::BLACK; patch_width * height];
    pixels
        .par_chunks_mut(patch_width)
        .enumerate()
        .for_each(|(y, row)| {
            let frequency = height - 1 - y;
            for (offset, pixel) in row.iter_mut().enumerate() {
                let x = first_x + offset;
                let source_start = x * n_total_time / width;
                let source_end = ((x + 1) * n_total_time / width)
                    .max(source_start + 1)
                    .min(n_total_time);
                let local_start = source_start.saturating_sub(patch.first_time_frame);
                let local_end = source_end
                    .saturating_sub(patch.first_time_frame)
                    .min(patch.n_time);
                let row_start = frequency * patch.n_time;
                let slice = &patch.matrix[row_start + local_start..row_start + local_end];
                if !slice.is_empty() {
                    if slice.iter().all(|value| *value == f32::NEG_INFINITY) {
                        *pixel = egui::Color32::BLACK;
                    } else {
                        let value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        *pixel = viridis(((value - db_min) / range).clamp(0.0, 1.0));
                    }
                }
            }
        });
    Some((
        first_x,
        egui::ColorImage::new([patch_width, height], pixels),
    ))
}

fn viridis(value: f32) -> egui::Color32 {
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
