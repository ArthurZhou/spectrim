use eframe::egui;

use crate::audio::types::{Selection, SelectionAction, SelectionShape};

#[derive(Default)]
pub struct RegionEvents {
    pub changed: bool,
    pub deleted: Vec<u64>,
    pub selected: Option<Option<u64>>,
}

pub fn show(
    ui: &mut egui::Ui,
    selections: &mut Vec<Selection>,
    selected_id: Option<u64>,
    nyquist: f32,
) -> RegionEvents {
    let mut events = RegionEvents::default();
    ui.heading("Regions");
    let mut remove = Vec::new();
    for selection in selections.iter_mut() {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                events.changed |= ui.checkbox(&mut selection.enabled, "Enabled").changed();
                if ui
                    .selectable_label(
                        selected_id == Some(selection.id),
                        format!("#{}", selection.id),
                    )
                    .clicked()
                {
                    events.selected = Some(if selected_id == Some(selection.id) {
                        None
                    } else {
                        Some(selection.id)
                    });
                }
                let previous = selection.action;
                if selection.is_polygon() {
                    ui.label("Polygon");
                } else {
                    ui.label("Rectangle");
                }
                egui::ComboBox::from_id_salt(("action", selection.id))
                    .selected_text(match selection.action {
                        SelectionAction::Solo => "Solo",
                        SelectionAction::Gain => "Gain",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut selection.action, SelectionAction::Solo, "Solo");
                        ui.selectable_value(&mut selection.action, SelectionAction::Gain, "Gain");
                    });
                if previous != selection.action {
                    events.changed = true;
                }
                if ui.small_button("×").clicked() {
                    remove.push(selection.id);
                }
            });
            if selection.is_polygon() {
                ui.label(format!(
                    "Time: {:.3}–{:.3}s · edit vertices on plot",
                    selection.time_start, selection.time_end
                ));
            } else {
                ui.horizontal(|ui| {
                    ui.label("Time");
                    events.changed |= ui
                        .add(
                            egui::DragValue::new(&mut selection.time_start)
                                .range(0.0..=f32::MAX)
                                .speed(0.01),
                        )
                        .changed();
                    ui.label("to");
                    events.changed |= ui
                        .add(
                            egui::DragValue::new(&mut selection.time_end)
                                .range(0.0..=f32::MAX)
                                .speed(0.01),
                        )
                        .changed();
                });
            }
            if selection.action == SelectionAction::Gain {
                ui.horizontal(|ui| {
                    ui.label("Gain");
                    events.changed |= ui
                        .add(egui::Slider::new(&mut selection.gain, 0.0..=4.0).suffix("×"))
                        .changed();
                });
            }
            if matches!(selection.shape, SelectionShape::Polygon { .. }) {
                ui.label("Vertices: polygon region");
            } else if selection.full_band {
                ui.label(format!("Frequency: 0 – {nyquist:.0} Hz (full band)"));
            } else {
                ui.horizontal(|ui| {
                    ui.label("Hz");
                    events.changed |= ui
                        .add(
                            egui::DragValue::new(&mut selection.freq_low)
                                .range(0.0..=nyquist)
                                .speed(10.0),
                        )
                        .changed();
                    ui.label("to");
                    events.changed |= ui
                        .add(
                            egui::DragValue::new(&mut selection.freq_high)
                                .range(0.0..=nyquist)
                                .speed(10.0),
                        )
                        .changed();
                });
            }
            if !selection.is_polygon() {
                let was_full_band = selection.full_band;
                let full_band_changed =
                    ui.checkbox(&mut selection.full_band, "Full band").changed();
                events.changed |= full_band_changed;
                if selection.full_band {
                    events.changed |= selection.freq_low != 0.0 || selection.freq_high != nyquist;
                    selection.freq_low = 0.0;
                    selection.freq_high = nyquist;
                } else if was_full_band {
                    // Restore a useful narrow band when the user unchecks Full band.
                    selection.freq_low = nyquist * 0.2;
                    selection.freq_high = nyquist * 0.4;
                }
            }
        });
    }
    for id in remove {
        selections.retain(|selection| selection.id != id);
        events.deleted.push(id);
        events.changed = true;
    }
    for selection in selections.iter_mut() {
        if selection.time_start > selection.time_end {
            std::mem::swap(&mut selection.time_start, &mut selection.time_end);
            events.changed = true;
        }
        if selection.freq_low > selection.freq_high {
            std::mem::swap(&mut selection.freq_low, &mut selection.freq_high);
            events.changed = true;
        }
    }
    if selections.is_empty() {
        ui.label("No regions selected.");
    }
    events
}
