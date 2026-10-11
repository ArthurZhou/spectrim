use eframe::egui;

#[derive(Default)]
pub struct TransportEvents {
    pub play: bool,
    pub pause: bool,
    pub stop: bool,
}

pub fn show(
    ui: &mut egui::Ui,
    is_playing: bool,
    current_time: f32,
    duration: f32,
    processed_mode: &mut bool,
    enabled: bool,
) -> TransportEvents {
    let mut events = TransportEvents::default();
    ui.add_enabled_ui(enabled, |ui| {
        if ui
            .button(if is_playing { "Pause" } else { "Play" })
            .clicked()
        {
            if is_playing {
                events.pause = true;
            } else {
                events.play = true;
            }
        }
        if ui.button("Stop").clicked() {
            events.stop = true;
        }
    });
    ui.checkbox(processed_mode, "Edited playback");
    ui.label(format!(
        "{:02}:{:02} / {:02}:{:02}",
        (current_time / 60.0) as u32,
        current_time as u32 % 60,
        (duration / 60.0) as u32,
        duration as u32 % 60
    ));
    events
}
