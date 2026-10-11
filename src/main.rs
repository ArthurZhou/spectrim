mod app;
pub mod audio;
mod playback;
mod ui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Spectrim — Audio Spectrum Editor",
        options,
        Box::new(|_creation_context| Ok(Box::new(app::SpectrimApp::default()))),
    )
}
