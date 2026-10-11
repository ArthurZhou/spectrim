mod app;
pub mod audio;
mod playback;
mod ui;

fn main() -> eframe::Result<()> {
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if let Err(error) = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build_global()
    {
        eprintln!("Could not configure Rayon worker pool: {error}");
    }
    eprintln!("Audio analysis worker threads: {worker_count}");
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Spectrim — Audio Spectrum Editor",
        options,
        Box::new(|_creation_context| Ok(Box::new(app::SpectrimApp::default()))),
    )
}
