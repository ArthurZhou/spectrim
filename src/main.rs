mod app;
pub mod audio;
mod playback;
mod ui;

fn configure_cjk_fonts(context: &eframe::CreationContext<'_>) {
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ];
    let Some(bytes) = candidates.iter().find_map(|path| std::fs::read(path).ok()) else {
        eprintln!("No CJK font found; Chinese glyphs may use the platform fallback.");
        return;
    };

    let mut fonts = eframe::egui::FontDefinitions::default();
    fonts.font_data.insert(
        "cjk_fallback".to_owned(),
        eframe::egui::FontData::from_owned(bytes).into(),
    );
    for family in [
        eframe::egui::FontFamily::Proportional,
        eframe::egui::FontFamily::Monospace,
    ] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "cjk_fallback".to_owned());
    }
    context.egui_ctx.set_fonts(fonts);
}

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
        Box::new(|creation_context| {
            configure_cjk_fonts(creation_context);
            Ok(Box::new(app::SpectrimApp::default()))
        }),
    )
}
