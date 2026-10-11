开发指导书：Rust + egui 音频频谱编辑器
本文件用途：交给编码 agent 作为实现规范。请按"阶段任务"顺序逐阶段实现，每阶段结束必须能编译通过并满足该阶段的验收标准，再进入下一阶段。

0. 项目目标
实现一个桌面应用：导入音频文件 → 显示频谱瀑布图 → 用户可在图上框选一个或多个矩形区域 → 对选中区域执行静音或独奏（solo）→ 试听效果 → 导出处理后的音频。

最终交付物：一个可 cargo run 启动的原生 GUI 应用，能完成上述完整流程。

1. 技术约束
语言：Rust 2021 edition

GUI：eframe / egui（即时模式）

解码：symphonia

播放：rodio

FFT：rustfft

导出：hound

文件对话框：rfd

并行（可选）：rayon

禁止：不得引入 Web 前端、不得调用外部 FFmpeg 二进制（后续可添加调用ffmpeg的自动转换，但该应用自己至少支持一两种常用文件形式）、不得使用 unsafe（除非有充分理由并注释说明）

版本策略：使用各 crate 的最新稳定版本。若某个 API 与本文档描述不符，以 cargo doc / 编译错误为准，自行适配并在代码注释中记录差异。不要盲目照抄本文档的 API 签名。

2. 项目骨架
text
audio-editor/
├── Cargo.toml
└── src/
    ├── main.rs                 # 入口，eframe 启动
    ├── app.rs                  # App 结构体，eframe::App 实现，UI 总装
    ├── audio/
    │   ├── mod.rs
    │   ├── types.rs            # AudioData, Spectrogram, Selection 等数据类型
    │   ├── decode.rs           # 文件 → AudioData
    │   ├── spectrogram.rs      # AudioData → Spectrogram
    │   ├── process.rs          # 时域/频域掩码处理
    │   └── export.rs           # PCM → WAV
    ├── playback.rs             # rodio 音频线程 + 命令通道
    └── ui/
        ├── mod.rs
        ├── spectrogram_view.rs # 瀑布图渲染 + 框选交互
        ├── transport.rs        # 播放/暂停/进度条/模式开关
        └── region_list.rs      # 选区列表 + 属性编辑
3. 核心数据结构（src/audio/types.rs）
这些结构是模块间契约，必须精确定义：

rust
/// 解码后的音频数据，PCM 交错存储
pub struct AudioData {
    pub samples: Vec<f32>,       // 交错，范围 [-1.0, 1.0]
    pub sample_rate: u32,        // Hz
    pub channels: u16,           // 1 或 2
    pub duration: f32,           // 秒
}

impl AudioData {
    /// 总帧数（每帧包含 channels 个采样）
    pub fn frame_count(&self) -> usize { self.samples.len() / self.channels as usize }

    /// 取单声道混合（用于频谱分析）
    pub fn to_mono(&self) -> Vec<f32>;

    /// 时间（秒）→ 帧索引
    pub fn time_to_frame(&self, t: f32) -> usize;
}

/// STFT 结果。matrix 为行优先 [n_freq * n_time]，值为 dB
pub struct Spectrogram {
    pub matrix: Vec<f32>,
    pub n_freq: usize,           // n_fft/2 + 1
    pub n_time: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: u32,
    pub db_min: f32,             // 用于归一化显示的 5% 分位
    pub db_max: f32,             // 95% 分位
}

impl Spectrogram {
    pub fn at(&self, freq_bin: usize, time_frame: usize) -> f32;
    /// 频率 bin → Hz
    pub fn bin_to_hz(&self, bin: usize) -> f32;
    /// Hz → 最近的频率 bin
    pub fn hz_to_bin(&self, hz: f32) -> usize;
    /// 时间帧 → 秒
    pub fn frame_to_time(&self, frame: usize) -> f32;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SelectionAction { Mute, Solo }

#[derive(Clone, Debug)]
pub struct Selection {
    pub id: u64,                 // 唯一标识，用于 UI 列表操作
    pub time_start: f32,         // 秒
    pub time_end: f32,           // 秒
    pub freq_low: f32,           // Hz
    pub freq_high: f32,          // Hz
    pub action: SelectionAction,
    /// 是否只作用于时间维度（忽略频率范围）。
    /// 若用户框选跨越整个频率轴，可设为 true 以走快速路径。
    pub full_band: bool,
}
约定：

所有时间单位为秒，频率单位为 Hz。

Selection 保证 time_start <= time_end、freq_low <= freq_high（在构造时归一化）。

4. 模块接口契约
4.1 audio/decode.rs
rust
/// 从文件路径解码音频。成功返回 AudioData。
/// 内部将音频重采样到 44100 Hz、统一为 1 或 2 声道。
pub fn decode_file(path: &Path) -> Result<AudioData, DecodeError>;

pub enum DecodeError {
    Io(std::io::Error),
    UnsupportedFormat(String),
    DecodeFailure(String),
}
要点：

使用 symphonia::default::get_probe() 探测格式。

遍历所有音轨，选择第一个音频轨。

解码循环：format.next_packet() → decoder.decode() → 写入 Vec<f32>。

不做重采样（保持原始采样率），但若原始采样率非 44100，在 decode_file 内部用简单的线性插值重采样到 44100。这是简化决定，避免引入 rubato。

声道数 > 2 时降为前 2 声道；单声道保持单声道。

4.2 audio/spectrogram.rs
rust
/// 计算 STFT 频谱。n_fft 建议 2048，hop 建议 512。
pub fn compute(audio: &AudioData, n_fft: usize, hop: usize) -> Spectrogram;

/// 计算 dB 范围（5%~95% 分位）
fn compute_db_range(matrix: &[f32]) -> (f32, f32);
算法要求：

用 audio.to_mono() 得到分析信号。

Hann 窗，长度 n_fft。

n_time = (mono.len().saturating_sub(n_fft)) / hop + 1。

对每帧：加窗 → rustfft → 取前 n_fft/2+1 个 bin 的幅度 → 转 dB：20 * log10(mag + 1e-9)。

若 mono.len() < n_fft，返回全零矩阵（不要 panic）。

性能要求：对 3 分钟音频（约 15000 帧）应在 3 秒内完成。若单线程不达标，用 rayon 按帧并行。并行时注意 FftPlanner 的 Arc<dyn Fft> 可跨线程共享。

4.3 audio/process.rs
rust
/// 对音频应用所有选区，返回处理后的交错 PCM。
/// 内部策略：
///   - 若所有选区都是 full_band（只涉及时间维度），走快速时域路径。
///   - 否则走 STFT 掩码路径（精确到频率维度）。
pub fn apply_selections(audio: &AudioData, sels: &[Selection]) -> Vec<f32>;

/// 时域快速路径：逐帧判断是否静音
pub fn apply_time_domain(audio: &AudioData, sels: &[Selection]) -> Vec<f32>;

/// 频域精确路径：STFT → 掩码 → ISTFT
pub fn apply_spectral_masks(
    audio: &AudioData,
    sels: &[Selection],
    n_fft: usize,
    hop: usize,
) -> Vec<f32>;
掩码逻辑（必须严格实现）：

设 has_solo = sels.iter().any(|s| s.action == Solo)。

对每个采样点（时域路径）或每个时频单元（频域路径）：

若 has_solo：保留当且仅当它落在至少一个 Solo 选区内；否则置零。

若 !has_solo：置零当且仅当它落在至少一个 Mute 选区内。

频域路径的时频单元判定：

时间帧 t 对应秒数 frame_to_time(t) 落在 [time_start, time_end]。

频率 bin f 对应 Hz bin_to_hz(f) 落在 [freq_low, freq_high]。

ISTFT 要求：

每个声道独立处理。

overlap-add 后用窗平方和归一化（每个样本位置累加 window[i]^2，最后相除），避免音量起伏。

若某位置窗平方和为 0，该处输出 0。

处理后的样本长度应与输入等长（截断或补零）。

4.4 audio/export.rs
rust
/// 导出为 32-bit float WAV
pub fn export_wav(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), hound::Error>;
4.5 playback.rs
设计要求：rodio 的 OutputStream 必须在专用线程中创建，主线程通过 channel 发送命令。

rust
pub enum AudioCmd {
    Play { samples: Vec<f32>, sample_rate: u32, channels: u16 },
    Pause,
    Resume,
    Stop,
}

pub struct PlaybackHandle {
    tx: Sender<AudioCmd>,
    /// 播放位置（毫秒），由音频线程定期更新
    position_ms: Arc<AtomicU32>,
}

impl PlaybackHandle {
    /// 启动音频线程，返回句柄
    pub fn spawn() -> Self;
    pub fn play(&self, samples: Vec<f32>, sample_rate: u32, channels: u16);
    pub fn pause(&self);
    pub fn resume(&self);
    pub fn stop(&self);
    pub fn position(&self) -> f32;  // 秒
}
要点：

音频线程收到 Play 时，停掉旧 Sink，新建 Sink，append(SamplesBuffer)，play()。

播放位置：音频线程在收到 Play 时记录起点，之后每 50ms 通过 AtomicU32 更新已播放毫秒数。用 sink.get_pos() 或自行累计（Instant::now() - start - paused_duration）均可。

注意：rodio 各版本 API 有差异，若 Sink::get_pos 不存在，改用自行累计方案。

主线程不得阻塞等待音频线程。

5. 分阶段任务
阶段 1：骨架 + 文件加载 + 频谱计算
任务：

建立项目骨架和 Cargo.toml。

实现 types.rs 全部数据结构。

实现 decode_file。

实现 compute（先单线程）。

main.rs 启动一个空窗口，顶部有一个"打开文件"按钮，点击后用 rfd 选文件，加载成功后在窗口中央用 egui::Label 显示 文件名 / 时长 / 采样率 / 声道数，并在控制台打印频谱矩阵尺寸。

验收标准：

cargo run 能启动窗口。

能加载一个 WAV 文件并显示基本信息。

控制台输出的 n_freq × n_time 与预期一致（例如 44.1kHz、n_fft=2048、hop=512、30 秒音频 → 1025 × 2583）。

阶段 2：频谱纹理渲染
任务：

实现 Spectrogram → egui::ColorImage 转换（Viridis 色图，可自行实现或用简单近似）。

在 App 中缓存 egui::TextureHandle。

在 CentralPanel 中用一个占满可用空间的矩形绘制纹理。

Y 轴翻转（低频在底部）。

关键约束（重要）：

纹理尺寸限制：GPU 纹理最大边长通常为 8192 或 16384。若 n_time > 8192，必须在生成纹理前做时间方向降采样（例如按整数倍聚合取平均或取最大），使纹理宽度 ≤ 8192。降采样只影响显示，不影响 Spectrogram.matrix 本身（后续 DSP 仍用原始矩阵）。

降采样倍率记录在 App 中，用于后续坐标换算（但坐标换算仍基于时间/频率的物理量，不依赖像素，所以一般无需特殊处理）。

使用 ColorImage 时注意 egui 版本的 API 差异（ColorImage::new 参数形式在 0.30 前后有变化）。以编译为准。

验收标准：

加载音频后能看到清晰的频谱瀑布图。

低频在底部，高频在顶部。

窗口拉伸时图像随之缩放，不变形。

阶段 3：播放 + Playhead
任务：

实现 playback.rs。

transport.rs：播放/暂停/停止按钮 + 当前时间显示。

每帧从 PlaybackHandle 读取位置，在瀑布图上绘制一条竖直的 playhead 线。

播放时调用 ctx.request_repaint_after(Duration::from_millis(16)) 驱动重绘。

验收标准：

点播放能听到原始音频。

Playhead 随时间从左侧移动到右侧，位置与听感基本一致（误差 < 200ms 可接受）。

暂停/继续/停止都正常工作。

阶段 4：框选交互
任务：

在瀑布图区域用 ui.allocate_exact_size(size, Sense::click_and_drag()) 获取响应。

拖动时绘制实时预览矩形（半透明黄色）。

drag_stopped 时生成 Selection 并加入 App.selections。

已存在的选区用不同颜色描边 + 半透明填充。Mute 用深灰填充，Solo 用选区自身颜色填充。

右键点击落在某个选区内 → 删除该选区。

顶栏添加一个开关："新框选默认动作：Mute / Solo"。

验收标准：

能画出多个矩形选区，颜色各异。

选区在窗口拉伸时位置正确（跟随时间/频率坐标，而非固定像素）。

能删除选区。

阶段 5：时间域静音 + 重新播放
任务：

实现 apply_time_domain。

Transport 栏添加模式切换："原始 / 编辑后"。

切到"编辑后"并按下播放时：调用 apply_selections（此时只用了时域路径），把结果喂给 PlaybackHandle::play。

选区变化时，若当前处于"编辑后播放"状态，停止播放（或重新计算后自动重播——先实现"停止"即可，简单可靠）。

验收标准：

框选一段时间区域设为 Mute，切换到"编辑后"播放，该时间段确实无声。

框选一段设为 Solo，播放时只有该段有声（其余静音）。

撤销选区（删除）后重新播放，行为正确。

阶段 6：频域精确处理
任务：

实现 apply_spectral_masks，包含完整的 STFT → 掩码 → ISTFT。

apply_selections 中：若存在非 full_band 的选区，走频域路径；否则走时域路径。

频域处理可能耗时（3 分钟音频约 2-5 秒），必须放到后台线程，处理期间 UI 显示"处理中…"并禁用播放按钮。

处理完成后通过 channel 回传 PCM，主线程接收后更新播放 buffer。

验收标准：

只框选频谱图中间某频段（例如 1kHz~3kHz 的一个窄条）设为 Mute，导出或播放时该频段能量显著下降（用耳朵判断：声音变闷）。

时间维度边界处没有明显的咔哒声（说明 ISTFT 归一化正确）。

处理期间 UI 不卡死。

关键实现提示：

ISTFT 归一化必须按窗平方和逐样本累加后相除。

处理前后建议做一次淡入淡出（前后各 5ms）以消除可能的边界爆音。

若时间紧，可以先用 apply_time_domain 处理时间维、再用频域处理频域维，但不要简单串联（会重复处理）。正确做法是单一 STFT 掩码流程。

阶段 7：导出 + 收尾
任务：

实现 export_wav。

顶栏添加"导出 WAV"按钮，用 rfd 选保存路径。

导出内容为"编辑后"的 PCM（若当前无选区则导出原始）。

导出也走后台线程 + 进度提示。

收尾项：

选区列表面板（region_list.rs）：显示所有选区，允许用 DragValue 微调 time_start/end 和 freq_low/high，允许切换 action，允许删除。

频率轴对数/线性切换（可选项）。

键盘快捷键：空格播放/暂停，Delete 删除选中选区。

验收标准：

导出的 WAV 能用系统播放器正常播放。

用 Audacity 等工具查看导出文件的频谱图，静音区域的能量确实被移除。

6. UI 布局规范
text
┌──────────────────────────────────────────────────────────┐
│ TopPanel: [打开] [播放/暂停] [停止] 时间 00:12/03:45      │
│           [模式: 原始|编辑后]  [默认框选: Mute|Solo] [导出]│
├────────────────────────────────────┬─────────────────────┤
│                                    │ SidePanel           │
│   CentralPanel                     │ 选区列表：          │
│   ┌────────────────────────────┐   │  #1 Mute 0:12-0:15  │
│   │                            │   │      1k-3kHz  [×]   │
│   │     频谱瀑布图   横向延展    │   │  #2 Solo 0:30-0:40  │
│   │     + 选区覆盖              │   │      full band [×]  │
│   │     + playhead              │   │                     │
│   └────────────────────────────┘   │ [添加选区]          │
│                                    │                     │
├────────────────────────────────────┴─────────────────────┤
│ BottomPanel: 状态栏（文件信息 / 处理进度 / 错误消息）      │
└──────────────────────────────────────────────────────────┘
7. 边界情况与错误处理
情况	要求行为
加载失败（格式不支持）	状态栏显示错误，不 panic，保持原有状态
音频过短（< n_fft）	频谱返回全零矩阵，UI 正常显示空白
选区宽度为 0	拒绝创建（要求像素宽 > 4 且高 > 4）
多个选区重叠	Mute 取并集；有 Solo 时忽略所有 Mute
处理中的重复触发	用 Arc<AtomicBool> 标志位拒绝，或取消前一个任务
导出路径无权限	状态栏报错，不崩溃
音频文件极大（> 30 分钟）	允许加载，但频谱计算前弹出确认；或直接拒绝并提示
8. 禁止事项
禁止在 egui 的 update 函数中执行超过 ~16ms 的同步操作（解码、FFT、ISTFT、文件 IO）。必须走后台线程 + channel。

禁止在主线程创建 rodio::OutputStream。

禁止把原始 Spectrogram.matrix 直接当作纹理上传（尺寸可能超限）。必须经过降采样。

禁止在 ISTFT 中省略窗平方和归一化。

禁止用 unwrap() 处理用户输入相关的 Result（文件路径、格式等）。内部不变量可以用 expect 并写明原因。

禁止为了"看起来完整"而实现未在本文档要求的功能（如多轨、均衡器、效果器）。保持范围收敛。

9. 交付前自检清单
□ cargo build --release 无警告（或警告均有注释说明）
□ cargo clippy 无 error 级问题
□ 用 WAV / MP3 / FLAC 各测一次导入
□ 3 分钟音频的频谱计算 < 5 秒
□ 框选后切到"编辑后"播放，行为符合预期
□ 导出文件能被 Audacity 正常打开且频谱正确
□ 所有后台任务有进度/错误反馈，UI 不卡死
□ 代码中所有 TODO 都已清理或转为 issue 注释
10. 建议的实现顺序（给 agent 的元指令）
请严格按阶段 1 → 7 顺序推进。每完成一个阶段：

运行 cargo build 确认编译通过。

逐条对照该阶段"验收标准"自查。

用一句话总结本阶段实际完成情况，再进入下一阶段。

如果某个 API 与文档描述不符（例如 egui 的 ColorImage 构造、rodio 的 Sink 接口），以实际编译结果为准调整，并在代码注释中写明"此版本 API 与文档差异"。不要为了迁就文档而引入不必要的抽象层或降级依赖版本。

如果某阶段耗时超出预期，优先保证功能正确（能跑通验收标准），再考虑性能优化。