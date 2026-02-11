//! logging utilities

use llama_cpp_2::{LogOptions, send_logs_to_tracing};

// 初始化日志
pub fn init_logger() {
    let _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_max_level(tracing::Level::DEBUG)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .try_init();
}

/// 将日志发送到 tracing
pub fn logs_to_tracing(enabled: bool) {
    // llama.cpp 日志
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(enabled));
}
