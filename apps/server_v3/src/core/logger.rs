pub fn init_log() {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_max_level(tracing::Level::DEBUG)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .try_init();
}
