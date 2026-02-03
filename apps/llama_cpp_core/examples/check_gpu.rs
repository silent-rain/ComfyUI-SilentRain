use llama_cpp_core::{Backend, Model, utils::log::init_logger};

fn main() -> anyhow::Result<()> {
    init_logger();

    // 初始化后端
    let _backend = Backend::init_backend()?;

    // 检测设备是否可用，如果没有 GPU 则使用 CPU
    let model = Model::new("", None::<String>);
    let devices = model.devices();

    println!("devices list {devices:?}");
    println!("Detected {} GPU device(s)", devices.len());
    Ok(())
}
