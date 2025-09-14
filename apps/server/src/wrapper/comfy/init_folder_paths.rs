//! 初始化文件路径

use crate::wrapper::comfy::folder_paths::FolderPaths;

/// 初始化文件路径
pub fn apply_custom_paths() {
    let mut folder_paths = FolderPaths::default();
    folder_paths.add_model_folder_path(
        "checkpoints",
        folder_paths.model_path().join("checkpoints"),
        false,
    );

    folder_paths.add_model_folder_path(
        "checkpoints",
        folder_paths.model_path().join("checkpoints"),
        false,
    );

    folder_paths.add_model_folder_path("clip", folder_paths.model_path().join("clip"), false);
    folder_paths.add_model_folder_path("vae", folder_paths.model_path().join("vae"), false);
    folder_paths.add_model_folder_path(
        "diffusion_models",
        folder_paths.model_path().join("diffusion_models"),
        false,
    );
    folder_paths.add_model_folder_path("loras", folder_paths.model_path().join("loras"), false);

    folder_paths.add_model_folder_path(
        "text_encoders",
        folder_paths.model_path().join("text_encoders"),
        false,
    );

    folder_paths.add_model_folder_path("LLM", folder_paths.model_path().join("LLM"), false);
}
