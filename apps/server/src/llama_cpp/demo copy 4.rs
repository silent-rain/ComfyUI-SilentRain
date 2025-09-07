//! Based on the mtmd cli example from llama.cpp.

use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;

use clap::Parser;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText,
};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

fn run_single_turn(
    ctx: &mut MtmdCliContext,
    model: &LlamaModel,
    context: &mut LlamaContext,
    sampler: &mut LlamaSampler,
    params: &MtmdCliParams,
) -> Result<(), Box<dyn std::error::Error>> {
    // Add media marker if not present
    let mut prompt = params.prompt.clone();
    let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
    let media_marker = params.media_marker.as_ref().unwrap_or(&default_marker);
    if !prompt.contains(media_marker) {
        prompt.push_str(media_marker);
    }

    // Load media files
    for image_path in &params.images {
        println!("Loading image: {image_path}");
        ctx.load_media(image_path)?;
    }
    for audio_path in &params.audio {
        ctx.load_media(audio_path)?;
    }

    // Create user message
    let msg = LlamaChatMessage::new("user".to_string(), prompt)?;

    println!("Evaluating message: {msg:?}");

    // Evaluate the message (prefill)
    ctx.eval_message(model, context, msg, true, params.batch_size)?;

    // Generate response (decode)
    ctx.generate_response(model, context, sampler, params.n_predict)?;

    Ok(())
}
