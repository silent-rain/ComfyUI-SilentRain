//! 采样器

use llama_cpp_2::sampling::LlamaSampler;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// 采样器配置
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Controls diversity via top-k sampling.
    /// Higher values mean more diverse outputs.
    #[serde(default)]
    pub top_k: i32,

    /// Controls diversity via nucleus sampling.
    /// Lower values mean more focused outputs.
    #[serde(default)]
    pub top_p: f32,

    /// Controls randomness.
    /// 0.0 means deterministic, 1.0 means fully random.
    #[serde(default)]
    pub temperature: f32,

    /// Min-p 采样阈值
    pub min_p: f32,

    /// Seed for random number generation.
    /// Set to a fixed value for reproducible outputs.
    /// 0 表示随机
    #[serde(default)]
    pub seed: u32,

    /// Size of the sliding window for repeat penalty
    /// Specifies how many most recent tokens to consider for repeat penalty
    #[serde(default)]
    pub penalty_last_n: i32,

    /// Repeat penalty coefficient
    /// Penalizes repeated tokens - higher values enforce more diversity
    #[serde(default)]
    pub penalty_repeat: f32,

    /// Frequency penalty coefficient
    /// Penalizes tokens based on their frequency in the text
    #[serde(default)]
    pub penalty_freq: f32,

    /// Presence penalty coefficient
    /// Penalizes tokens already present in the context
    #[serde(default)]
    pub penalty_present: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            seed: 0,
            penalty_last_n: 64,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
        }
    }
}

impl SamplerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_penalty_last_n(mut self, penalty_last_n: i32) -> Self {
        self.penalty_last_n = penalty_last_n;
        self
    }

    pub fn with_penalty_repeat(mut self, penalty_repeat: f32) -> Self {
        self.penalty_repeat = penalty_repeat;
        self
    }

    pub fn with_penalty_freq(mut self, penalty_freq: f32) -> Self {
        self.penalty_freq = penalty_freq;
        self
    }

    pub fn with_penalty_present(mut self, penalty_present: f32) -> Self {
        self.penalty_present = penalty_present;
        self
    }

    pub fn seed(&self) -> u32 {
        // 随机值
        if self.seed == 0 {
            rand::rng().next_u32()
        } else {
            self.seed
        }
    }
}

pub struct Sampler {}

impl Sampler {
    /// load sampler
    pub fn load_sampler(params: &SamplerConfig) -> Result<LlamaSampler, Error> {
        /* penalties 配置说明:
            减少重复性: penalties(64, 1.2, 0.0, 0.2)
            增加多样性: penalties(64, 1.1, 0.1, 0.0)
            默认平衡: penalties(64, 1.0, 0.0, 0.0)
        */

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k),
            LlamaSampler::top_p(params.top_p, 0),
            LlamaSampler::min_p(params.min_p, 0),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::penalties(
                params.penalty_last_n,
                params.penalty_repeat,
                params.penalty_freq,
                params.penalty_present,
            ),
            // LlamaSampler::greedy(),   // 贪婪采样器，始终选择概率最高的 token, 应用于最后一个
            LlamaSampler::dist(params.seed()), // 随机种子，用于生成随机数, 应用于最后一个
        ]);
        Ok(sampler)
    }
}
