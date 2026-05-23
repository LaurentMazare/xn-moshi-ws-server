use anyhow::{Context as _, Result};
use std::sync::Arc;
use xn::BackendQ;
use xn::nn::VB;
use xn_moshi::asr::Asr;
use xn_moshi::lm::{self, LmModel};
use xn_moshi::mimi::{self, Mimi};

pub const REPO_ID: &str = "kyutai/stt-2.6b-en-candle";
pub const LM_FILE: &str = "model.safetensors";
pub const MIMI_FILE: &str = "mimi-pytorch-e351c8d8@125.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer_en_audio_4000.model";

pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_SIZE: u32 = 1_920;
pub const ASR_DELAY_S: f64 = 2.5;

pub struct AppStateB<Q: BackendQ> {
    model: Arc<Asr<Q>>,
    pub tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    pub model_name: String,
    sample_rate: u32,
    pub frame_size: u32,
    pub delay_in_frames: u32,
    /// Held for the lifetime of the active session to enforce one session at a time.
    pub session_lock: Arc<tokio::sync::Mutex<()>>,
}

impl<Q: BackendQ> AppStateB<Q> {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn model(&self) -> Arc<Asr<Q>> {
        self.model.clone()
    }
}

#[derive(Clone)]
pub enum AppState {
    Cpu(Arc<AppStateB<xn::Unquantized<f32, xn::CpuDevice>>>),
    Q80(Arc<AppStateB<xn::quantized::Q80F32>>),
    Q81(Arc<AppStateB<xn::quantized::Q81F32>>),
    Q8k(Arc<AppStateB<xn::quantized::Q8kF32>>),
    Q6k(Arc<AppStateB<xn::quantized::Q6kF32>>),
    Q50(Arc<AppStateB<xn::quantized::Q50F32>>),
    Q51(Arc<AppStateB<xn::quantized::Q51F32>>),
    Q5k(Arc<AppStateB<xn::quantized::Q5kF32>>),
    Q40(Arc<AppStateB<xn::quantized::Q40F32>>),
    Q41(Arc<AppStateB<xn::quantized::Q41F32>>),
    Q4k(Arc<AppStateB<xn::quantized::Q4kF32>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<AppStateB<xn::Unquantized<half::bf16, xn::CudaDevice>>>),
}

pub fn load_asr<Q: BackendQ>(temperature: f64, dev: Q::B) -> Result<AppStateB<Q>> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    tracing::info!(repo_id = %REPO_ID, "downloading model");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(REPO_ID.to_string(), RepoType::Model));
    let lm_path = repo.get(LM_FILE).map_err(anyhow::Error::from)?;
    let mimi_path = repo.get(MIMI_FILE).map_err(anyhow::Error::from)?;
    let tokenizer_path = repo.get(TOKENIZER_FILE).map_err(anyhow::Error::from)?;
    tracing::info!(?lm_path, ?mimi_path, ?tokenizer_path, "model weights ready");

    let tokenizer_path_str = tokenizer_path.to_str().context("invalid tokenizer path")?;
    let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path_str)
        .with_context(|| format!("failed to open tokenizer at {tokenizer_path_str}"))?;

    let mimi_vb = VB::load(&[mimi_path], dev.clone())?.root();
    let mimi_config = mimi::Config::v0_1(Some(32));
    let mimi: Mimi<f32, Q::B> = Mimi::load(&mimi_vb, mimi_config)?;
    mimi_vb.check_all_used_with_ignore(|s| {
        s.ends_with("_codebook._initialized")
            || s.ends_with("_codebook.cluster_usage")
            || s.ends_with("_codebook.embedding_sum")
    })?;

    let lm_vb = VB::load(&[lm_path], dev)?.root();
    let lm_config = lm::Config::stt_2_6b();
    let lm: LmModel<Q> = LmModel::load(&lm_vb, &lm_config)?;
    lm_vb.check_all_used()?;

    let asr_delay_in_tokens = (ASR_DELAY_S * SAMPLE_RATE as f64 / FRAME_SIZE as f64) as usize;
    let model = Asr::new(asr_delay_in_tokens, temperature, mimi, lm);

    Ok(AppStateB {
        model: Arc::new(model),
        tokenizer: Arc::new(sp),
        model_name: REPO_ID.to_string(),
        sample_rate: SAMPLE_RATE,
        frame_size: FRAME_SIZE,
        delay_in_frames: asr_delay_in_tokens as u32,
        session_lock: Arc::new(tokio::sync::Mutex::new(())),
    })
}
