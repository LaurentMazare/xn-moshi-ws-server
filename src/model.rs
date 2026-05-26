use anyhow::{Context as _, Result};
use std::sync::Arc;
use xn::BackendQ;
use xn::nn::VB;
use xn_moshi::asr::Asr;
use xn_moshi::lm::{self, LmModel};
use xn_moshi::mimi::{self, Mimi};

pub const ASR_DELAY_S: f64 = 2.5;

pub struct AppStateB<Q: BackendQ> {
    model: Arc<Asr<Q>>,
    pub tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    pub model_name: String,
    sample_rate: u32,
    pub frame_size: u32,
    pub delay_in_frames: u32,
    pub vad_horizons: Vec<f32>,
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

struct ModelPaths<Q: BackendQ> {
    model: Asr<Q>,
    tokenizer: std::path::PathBuf,
    model_name: String,
    sample_rate: u32,
    frame_size: u32,
    asr_delay_in_tokens: usize,
    vad_horizons: Vec<f32>,
}

impl<Q: BackendQ> ModelPaths<Q> {
    const REPO_ID: &str = "kyutai/stt-2.6b-en-candle";
    const LM_FILE: &str = "model.safetensors";
    const MIMI_FILE: &str = "mimi-pytorch-e351c8d8@125.safetensors";
    const TOKENIZER_FILE: &str = "tokenizer_en_audio_4000.model";

    fn stt_2b(temperature: f64, dev: &Q::B) -> Result<Self> {
        use hf_hub::{Repo, RepoType, api::sync::Api};
        tracing::info!(repo_id = %Self::REPO_ID, "downloading model");
        let api = Api::new()?;
        let repo = api.repo(Repo::new(Self::REPO_ID.to_string(), RepoType::Model));
        let lm = repo.get(Self::LM_FILE).map_err(anyhow::Error::from)?;
        let mimi = repo.get(Self::MIMI_FILE).map_err(anyhow::Error::from)?;
        let tokenizer = repo.get(Self::TOKENIZER_FILE).map_err(anyhow::Error::from)?;
        tracing::info!(?lm, ?mimi, ?tokenizer, "model weights ready");
        let mimi_vb = VB::load(&[mimi], dev.clone())?.root();
        let mimi_config = mimi::Config::v0_1(Some(32));
        let sample_rate = mimi_config.sample_rate as u32;
        let frame_size = (mimi_config.sample_rate / mimi_config.frame_rate) as u32;
        let asr_delay_in_tokens = (ASR_DELAY_S * sample_rate as f64 / frame_size as f64) as usize;
        let mimi: Mimi<f32, Q::B> = Mimi::load(&mimi_vb, mimi_config)?;
        mimi_vb.check_all_used_with_ignore(|s| {
            s.ends_with("_codebook._initialized")
                || s.ends_with("_codebook.cluster_usage")
                || s.ends_with("_codebook.embedding_sum")
        })?;

        let lm_vb = VB::load(&[lm], dev.clone())?.root();
        let lm_config = lm::Config::stt_2_6b();
        let lm: LmModel<Q> = LmModel::load(&lm_vb, &lm_config)?;
        lm_vb.check_all_used()?;

        let model = Asr::new(asr_delay_in_tokens, temperature, mimi, lm);

        Ok(Self {
            model,
            tokenizer,
            model_name: Self::REPO_ID.to_string(),
            sample_rate,
            frame_size,
            asr_delay_in_tokens,
            vad_horizons: vec![],
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct AsrConfig {
    asr_delay_in_tokens: Option<usize>,
    vad_horizons: Option<Vec<f32>>,
}

pub fn load_asr<Q: BackendQ>(
    model_path: Option<&str>,
    temperature: f64,
    dev: Q::B,
) -> Result<AppStateB<Q>> {
    let model_path = match model_path {
        None => ModelPaths::stt_2b(temperature, &dev)?,
        Some(path) if path.ends_with(".json") => {
            let sample_rate = 24000;
            let frame_size = 1920;
            let path = std::path::Path::new(path);
            let parent = path.parent().context("model config path has no parent directory")?;
            let asr_config = serde_json::from_str::<AsrConfig>(&std::fs::read_to_string(path)?)
                .context("failed to parse config.json")?;
            tracing::info!(?asr_config, "ASR config loaded");
            let asr_delay_in_tokens = asr_config
                .asr_delay_in_tokens
                .unwrap_or((ASR_DELAY_S * sample_rate as f64 / frame_size as f64) as usize);
            let model = xn_moshi::asr::Asr::load(
                parent.join("mimi.safetensors").to_str().context("invalid mimi path")?,
                parent.join("model.safetensors").to_str().context("invalid model path")?,
                Some(path.to_str().context("invalid config path")?),
                asr_delay_in_tokens,
                temperature,
                dev,
            )?;
            let model_name =
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("custom_model").to_string();
            ModelPaths {
                model,
                tokenizer: parent.join("tokenizer.model"),
                model_name,
                sample_rate,
                frame_size,
                asr_delay_in_tokens,
                vad_horizons: asr_config.vad_horizons.unwrap_or_default(),
            }
        }
        Some(repo_id) => {
            let sample_rate = 24000;
            let frame_size = 1920;
            use hf_hub::{Repo, RepoType, api::sync::Api};
            let api = Api::new()?;
            let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
            let config = repo.get("config.json").map_err(anyhow::Error::from)?;
            let lm = repo.get("model.safetensors").map_err(anyhow::Error::from)?;
            let mimi = repo.get("mimi.safetensors").map_err(anyhow::Error::from)?;
            let tokenizer = repo.get("tokenizer.model").map_err(anyhow::Error::from)?;
            tracing::info!(?lm, ?mimi, ?tokenizer, "model weights ready");
            let asr_config = serde_json::from_str::<AsrConfig>(&std::fs::read_to_string(&config)?)
                .context("failed to parse config.json")?;
            tracing::info!(?asr_config, "ASR config loaded");
            let asr_delay_in_tokens = asr_config
                .asr_delay_in_tokens
                .unwrap_or((ASR_DELAY_S * sample_rate as f64 / frame_size as f64) as usize);
            let model = xn_moshi::asr::Asr::load(
                mimi.to_str().context("invalid mimi path")?,
                lm.to_str().context("invalid model path")?,
                Some(config.to_str().context("invalid config path")?),
                asr_delay_in_tokens,
                temperature,
                dev,
            )?;
            let model_name = repo_id.to_string();
            ModelPaths {
                model,
                tokenizer,
                model_name,
                sample_rate,
                frame_size,
                asr_delay_in_tokens,
                vad_horizons: asr_config.vad_horizons.unwrap_or_default(),
            }
        }
    };

    let tokenizer_path_str = model_path.tokenizer.to_str().context("invalid tokenizer path")?;
    let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path_str)
        .with_context(|| format!("failed to open tokenizer at {tokenizer_path_str}"))?;

    Ok(AppStateB {
        model: model_path.model.into(),
        tokenizer: Arc::new(sp),
        model_name: model_path.model_name,
        sample_rate: model_path.sample_rate,
        frame_size: model_path.frame_size,
        delay_in_frames: model_path.asr_delay_in_tokens as u32,
        session_lock: Arc::new(tokio::sync::Mutex::new(())),
        vad_horizons: model_path.vad_horizons,
    })
}
