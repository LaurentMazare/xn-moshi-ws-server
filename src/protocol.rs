#[allow(dead_code)]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ErrorMsg {
    Error { message: String },
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrRequest {
    Setup {
        #[serde(default)]
        json_config: String,
        #[serde(default)]
        model_name: String,
        input_format: String,
        #[serde(default)]
        close_ws_on_eos: bool,
    },
    Audio {
        audio: String,
    },
    Flush {
        flush_id: u64,
    },
    EndOfStream,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VadPrediction {
    pub horizon_s: f32,
    pub inactivity_prob: f32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrReply {
    Ready {
        model_name: String,
        sample_rate: u32,
        frame_size: u32,
        delay_in_frames: u32,
        text_stream_names: Vec<String>,
        request_id: String,
    },
    Text {
        text: String,
        start_s: f32,
        stream_id: u32,
    },
    EndText {
        stop_s: f32,
        stream_id: u32,
    },
    Step {
        step_idx: u64,
        step_duration_s: f32,
        total_duration_s: f32,
        vad: Vec<VadPrediction>,
    },
    Flushed {
        flush_id: u64,
    },
    Error {
        message: String,
        code: u32,
    },
    EndOfStream,
}

#[allow(dead_code)]
pub mod error_codes {
    pub const BAD_REQUEST: u32 = 400;
    pub const INTERNAL: u32 = 500;
    pub const NOT_IMPLEMENTED: u32 = 501;
}
