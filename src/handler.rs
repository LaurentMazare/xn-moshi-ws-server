use crate::model::{AppState, AppStateB};
use crate::protocol::{AsrReply, AsrRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use std::sync::Arc;
use xn::streaming::{StreamMask, StreamTensor};
use xn::{BackendQ, Tensor};
use xn_moshi::asr::AsrWord;

pub async fn ws_handler(
    State(app): State<AppState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    async fn handle_socket(socket: WebSocket, app: AppState) {
        let result = match app {
            AppState::Cpu(s) => serve_q(socket, s).await,
            AppState::Q80(s) => serve_q(socket, s).await,
            AppState::Q81(s) => serve_q(socket, s).await,
            AppState::Q8k(s) => serve_q(socket, s).await,
            AppState::Q6k(s) => serve_q(socket, s).await,
            AppState::Q50(s) => serve_q(socket, s).await,
            AppState::Q51(s) => serve_q(socket, s).await,
            AppState::Q5k(s) => serve_q(socket, s).await,
            AppState::Q40(s) => serve_q(socket, s).await,
            AppState::Q41(s) => serve_q(socket, s).await,
            AppState::Q4k(s) => serve_q(socket, s).await,
            #[cfg(feature = "cuda")]
            AppState::Cuda(s) => serve_q(socket, s).await,
        };
        if let Err(e) = result {
            tracing::error!(error = %e, "ws session terminated");
        }
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn serve_q<Q>(socket: WebSocket, app: Arc<AppStateB<Q>>) -> Result<()>
where
    Q: BackendQ + Send + Sync + 'static,
    Q::T: Send + Sync + 'static,
    Q::B: Send + Sync + 'static,
{
    use futures_util::{SinkExt, StreamExt};
    let (mut tx, mut rx) = socket.split();

    // Enforce a single active session at a time.
    let session_guard = match Arc::clone(&app.session_lock).try_lock_owned() {
        Ok(g) => g,
        Err(_) => anyhow::bail!("rejecting connection: server already has an active session"),
    };

    let (reply_tx, mut reply_rx) = tokio::sync::mpsc::unbounded_channel::<AsrReply>();

    let forwarder = tokio::spawn(async move {
        while let Some(reply) = reply_rx.recv().await {
            let json = serde_json::to_string(&reply)?;
            if tx.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
        let _ = tx.close().await;
        Ok::<_, anyhow::Error>(())
    });

    let outcome = run_session(app, &mut rx, &reply_tx).await;
    drop(reply_tx);
    let _ = forwarder.await;
    drop(session_guard);
    tracing::info!("websocket session ended");
    outcome
}

enum SessionState {
    Awaiting,
    Ready { ctrl_tx: std::sync::mpsc::Sender<ModelCtrl>, pcm_buf: Vec<f32> },
}

enum ModelCtrl {
    Audio(Vec<f32>),
    Flush { flush_id: u64 },
    EndOfStream,
}

async fn run_session<Q>(
    app: Arc<AppStateB<Q>>,
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<AsrReply>,
) -> Result<()>
where
    Q: BackendQ + Send + Sync + 'static,
    Q::T: Send + Sync + 'static,
    Q::B: Send + Sync + 'static,
{
    use futures_util::StreamExt;
    let mut sess = SessionState::Awaiting;
    let mut model_thread: Option<std::thread::JoinHandle<Result<()>>> = None;

    let frame_size = app.frame_size as usize;
    let bytes_per_frame = frame_size * 2;

    'outer: while let Some(msg) = stream.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
        };
        let req: AsrRequest = match serde_json::from_str(text.as_str()) {
            Ok(r) => r,
            Err(e) => {
                send_error(reply_tx, error_codes::BAD_REQUEST, format!("invalid request: {e}"))?;
                continue;
            }
        };
        match (&mut sess, req) {
            (
                SessionState::Awaiting,
                AsrRequest::Setup { model_name, input_format, close_ws_on_eos: _, .. },
            ) => {
                let sample_rate = app.sample_rate();
                if !is_supported_input_format(&input_format) {
                    send_error(
                        reply_tx,
                        error_codes::NOT_IMPLEMENTED,
                        format!(
                            "unsupported input_format '{input_format}', expected raw s16le PCM at {sample_rate} Hz",
                        ),
                    )?;
                    continue;
                }
                let request_id = uuid::Uuid::new_v4().to_string();
                let model_name =
                    if model_name.is_empty() { app.model_name.clone() } else { model_name };
                tracing::info!(?model_name, ?input_format, "starting new ASR session");
                let ready = AsrReply::Ready {
                    model_name,
                    sample_rate,
                    frame_size: app.frame_size,
                    delay_in_frames: app.delay_in_frames,
                    text_stream_names: vec![],
                    request_id,
                };
                if reply_tx.send(ready).is_err() {
                    anyhow::bail!("reply channel closed before ready");
                }

                let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel::<ModelCtrl>();
                let model = app.model();
                let tokenizer = Arc::clone(&app.tokenizer);
                let reply_tx_clone = reply_tx.clone();
                let handle = std::thread::Builder::new()
                    .name("asr-worker".to_string())
                    .spawn(move || run_asr_loop(model, tokenizer, ctrl_rx, reply_tx_clone))?;
                model_thread = Some(handle);
                sess =
                    SessionState::Ready { ctrl_tx, pcm_buf: Vec::with_capacity(bytes_per_frame) };
            }
            (SessionState::Awaiting, _) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "expected setup as first message".into(),
                )?;
            }
            (SessionState::Ready { .. }, AsrRequest::Setup { .. }) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "session already initialized".into(),
                )?;
            }
            (SessionState::Ready { ctrl_tx, pcm_buf }, AsrRequest::Audio { audio }) => {
                use base64::Engine;
                let bytes = match base64::engine::general_purpose::STANDARD.decode(audio) {
                    Ok(b) => b,
                    Err(e) => {
                        send_error(
                            reply_tx,
                            error_codes::BAD_REQUEST,
                            format!("invalid base64 audio: {e}"),
                        )?;
                        continue;
                    }
                };
                decode_pcm_s16le_into(&bytes, pcm_buf);
                while pcm_buf.len() >= frame_size {
                    let frame: Vec<f32> = pcm_buf.drain(..frame_size).collect();
                    if ctrl_tx.send(ModelCtrl::Audio(frame)).is_err() {
                        tracing::error!("asr worker dropped channel");
                        break 'outer;
                    }
                }
            }
            (SessionState::Ready { ctrl_tx, .. }, AsrRequest::Flush { flush_id }) => {
                if ctrl_tx.send(ModelCtrl::Flush { flush_id }).is_err() {
                    tracing::error!("asr worker dropped channel");
                    break 'outer;
                }
            }
            (SessionState::Ready { ctrl_tx, .. }, AsrRequest::EndOfStream) => {
                let _ = ctrl_tx.send(ModelCtrl::EndOfStream);
                tracing::info!("websocket stream closed by client (end of stream)");
                break;
            }
        }
    }

    // Drop the control sender so the worker exits, then await it.
    if let SessionState::Ready { ctrl_tx, .. } = sess {
        drop(ctrl_tx);
    }
    if let Some(handle) = model_thread.take() {
        match tokio::task::spawn_blocking(move || handle.join()).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => tracing::error!(error = %e, "asr worker error"),
            _ => tracing::error!("asr worker panicked"),
        }
    }

    tracing::info!("websocket stream closed by client");
    Ok(())
}

fn run_asr_loop<Q: BackendQ>(
    asr: Arc<xn_moshi::asr::Asr<Q>>,
    tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    ctrl_rx: std::sync::mpsc::Receiver<ModelCtrl>,
    reply_tx: tokio::sync::mpsc::UnboundedSender<AsrReply>,
) -> Result<()> {
    const FRAME_SIZE: usize = 1920;
    const FRAME_DURATION_S: f32 = FRAME_SIZE as f32 / 24_000.0;

    let device = asr.device().clone();
    let mut state = asr.init_state(1)?;
    let mask = StreamMask::all_active(1);
    let silence = vec![0.0f32; FRAME_SIZE];

    let mut stream_id: u32 = 0;
    let mut step_idx: u64 = 0;
    let mut total_duration_s: f32 = 0.0;
    // Accumulate text tokens with the separator (3) re-inserted so SentencePiece can detokenize correctly.
    let mut text_tokens: Vec<u32> = vec![];
    let mut last_decoded_len: usize = 0;
    let mut unended_word = false;

    while let Ok(msg) = ctrl_rx.recv() {
        match msg {
            ModelCtrl::Audio(frame) => {
                step_one_frame(
                    &device,
                    &mut state,
                    &mask,
                    &frame,
                    &tokenizer,
                    &reply_tx,
                    stream_id,
                    &mut step_idx,
                    &mut total_duration_s,
                    FRAME_DURATION_S,
                    &mut text_tokens,
                    &mut last_decoded_len,
                    &mut unended_word,
                )?;
            }
            ModelCtrl::Flush { flush_id } => {
                let _ = reply_tx.send(AsrReply::Flushed { flush_id });
                stream_id = stream_id.saturating_add(1);
                text_tokens.clear();
                last_decoded_len = 0;
                unended_word = false;
            }
            ModelCtrl::EndOfStream => {
                let delay = asr.asr_delay_in_tokens();
                for _ in 0..delay {
                    step_one_frame(
                        &device,
                        &mut state,
                        &mask,
                        &silence,
                        &tokenizer,
                        &reply_tx,
                        stream_id,
                        &mut step_idx,
                        &mut total_duration_s,
                        FRAME_DURATION_S,
                        &mut text_tokens,
                        &mut last_decoded_len,
                        &mut unended_word,
                    )?;
                }
                if unended_word {
                    let _ = reply_tx
                        .send(AsrReply::EndText { stop_s: total_duration_s, stream_id })
                        .is_ok();
                }
                let _ = reply_tx.send(AsrReply::EndOfStream).is_ok();
                break;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn step_one_frame<Q: BackendQ>(
    device: &Q::B,
    state: &mut xn_moshi::asr::AsrState<Q>,
    mask: &StreamMask,
    frame: &[f32],
    tokenizer: &sentencepiece::SentencePieceProcessor,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<AsrReply>,
    stream_id: u32,
    step_idx: &mut u64,
    total_duration_s: &mut f32,
    frame_duration_s: f32,
    text_tokens: &mut Vec<u32>,
    last_decoded_len: &mut usize,
    unended_word: &mut bool,
) -> Result<()> {
    let audio: Tensor<f32, Q::B> = Tensor::from_vec(frame.to_vec(), (1, 1, frame.len()), device)?;
    let pcm = StreamTensor::from_tensor(audio);
    let step_results = state.step_pcm(&pcm, mask, |_, _, _| {})?;
    for sr in step_results {
        *step_idx += 1;
        *total_duration_s += frame_duration_s;
        let _ = reply_tx.send(AsrReply::Step {
            step_idx: *step_idx,
            step_duration_s: frame_duration_s,
            total_duration_s: *total_duration_s,
            vad: vec![],
        });
        for word in sr.words {
            match word {
                AsrWord::Word { tokens, batch_idx: _, start_time } => {
                    text_tokens.push(3); // separator/space
                    text_tokens.extend_from_slice(&tokens);
                    let decoded = tokenizer.decode_piece_ids(text_tokens).unwrap_or_default();
                    if decoded.len() > *last_decoded_len {
                        let new_text = decoded[*last_decoded_len..].to_string();
                        *last_decoded_len = decoded.len();
                        let _ = reply_tx.send(AsrReply::Text {
                            text: new_text,
                            start_s: start_time as f32,
                            stream_id,
                        });
                    }
                    *unended_word = true;
                }
                AsrWord::EndWord { stop_time, batch_idx: _ } => {
                    *unended_word = false;
                    let _ =
                        reply_tx.send(AsrReply::EndText { stop_s: stop_time as f32, stream_id });
                }
            }
        }
    }
    Ok(())
}

fn is_supported_input_format(format: &str) -> bool {
    matches!(format.to_lowercase().as_str(), "" | "pcm" | "pcm_s16le")
}

fn decode_pcm_s16le_into(bytes: &[u8], out: &mut Vec<f32>) {
    let scale = 1.0 / i16::MAX as f32;
    out.reserve(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(s as f32 * scale);
    }
}

fn send_error(
    tx: &tokio::sync::mpsc::UnboundedSender<AsrReply>,
    code: u32,
    message: String,
) -> Result<()> {
    tx.send(AsrReply::Error { message, code })
        .map_err(|_| anyhow::anyhow!("reply channel closed"))?;
    Ok(())
}
