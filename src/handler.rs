use crate::protocol::{AsrReply, AsrRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use std::sync::Arc;

pub struct AppStateInner {
    pub model_name: String,
    pub sample_rate: u32,
    pub frame_size: u32,
    pub delay_in_frames: u32,
}

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<AppStateInner>,
}

impl AppState {
    pub fn new(
        model_name: String,
        sample_rate: u32,
        frame_size: u32,
        delay_in_frames: u32,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner { model_name, sample_rate, frame_size, delay_in_frames }),
        }
    }
}

pub async fn ws_handler(
    State(app): State<AppState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    async fn handle_socket(socket: WebSocket, app: AppState) {
        if let Err(e) = serve(socket, app).await {
            tracing::error!(error = %e, "ws session terminated");
        }
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn serve(socket: WebSocket, app: AppState) -> Result<()> {
    use futures_util::{SinkExt, StreamExt};
    let (mut tx, mut rx) = socket.split();
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
    tracing::info!("websocket session ended");
    outcome
}

enum SessionState {
    Awaiting,
    Ready { close_ws_on_eos: bool },
}

async fn run_session(
    app: AppState,
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<AsrReply>,
) -> Result<()> {
    use futures_util::StreamExt;
    let mut sess = SessionState::Awaiting;

    while let Some(msg) = stream.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => return Ok(()),
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
                AsrRequest::Setup { model_name, input_format, close_ws_on_eos, .. },
            ) => match handle_setup(&app, model_name, input_format, close_ws_on_eos, reply_tx)? {
                Some(new_state) => sess = new_state,
                None => continue,
            },
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
            (SessionState::Ready { .. }, AsrRequest::Audio { audio }) => {
                use base64::Engine;
                let _bytes = match base64::engine::general_purpose::STANDARD.decode(audio) {
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
                // TODO: feed audio bytes through the ASR model and emit Text/Step/EndText replies.
            }
            (SessionState::Ready { .. }, AsrRequest::Flush { flush_id }) => {
                // TODO: drain pending audio through the model before acknowledging.
                let _ = reply_tx.send(AsrReply::Flushed { flush_id });
            }
            (SessionState::Ready { close_ws_on_eos }, AsrRequest::EndOfStream) => {
                // TODO: flush remaining audio and emit final transcription before closing.
                let _ = reply_tx.send(AsrReply::EndOfStream);
                tracing::info!("websocket stream closed by client (end of stream)");
                if *close_ws_on_eos {
                    return Ok(());
                }
            }
        }
    }
    tracing::info!("websocket stream closed by client");
    Ok(())
}

fn handle_setup(
    app: &AppState,
    model_name: String,
    input_format: String,
    close_ws_on_eos: bool,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<AsrReply>,
) -> Result<Option<SessionState>> {
    if !input_format.is_empty() && !is_supported_input_format(&input_format) {
        send_error(
            reply_tx,
            error_codes::NOT_IMPLEMENTED,
            format!("unsupported input_format '{input_format}'"),
        )?;
        return Ok(None);
    }
    let request_id = uuid::Uuid::new_v4().to_string();
    let model_name = if model_name.is_empty() { app.inner.model_name.clone() } else { model_name };
    tracing::info!(?model_name, ?input_format, "starting new ASR session");
    let ready = AsrReply::Ready {
        model_name,
        sample_rate: app.inner.sample_rate,
        frame_size: app.inner.frame_size,
        delay_in_frames: app.inner.delay_in_frames,
        text_stream_names: vec![],
        request_id,
    };
    if reply_tx.send(ready).is_err() {
        anyhow::bail!("reply channel closed before ready");
    }
    Ok(Some(SessionState::Ready { close_ws_on_eos }))
}

fn is_supported_input_format(format: &str) -> bool {
    matches!(format.to_lowercase().as_str(), "pcm" | "pcm_s16_le" | "opus")
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
