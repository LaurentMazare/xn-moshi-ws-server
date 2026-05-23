use crate::model::{AppState, AppStateB};
use crate::protocol::{AsrReply, AsrRequest};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use std::sync::Arc;
use xn::streaming::{StreamMask, StreamTensor};
use xn::{BackendQ, Tensor};

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

enum InMsg {
    Audio(Vec<f32>),
    EndOfStream,
}

async fn serve_q<Q: BackendQ>(socket: WebSocket, app: Arc<AppStateB<Q>>) -> Result<()> {
    use futures_util::{SinkExt, StreamExt};
    let (mut tx, mut rx) = socket.split();

    let first_message = loop {
        match rx.next().await {
            Some(msg) => match msg? {
                Message::Text(t) => break t,
                Message::Close(_) => {
                    anyhow::bail!("websocket stream closed by client during setup")
                }
                Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
            },
            None => {
                tracing::info!("websocket stream closed by client");
                return Ok(());
            }
        }
    };
    let sample_rate = app.sample_rate();
    let frame_size = app.frame_size;
    let (reply_tx, mut reply_rx) = tokio::sync::mpsc::unbounded_channel::<AsrReply>();
    let (in_tx, in_rx) = std::sync::mpsc::channel::<InMsg>();
    let mut decoder = match serde_json::from_str::<AsrRequest>(&first_message)? {
        AsrRequest::Setup { input_format, .. } => {
            use std::str::FromStr;
            let request_id = uuid::Uuid::new_v4().to_string();
            let ready = AsrReply::Ready {
                model_name: app.model_name.clone(),
                sample_rate,
                frame_size,
                delay_in_frames: app.delay_in_frames,
                text_stream_names: vec![],
                request_id,
            };
            if reply_tx.send(ready).is_err() {
                anyhow::bail!("reply channel closed before ready");
            }

            let format = crate::decoder::Format::from_str(&input_format)?;
            crate::decoder::Decoder::new(format, sample_rate as usize, frame_size as usize)?
        }
        _ => {
            anyhow::bail!("expected setup message as first message, got something else")
        }
    };

    // Enforce a single active session at a time.
    let session_guard = match Arc::clone(&app.session_lock).try_lock_owned() {
        Ok(g) => g,
        Err(_) => anyhow::bail!("rejecting connection: server already has an active session"),
    };

    let send_loop = async move {
        while let Some(reply) = reply_rx.recv().await {
            let json = serde_json::to_string(&reply)?;
            if tx.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
        let _ = tx.close().await;
        Ok::<_, anyhow::Error>(())
    };

    let recv_loop = async move {
        while let Some(msg) = rx.next().await {
            let msg = match msg? {
                Message::Text(msg) => serde_json::from_str::<AsrRequest>(&msg)?,
                Message::Close(_) => {
                    tracing::info!("websocket stream closed by client");
                    break;
                }
                Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => {
                    continue;
                }
            };
            match msg {
                AsrRequest::Setup { .. } => {
                    anyhow::bail!("received unexpected setup message after initial setup");
                }
                AsrRequest::Audio { audio } => {
                    use base64::Engine;
                    let audio = base64::engine::general_purpose::STANDARD.decode(audio)?;
                    let pcm = decoder.decode(&audio)?;
                    in_tx.send(InMsg::Audio(pcm))?;
                }
                AsrRequest::Flush { .. } => {}
                AsrRequest::EndOfStream => {
                    // Do not shutdown things yet in case there is audio to be processed in the
                    // queue.
                    in_tx.send(InMsg::EndOfStream)?;
                }
            }
        }
        Ok::<_, anyhow::Error>(())
    };

    tokio::select! {
        res = run_session(app, in_rx, reply_tx) => {
            match res {
                Ok(()) => tracing::info!("session loop ended"),
                Err(err) => tracing::error!(?err, "session loop error"),
            }
        }
        res = send_loop => {
            match res {
                Ok(()) => tracing::info!("send loop ended"),
                Err(err) => tracing::error!(?err, "send loop error"),
            }
        }
        res = recv_loop => {
            match res {
                Ok(()) => tracing::info!("recv loop ended"),
                Err(err) => tracing::error!(?err, "recv loop error"),
            }
        }
    };
    drop(session_guard);
    tracing::info!("websocket session ended");
    Ok(())
}

async fn run_session<Q: BackendQ>(
    app: Arc<AppStateB<Q>>,
    in_rx: std::sync::mpsc::Receiver<InMsg>,
    reply_tx: tokio::sync::mpsc::UnboundedSender<AsrReply>,
) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        use xn_moshi::asr::AsrWord;

        let mut state = app.model().init_state(1)?;
        let dev = app.model().device().clone();
        let tokenizer = &app.tokenizer;
        let mask = StreamMask::all_active(1);
        loop {
            let input = in_rx.recv()?;
            match input {
                InMsg::Audio(pcm) => {
                    let pcm = Tensor::from_vec(pcm, (1, 1, ()), &dev)?;
                    let pcm = StreamTensor::from_tensor(pcm);
                    for sr in state.step_pcm(&pcm, &mask, |_, _, _| {})? {
                        for word in sr.words {
                            let msg = match word {
                                AsrWord::Word { tokens, batch_idx: _, start_time } => {
                                    let decoded = tokenizer.decode_piece_ids(&tokens)?;
                                    AsrReply::Text {
                                        text: decoded,
                                        start_s: start_time as f32,
                                        stream_id: 0,
                                    }
                                }
                                AsrWord::EndWord { stop_time, batch_idx: _ } => {
                                    AsrReply::EndText { stop_s: stop_time as f32, stream_id: 0 }
                                }
                            };
                            reply_tx.send(msg)?;
                        }
                    }
                }
                InMsg::EndOfStream => {
                    // TODO(laurent): drain any remaining audio in the model and send end of stream
                    // reply.
                    break;
                }
            }
        }
        Ok::<(), anyhow::Error>(())
    })
    .await??;
    Ok(())
}
