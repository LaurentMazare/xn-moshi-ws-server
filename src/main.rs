mod handler;
mod protocol;

use anyhow::Result;
use axum::Router;
use axum::routing::any;
use clap::Parser;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "xn-moshi-ws-server")]
#[command(about = "WebSocket server for Moshi STT")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,

    /// Quantization for the LM transformer linear weights.
    /// One of: q8|q8_0, q8_1, q8k, q6k, q5|q5_0, q5_1, q5k, q4|q4_0, q4_1, q4k.
    /// CPU only.
    #[arg(long)]
    quant: Option<String>,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    let app_state = build_app_state(&args)?;

    let app = Router::new()
        .route("/speech/asr", any(handler::ws_handler))
        .with_state(app_state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, "listening on /speech/asr");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    Ok(())
}

fn build_app_state(_args: &Args) -> Result<handler::AppState> {
    // Model loading not wired up yet — emit the protocol-level shape the
    // Moshi STT model will report once integrated (24kHz mimi, 12.5Hz frames).
    Ok(handler::AppState::new("kyutai/stt-2.6b-en".to_string(), 24000, 1920, 0))
}
