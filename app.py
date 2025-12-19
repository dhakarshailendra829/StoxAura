import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
import time
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="QuickTrade AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #020617 100%);
    color: #e5e7eb;
}
.main .block-container { padding-top: 2rem; max-width: 1400px; }

.brand-title { font-size: 2.4rem; font-weight: 700; letter-spacing: 0.08em; }
.brand-sub   { font-size: 0.95rem; color: #9ca3af; }

.card {
    background: rgba(15,23,42,0.96);
    border-radius: 18px;
    padding: 1.5rem 1.5rem 1.25rem 1.5rem;
    box-shadow: 0 24px 60px rgba(0,0,0,0.45);
    border: 1px solid rgba(148,163,184,0.25);
}
.card-soft {
    background: rgba(15,23,42,0.9);
    border-radius: 18px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 18px 45px rgba(0,0,0,0.4);
    border: 1px solid rgba(148,163,184,0.15);
}

.section-title { font-size: 1.05rem; font-weight: 600; color: #e5e7eb; }

.stButton>button {
    width: 100%;
    border-radius: 999px;
    padding: 0.65rem 1.25rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(135deg, #2563eb, #22c55e);
    color: white;
    box-shadow: 0 12px 30px rgba(37,99,235,0.55);
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 40px rgba(34,197,94,0.75);
}

.signal-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.signal-buy  { background: rgba(34,197,94,0.14);  color: #4ade80; border: 1px solid rgba(34,197,94,0.65); }
.signal-sell { background: rgba(239,68,68,0.14);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.6); }
.signal-hold { background: rgba(251,191,36,0.14); color: #fbbf24; border: 1px solid rgba(251,191,36,0.55); }

.metric-label   { font-size: 0.9rem; color: #9ca3af; }
.metric-value-lg  { font-size: 1.5rem; font-weight: 700; color: #e5e7eb; }
.metric-value-xl  { font-size: 1.9rem; font-weight: 700; color: #f9fafb; }

.dataframe td, .dataframe th { color: #e5e7eb !important; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load("production_xgb.pkl")
        scaler_X = joblib.load("scaler_features.pkl")
        scaler_y = joblib.load("scaler_target.pkl")
        return xgb_model, scaler_X, scaler_y
    except Exception as e:
        st.error(f" Could not load model/scaler files: {e}")
        return None, None, None

model, scaler_X, scaler_y = load_models()

def fetch_history(ticker: str, days: int = 365):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    data = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if data is None or data.empty:
        return None
    return data.dropna()

def compute_rsi(close_series: pd.Series, window: int = 14):
    delta = close_series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_feature_row(hist_df: pd.DataFrame):
    if len(hist_df) < 200:
        return None

    df = hist_df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["MA7"] = close.rolling(7).mean()
    df["MA21"] = close.rolling(21).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["Daily_Return"] = close.pct_change()
    df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
    df["Volatility_21"] = df["Daily_Return"].rolling(21).std()
    df["RSI"] = compute_rsi(close, 14)
    df["Price_Momentum"] = close.pct_change(5)
    df["Volume_SMA"] = volume.rolling(20).mean()

    row = df.iloc[-1]

    features = np.array([
        row["Open"],
        row["High"],
        row["Low"],
        row["Close"],
        row["Volume"],
        row["MA7"],
        row["MA21"],
        row["MA50"],
        row["MA200"],
        row["Volatility_7"],
        row["Volatility_21"],
        row["RSI"],
        row["Price_Momentum"],
        row["Volume_SMA"],
        row["Daily_Return"],
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def predict_price(features: np.ndarray):
    if model is None or scaler_X is None or scaler_y is None:
        return None
    if features is None:
        return None
    X_scaled = scaler_X.transform(features.reshape(1, -1))
    y_scaled_pred = model.predict(X_scaled)
    pred_price = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0, 0]
    return float(pred_price)

def get_signal(current_price: float, predicted_price: float, hist_df: pd.DataFrame):
    if predicted_price is None or hist_df is None or hist_df.empty:
        return "HOLD", 0.0

    change_pct = (predicted_price - current_price) / current_price * 100

    daily_ret = hist_df["Close"].pct_change()
    vol_14 = daily_ret.rolling(14).std().iloc[-1]
    if vol_14 is None or np.isnan(vol_14) or vol_14 == 0:
        vol_14 = 0.01

    raw_conf = abs(change_pct) / (vol_14 * 100)
    conf_pct = float(np.clip(raw_conf * 100, 5, 98))

    if change_pct > 1.5:
        action = "BUY"
    elif change_pct < -1.5:
        action = "SELL"
    else:
        action = "HOLD"

    return action, conf_pct

def fake_sentiment_label(change_pct: float):
    if change_pct > 4:
        return "STRONG POSITIVE"
    if change_pct > 1:
        return "POSITIVE"
    if change_pct < -4:
        return "STRONG NEGATIVE"
    if change_pct < -1:
        return "NEGATIVE"
    return "NEUTRAL"

top_left, top_right = st.columns([2, 1])

with top_left:
    st.markdown("<div class='brand-title'>QUICKTRADE AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='brand-sub'>Powered by Yahoo Finance Data · AI-driven trade signals in real time</div>",
        unsafe_allow_html=True,
    )

with top_right:
    st.write("")
    st.write("")
    st.caption("Model status:")
    if model is None:
        st.error("Model: not loaded")
    else:
        st.success("Model: loaded and ready")

st.write("")

left_col, mid_col, right_col = st.columns([1.3, 2.1, 1.4])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Select Stock</div>", unsafe_allow_html=True)

    ticker = st.selectbox(
        "",
        ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "JPM", "NFLX", "AMD", "IBM"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    horizon = st.slider(
        "Prediction Horizon (Days)",
        min_value=5,
        max_value=60,
        value=30,
        step=1,
    )

    include_sentiment = st.checkbox("Include Sentiment Analysis (demo)", value=True)
    live_mode = st.checkbox("Live Mode (auto refresh)", value=False)
    refresh_every = st.slider("Refresh seconds", 5, 30, 10) if live_mode else None

    st.markdown("")
    run_prediction = st.button("RUN PREDICTION")

    st.markdown("</div>", unsafe_allow_html=True)

with mid_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Price Prediction + Data</div>",
        unsafe_allow_html=True,
    )

    hist = fetch_history(ticker, days=365)
    if hist is None or hist.empty:
        st.warning(f"No data for {ticker}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        current_price = float(hist["Close"].iloc[-1])

        predicted_price = None
        pred_series = None

        if run_prediction and model is not None:
            feat = prepare_feature_row(hist)
            predicted_price = predict_price(feat)

            if predicted_price is not None:
                pred_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
                projected = np.linspace(current_price, predicted_price, num=horizon)
                pred_series = pd.Series(projected, index=pred_dates)

        hist_reset = hist.reset_index()
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)

        fig.add_trace(
            go.Candlestick(
                x=hist_reset["Date"],
                open=hist_reset["Open"],
                high=hist_reset["High"],
                low=hist_reset["Low"],
                close=hist_reset["Close"],
                name="OHLC",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
                showlegend=False,
            ),
            row=1, col=1
        )

        hist_reset["MA20"] = hist_reset["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=hist_reset["Date"],
                y=hist_reset["MA20"],
                mode="lines",
                name="20‑Day MA",
                line=dict(color="#eab308", width=1.6),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>MA20: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=hist_reset["Date"],
                y=hist_reset["Volume"],
                name="Volume",
                marker_color="#4b5563",
            ),
            row=2, col=1
        )

        if pred_series is not None:
            pred_reset = pred_series.reset_index()
            pred_reset.columns = ["Date", "Predicted"]

            fig.add_vrect(
                x0=hist_reset["Date"].iloc[-1],
                x1=pred_reset["Date"].iloc[-1],
                fillcolor="rgba(248, 250, 252, 0.05)",
                line_width=0,
                layer="below",
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=pred_reset["Date"],
                    y=pred_reset["Predicted"],
                    mode="lines+markers",
                    name=f"Prediction (+{horizon}d)",
                    line=dict(color="#fb923c", width=2.4, dash="dash"),
                    marker=dict(size=5),
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>Pred: $%{y:.2f}<extra></extra>",
                ),
                row=1, col=1
            )

            if len(pred_reset) > 3:
                buy_idx = pred_reset["Predicted"].idxmin()
                sell_idx = pred_reset["Predicted"].idxmax()

                fig.add_trace(
                    go.Scatter(
                        x=[pred_reset.loc[buy_idx, "Date"]],
                        y=[pred_reset.loc[buy_idx, "Predicted"]],
                        mode="markers+text",
                        text=["BUY"],
                        textposition="top center",
                        marker=dict(color="#22c55e", size=14, symbol="triangle-up"),
                        name="Buy Zone",
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=[pred_reset.loc[sell_idx, "Date"]],
                        y=[pred_reset.loc[sell_idx, "Predicted"]],
                        mode="markers+text",
                        text=["SELL"],
                        textposition="bottom center",
                        marker=dict(color="#ef4444", size=14, symbol="triangle-down"),
                        name="Sell Zone",
                    ),
                    row=1, col=1
                )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig.update_xaxes(
            row=1, col=1,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
            tickformat="%b %d\n%Y",
        )
        fig.update_yaxes(
            row=1, col=1,
            title_text="Price (USD)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
        )
        fig.update_xaxes(
            row=2, col=1,
            showgrid=False,
            tickformat="%b %d\n%Y",
        )
        fig.update_yaxes(
            row=2, col=1,
            title_text="Volume",
            showgrid=True,
            gridcolor="rgba(30,64,175,0.35)",
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Current Price</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-value-xl'>${current_price:,.2f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-label'>Predicted Price ({horizon} days)</div>",
                unsafe_allow_html=True,
            )
            if predicted_price is not None:
                st.markdown(
                    f"<div class='metric-value-xl' style='color:#fb923c;'>${predicted_price:,.2f}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='metric-value-xl' style='opacity:0.4;'>Click RUN PREDICTION</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Trading Signals</div>", unsafe_allow_html=True)

    action_text = "HOLD"
    conf_pct = 0.0
    if "current_price" in locals() and "hist" in locals() and hist is not None and predicted_price is not None:
        action_text, conf_pct = get_signal(current_price, predicted_price, hist)

    if action_text == "BUY":
        signal_class = "signal-buy"
    elif action_text == "SELL":
        signal_class = "signal-sell"
    else:
        signal_class = "signal-hold"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='signal-badge {signal_class}'>ACTION: {action_text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<div class='metric-label'>Confidence</div>", unsafe_allow_html=True)
    if predicted_price is not None:
        st.markdown(
            f"<div class='metric-value-lg'>{conf_pct:.1f}%</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(int(conf_pct), 100))
    else:
        st.markdown(
            "<div class='metric-value-lg' style='opacity:0.5;'>Run prediction to see confidence</div>",
            unsafe_allow_html=True,
        )
        st.progress(0)

    if include_sentiment and predicted_price is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Sentiment</div>", unsafe_allow_html=True)
        label = fake_sentiment_label((predicted_price - current_price) / current_price * 100)
        st.markdown(
            f"<div class='metric-value-lg'>{label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Latest Transactions (Demo)</div>", unsafe_allow_html=True)

    if "current_price" in locals():
        demo_trades = pd.DataFrame(
            [
                [datetime.utcnow().date(), "BUY", round(current_price * 0.99, 2), 120],
                [datetime.utcnow().date() - timedelta(days=1), "SELL", round(current_price * 1.03, 2), 80],
                [datetime.utcnow().date() - timedelta(days=2), "SELL", round(current_price * 1.01, 2), 50],
            ],
            columns=["Date", "Type", "Price", "Quantity"],
        )
        st.dataframe(demo_trades, hide_index=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
if live_mode:
    time.sleep(refresh_every)
    st.rerun()
