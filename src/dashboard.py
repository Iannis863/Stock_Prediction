import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="S&P 500 Intelligence Oracle",
    page_icon="📈",
    layout="wide",
)

PARQUET_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/integrated_data_parquet")
FEATURES = [
    "Price_vs_7D_MA",
    "Price_vs_30D_MA",
    "Volatility_20D",
    "Daily_Sentiment_Score",
    "Daily_Article_Count",
    "Prev_Return",
]

# ---------------------------------------------------------------------------
# Data & model loading (cached so they only run once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Target_Return"]  = df["Close"].pct_change()
    df["Price_vs_7D_MA"]  = df["Close"] / df["Close_7D_Avg"]  - 1
    df["Price_vs_30D_MA"] = df["Close"] / df["Close_30D_Avg"] - 1
    df["Volatility_20D"]  = df["Target_Return"].rolling(20).std()
    df["Prev_Return"]     = df["Target_Return"].shift(1)
    df = df.dropna(subset=FEATURES + ["Target_Return"])
    return df


@st.cache_data
def train_model(df: pd.DataFrame):
    split_idx = int(len(df) * 0.8)
    X_train = df[FEATURES].iloc[:split_idx]
    y_train = df["Target_Return"].iloc[:split_idx]
    X_test  = df[FEATURES].iloc[split_idx:]
    y_test  = df["Target_Return"].iloc[split_idx:]
    dates_test = df["Date"].iloc[split_idx:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    return preds, y_test.reset_index(drop=True), dates_test.reset_index(drop=True), importances


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Integrated data not found at `data/processed/integrated_data_parquet`. "
        "Run `data_ingestion.ipynb` → `data_processing.ipynb` first."
    )
    st.stop()

preds, y_test, dates_test, importances = train_model(df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 S&P 500\nIntelligence Oracle")
    st.markdown("---")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    default_start = max_date - pd.Timedelta(days=365 * 5)

    date_range = st.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        start_ts = pd.Timestamp(date_range[0])
        end_ts   = pd.Timestamp(date_range[1])
        view = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()
    else:
        view = df.copy()

    st.markdown("---")
    st.metric("Trading days (full dataset)", f"{len(df):,}")
    st.metric("Earliest date", str(df["Date"].min().date()))
    st.metric("Latest date",   str(df["Date"].max().date()))
    coverage = (df["Daily_Article_Count"] > 0).mean()
    st.metric("Sentiment coverage", f"{coverage:.1%}")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("S&P 500 Intelligence Oracle")
st.caption(
    "End-to-end Big Data pipeline: historical market data × FNSPID financial news sentiment × Random Forest."
)

tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🗞️ News Sentiment", "🤖 Model Performance"])

# ===========================================================================
# TAB 1 — Market Overview
# ===========================================================================
with tab1:
    st.subheader("Price History & Moving Averages")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=["Close Price (USD)", "Volume"],
    )

    fig.add_trace(
        go.Scatter(x=view["Date"], y=view["Close"], name="Close",
                   line=dict(color="#1f77b4", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=view["Date"], y=view["Close_7D_Avg"], name="7-Day MA",
                   line=dict(color="#ff7f0e", width=1.5, dash="dot")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=view["Date"], y=view["Close_30D_Avg"], name="30-Day MA",
                   line=dict(color="#2ca02c", width=1.5, dash="dash")),
        row=1, col=1,
    )

    if "Volume" in view.columns:
        fig.add_trace(
            go.Bar(x=view["Date"], y=view["Volume"], name="Volume",
                   marker_color="lightblue", opacity=0.6),
            row=2, col=1,
        )

    fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",   f"${view['Close'].iloc[-1]:,.2f}")
    c2.metric("Period High",    f"${view['High'].max():,.2f}"  if "High" in view.columns else "N/A")
    c3.metric("Period Low",     f"${view['Low'].min():,.2f}"   if "Low"  in view.columns else "N/A")
    period_return = (view["Close"].iloc[-1] / view["Close"].iloc[0] - 1) * 100
    c4.metric("Period Return",  f"{period_return:+.1f}%")

# ===========================================================================
# TAB 2 — News Sentiment
# ===========================================================================
with tab2:
    st.subheader("FNSPID Daily News Sentiment")

    sentiment_view = view[view["Daily_Article_Count"] > 0].copy()

    if sentiment_view.empty:
        st.warning("No FNSPID sentiment coverage in the selected date range — try widening the window.")
    else:
        bar_colors = sentiment_view["Daily_Sentiment_Score"].apply(
            lambda x: "#2ca02c" if x > 0 else ("#d62728" if x < 0 else "#aec7e8")
        )

        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.06,
            subplot_titles=["Daily Sentiment Score", "Daily Article Count"],
        )

        fig2.add_trace(
            go.Bar(x=sentiment_view["Date"], y=sentiment_view["Daily_Sentiment_Score"],
                   name="Sentiment", marker_color=bar_colors, opacity=0.8),
            row=1, col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=sentiment_view["Date"],
                y=sentiment_view["Daily_Sentiment_Score"].rolling(7, min_periods=1).mean(),
                name="7-Day Avg", line=dict(color="purple", width=2),
            ),
            row=1, col=1,
        )
        fig2.add_trace(
            go.Bar(x=sentiment_view["Date"], y=sentiment_view["Daily_Article_Count"],
                   name="Articles", marker_color="steelblue", opacity=0.7),
            row=2, col=1,
        )

        fig2.update_yaxes(title_text="Score [−1, 1]", row=1, col=1)
        fig2.update_yaxes(title_text="Articles",      row=2, col=1)
        fig2.update_layout(height=460, hovermode="x unified",
                           legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg sentiment score",  f"{sentiment_view['Daily_Sentiment_Score'].mean():.3f}")
        c2.metric("Positive days",        f"{(sentiment_view['Daily_Sentiment_Score'] > 0).mean():.1%}")
        c3.metric("Avg articles / day",   f"{sentiment_view['Daily_Article_Count'].mean():.0f}")

        st.markdown("---")
        st.subheader("Sentiment → Next-Day Return")

        scatter_df = view[view["Daily_Article_Count"] > 0].copy()
        scatter_df["Next_Return"] = scatter_df["Target_Return"].shift(-1)
        scatter_df = scatter_df.dropna(subset=["Next_Return"])

        # Manual OLS trendline to avoid statsmodels dependency
        m, b = np.polyfit(scatter_df["Daily_Sentiment_Score"], scatter_df["Next_Return"], 1)
        x_line = np.linspace(scatter_df["Daily_Sentiment_Score"].min(),
                             scatter_df["Daily_Sentiment_Score"].max(), 100)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=scatter_df["Daily_Sentiment_Score"], y=scatter_df["Next_Return"],
            mode="markers", marker=dict(size=4, color="steelblue", opacity=0.4),
            name="Trading day",
        ))
        fig3.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", line=dict(color="red", width=2, dash="dash"),
            name=f"OLS trend (slope={m:.5f})",
        ))
        fig3.update_layout(
            xaxis_title="Daily Sentiment Score",
            yaxis_title="Next-Day Return",
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# TAB 3 — Model Performance
# ===========================================================================
with tab3:
    st.subheader("Random Forest Regressor — Test Set Results")

    mae      = mean_absolute_error(y_test, preds)
    rmse     = np.sqrt(mean_squared_error(y_test, preds))
    r2       = r2_score(y_test, preds)
    dir_acc  = float(np.mean(np.sign(y_test) == np.sign(preds)))
    base_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",  f"{mae:.6f}",  delta=f"{mae - base_mae:+.6f} vs baseline", delta_color="inverse")
    c2.metric("RMSE", f"{rmse:.6f}")
    c3.metric("R²",   f"{r2:.4f}")
    c4.metric("Directional Accuracy", f"{dir_acc:.2%}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        fi_df = (
            importances.reset_index()
            .rename(columns={"index": "Feature", 0: "Importance"})
            .sort_values("Importance")
        )
        fig4 = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title="Feature Importance",
            color="Importance", color_continuous_scale="Blues",
        )
        fig4.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig4, use_container_width=True)

    with col_right:
        errors = np.array(y_test) - preds
        fig5 = px.histogram(
            x=errors, nbins=60, title="Prediction Error Distribution",
            labels={"x": "Actual − Predicted", "y": "Frequency"},
            color_discrete_sequence=["coral"],
        )
        fig5.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="zero error")
        fig5.update_layout(height=320)
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Actual vs Predicted Returns (last year of test set)")
    plot_n = min(252, len(y_test))
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=dates_test.values[-plot_n:], y=np.array(y_test)[-plot_n:],
        name="Actual", line=dict(color="#1f77b4", width=1), opacity=0.85,
    ))
    fig6.add_trace(go.Scatter(
        x=dates_test.values[-plot_n:], y=preds[-plot_n:],
        name="Predicted", line=dict(color="#ff7f0e", width=1), opacity=0.85,
    ))
    fig6.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        height=360,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig6, use_container_width=True)
