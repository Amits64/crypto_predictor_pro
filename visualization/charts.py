import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from preprocessing.indicators import calculate_heikin_ashi

def display_live_chart(df: pd.DataFrame, candle_type: str, prediction: float = None) -> go.Figure:
    plot_df = df.copy()
    if candle_type=="Heikin-Ashi":
        plot_df = calculate_heikin_ashi(plot_df)
        o,h,l,c = "ha_open","ha_high","ha_low","ha_close"
    else:
        o,h,l,c = "open","high","low","close"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=plot_df.datetime, open=plot_df[o], high=plot_df[h],
        low=plot_df[l], close=plot_df[c],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    if prediction is not None:
        t0 = plot_df.datetime.iloc[-1]
        t1 = t0 + pd.Timedelta(minutes=5)
        fig.add_trace(go.Scatter(
            x=[t0,t1], y=[plot_df[c].iloc[-1],prediction],
            mode="lines+markers", name="AI Pred",
            line=dict(color="gold",dash="dot"), marker=dict(symbol="diamond")
        ), row=1, col=1)

    colors = ['#26a69a' if cl>=op else '#ef5350'
              for cl,op in zip(plot_df[c], plot_df[o])]

    fig.add_trace(go.Bar(
        x=plot_df.datetime, y=plot_df.volume, marker_color=colors
    ), row=2, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      height=450)
    return fig
