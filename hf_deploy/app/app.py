"""
ParkSense AI — Gradio dashboard for Hugging Face Spaces.
Inference only. Reads pre-computed CSVs + the trained pkl. No training code here.
"""
import pathlib
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from inference import ParkSensePredictor

BASE = pathlib.Path(__file__).parent.parent
HOTSPOTS = pd.read_csv(BASE / "model" / "hotspots_with_impact.csv")
DISPATCH = pd.read_csv(BASE / "model" / "enforcement_dispatch.csv")

_REQUIRED_HOTSPOT_COLS = {"impact_rank", "area_name", "police_station",
                          "congestion_pcu_hrs_day", "econ_loss_week_inr",
                          "PSI", "tier", "lat", "lon"}
_missing = _REQUIRED_HOTSPOT_COLS - set(HOTSPOTS.columns)
if _missing:
    raise ValueError(
        f"hotspots_with_impact.csv is missing required columns: {_missing}. "
        f"Re-export this file from the notebook's section 13 (Congestion Impact "
        f"Quantification) — do not substitute an older pipeline output."
    )

predictor = ParkSensePredictor()
META = predictor.model_card()

LIGHT_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="#F7F8FA",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E5EA",
    block_label_text_color="#3B4252",
    button_primary_background_fill="#1E5EFF",
    button_primary_text_color="#FFFFFF",
)

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#header {padding: 18px 0 4px 0;}
#header h1 {font-size: 22px; font-weight: 700; color: #1A2233; margin-bottom: 2px;}
#header p {color: #6B7280; font-size: 13px;}
.kpi-card {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 16px;}
.kpi-card .label {font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: .04em;}
.kpi-card .value {font-size: 24px; font-weight: 700; color: #111827;}
"""


def kpi_row():
    n_clusters   = HOTSPOTS["impact_rank"].max()
    n_critical   = (HOTSPOTS["tier"] == "Critical").sum() if "tier" in HOTSPOTS else 0
    pcu_total    = HOTSPOTS["congestion_pcu_hrs_day"].sum()
    econ_total   = (HOTSPOTS["econ_loss_week_inr"].sum() * 52) / 1e7
    return (
        f"### {n_clusters}\n**Hotspots detected**",
        f"### {n_critical}\n**Critical zones**",
        f"### {pcu_total:,.0f}\n**PCU-hrs blocked/day**",
        f"### ₹{econ_total:.2f} Cr\n**Annual economic loss**",
    )


def top_hotspots_table(n=15):
    cols = ["impact_rank", "area_name", "police_station", "congestion_pcu_hrs_day",
            "econ_loss_week_inr", "PSI", "tier"]
    return HOTSPOTS[cols].head(n)


def hotspot_map():
    fig = px.scatter_mapbox(
        HOTSPOTS, lat="lat", lon="lon",
        size="congestion_pcu_hrs_day", color="PSI",
        color_continuous_scale="OrRd",
        hover_name="area_name",
        hover_data={"police_station": True, "congestion_pcu_hrs_day": True, "tier": True,
                    "lat": False, "lon": False},
        zoom=10, height=480,
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#FFFFFF",
    )
    return fig


def dispatch_table(priority_filter):
    df = DISPATCH.copy()
    if priority_filter and priority_filter != "All":
        df = df[df["priority"] == priority_filter]
    officer_col = "officers" if "officers" in df.columns else "officers_required"
    saving_col  = "weekly_saving_inr" if "weekly_saving_inr" in df.columns else "weekly_saving_if_cleared_inr"
    cols = ["impact_rank", "area_name", officer_col, "shift", "barricade",
            "priority", saving_col]
    return df[cols].sort_values("impact_rank")


def fold_chart():
    fold_df = predictor.fold_results
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i}" for i in fold_df["fold"]],
        y=fold_df["auc"],
        marker_color="#1E5EFF",
        text=fold_df["auc"].round(3),
        textposition="outside",
    ))
    fig.add_hline(y=fold_df["auc"].mean(), line_dash="dash", line_color="#6B7280",
                  annotation_text=f"Mean {fold_df['auc'].mean():.3f}")
    fig.update_layout(
        height=320, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        yaxis_range=[0.5, 1.0], yaxis_title="AUC",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def shap_chart():
    sdf = predictor.shap_df.sort_values("mean_shap", ascending=True)
    fig = go.Figure(go.Bar(
        x=sdf["mean_shap"], y=sdf["feature"], orientation="h",
        marker_color="#1E5EFF",
    ))
    fig.update_layout(
        height=360, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis_title="Mean |SHAP| value",
        margin=dict(l=140, r=20, t=20, b=40),
    )
    return fig


def predict_ui(junction_name, hour, dow, roll_mean, roll_std,
                lag_1d, lag_7d, heavy_pct, compound_pct, max_severity, month_ord):
    row = {
        "junction_name": junction_name, "hour": int(hour), "dow": int(dow),
        "month_ord": int(month_ord), "roll_mean": float(roll_mean),
        "roll_std": float(roll_std), "lag_1d_same_hour": float(lag_1d),
        "lag_7d_same_hour": float(lag_7d), "heavy_pct": float(heavy_pct),
        "compound_pct": float(compound_pct), "max_severity": float(max_severity),
    }
    result = predictor.predict_single(row)
    verdict = "🔴 SPIKE PREDICTED" if result["is_spike"] else "🟢 Normal"
    note = " (unseen junction — using global fallback encoding)" if result["unseen_junction"] else ""
    return (
        f"## {verdict}{note}\n\n"
        f"**Calibrated probability:** {result['calibrated_prob']}\n\n"
        f"**Raw model score:** {result['raw_score']}\n\n"
        f"**Decision threshold:** {result['threshold_used']}"
    )


with gr.Blocks(theme=LIGHT_THEME, css=CSS, title="ParkSense AI") as demo:
    with gr.Row(elem_id="header"):
        gr.Markdown(
            "# ParkSense AI\n"
            "Parking-Induced Congestion Intelligence · Bengaluru Traffic Police (ASTraM)\n\n"
            f"Model trained {META.get('trained_at','')[:10]} · "
            f"Honest OOF AUC {META.get('honest_auc')} · "
            f"CV AUC {META.get('cv_auc_mean')} ± {META.get('cv_auc_std')}"
        )

    with gr.Row():
        k1, k2, k3, k4 = kpi_row()
        gr.Markdown(k1, elem_classes="kpi-card")
        gr.Markdown(k2, elem_classes="kpi-card")
        gr.Markdown(k3, elem_classes="kpi-card")
        gr.Markdown(k4, elem_classes="kpi-card")

    with gr.Tab("Hotspot Map"):
        gr.Plot(hotspot_map)
        gr.Dataframe(top_hotspots_table(), label="Top 15 hotspots by congestion impact")

    with gr.Tab("Enforcement Dispatch"):
        priority_dd = gr.Dropdown(
            choices=["All"] + sorted(DISPATCH["priority"].unique().tolist()),
            value="All", label="Filter by priority"
        )
        dispatch_df = gr.Dataframe(dispatch_table("All"))
        priority_dd.change(dispatch_table, inputs=priority_dd, outputs=dispatch_df)

    with gr.Tab("Model Performance"):
        with gr.Row():
            gr.Plot(fold_chart, label="Temporal CV — AUC per fold")
            gr.Plot(shap_chart, label="SHAP feature importance")
        gr.Markdown(
            f"**Target:** {META.get('target')}\n\n"
            f"**Brier score:** {META.get('brier')} · "
            f"**Average precision:** {META.get('avg_precision')} · "
            f"**Training rows:** {META.get('n_train'):,} · "
            f"**Junctions:** {META.get('n_junctions')}"
        )

    with gr.Tab("Live Inference"):
        gr.Markdown("Score a single junction-hour observation against the trained model.")
        with gr.Row():
            with gr.Column():
                jct_in   = gr.Dropdown(choices=predictor.known_junctions(), label="Junction")
                hour_in  = gr.Slider(0, 23, value=9, step=1, label="Hour")
                dow_in   = gr.Slider(0, 6, value=1, step=1, label="Day of week (0=Mon)")
                month_in = gr.Number(value=5, label="Month ordinal (training-relative)")
            with gr.Column():
                roll_mean_in = gr.Number(value=3.0, label="Rolling 14-day mean (causal)")
                roll_std_in  = gr.Number(value=1.5, label="Rolling 14-day std (causal)")
                lag1_in      = gr.Number(value=3.0, label="Same junction+hour, yesterday")
                lag7_in      = gr.Number(value=3.0, label="Same junction+hour, 7 days ago")
            with gr.Column():
                heavy_in    = gr.Slider(0, 1, value=0.1, label="Heavy vehicle %")
                compound_in = gr.Slider(0, 1, value=0.1, label="Compound violation %")
                severity_in = gr.Slider(0, 1, value=0.6, label="Max severity")
        predict_btn = gr.Button("Run prediction", variant="primary")
        result_md = gr.Markdown()
        predict_btn.click(
            predict_ui,
            inputs=[jct_in, hour_in, dow_in, roll_mean_in, roll_std_in,
                    lag1_in, lag7_in, heavy_in, compound_in, severity_in, month_in],
            outputs=result_md,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
