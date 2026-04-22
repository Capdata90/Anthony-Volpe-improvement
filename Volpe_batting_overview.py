"""
======================================================
 ANTHONY VOLPE — SWING DECISION & SHADOW ZONE ANALYSIS
 Yankees Front Office Type Report | 2025 Season

 HOW TO RUN:
   1. pip install pybaseball pandas plotly kaleido jinja2
   2. Run this file in PyCharm (Play button)
   3. Outputs: volpe_report.html + volpe_report.pdf
      (saved in the same folder as this script)
======================================================
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Template

# ── 1. FETCH DATA ──────────────────────────────────────────────────────────────
print("⏳ Fetching Statcast data for Anthony Volpe (2025)...")

try:
    from pybaseball import statcast_batter
    from pybaseball import cache

    cache.enable()
except ImportError:
    print("❌  pybaseball not found. Run:  pip install pybaseball")
    sys.exit(1)

# Anthony Volpe MLBAM ID = 694497
VOLPE_ID = 683011
START_DATE = "2025-03-27"
END_DATE = "2025-10-08"

df_raw = statcast_batter(START_DATE, END_DATE, player_id=VOLPE_ID)

if df_raw.empty:
    print("❌  No data returned. Check dates or player ID.")
    sys.exit(1)

df = df_raw.copy()
print(f"✅  {len(df)} pitches loaded  |  {df['game_date'].min()} → {df['game_date'].max()}")


# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────────

# Shadow Zone definition (Statcast zones 11-14 = border, 1-9 = inside, 13-14 overlap)
# Official Savant zone mapping:
#   1-9  = strike zone (heart + standard)
#   11-14 = shadow zone (edge)
#   21-24 = chase zone
#   31-35 = waste zone

def classify_zone(zone):
    if pd.isna(zone):
        return "Unknown"
    z = int(zone)
    if z in range(1, 10):
        return "Heart/Strike"
    elif z in [11, 12, 13, 14]:
        return "Shadow"
    elif z in [21, 22, 23, 24]:
        return "Chase"
    else:
        return "Waste"


df["zone_label"] = df["zone"].apply(classify_zone)

# Whiff = swing and miss
df["is_whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked"]).astype(int)
df["is_swing"] = df["description"].isin([
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "hit_into_play"
]).astype(int)
df["is_chase"] = ((df["zone_label"].isin(["Chase", "Shadow"])) & (df["is_swing"] == 1)).astype(int)
df["is_contact"] = df["description"].isin(["foul", "foul_tip", "hit_into_play"]).astype(int)

# Count label
df["count_label"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

# Pitch type clean names
PITCH_NAMES = {
    "FF": "4-Seam FB", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
    "CH": "Changeup", "FS": "Splitter", "KC": "Knuckle-curve",
    "CS": "Slow Curve", "SV": "Slurve"
}
df["pitch_name"] = df["pitch_type"].map(PITCH_NAMES).fillna(df["pitch_type"])

# plate_x / plate_z for heatmap (feet, home plate center = 0)
df = df.dropna(subset=["plate_x", "plate_z"])

print(f"   Whiff rate overall : {df['is_whiff'].sum() / max(df['is_swing'].sum(), 1):.1%}")
print(f"   Shadow zone pitches: {(df['zone_label'] == 'Shadow').sum()}")

# ── 3. COLOR PALETTE (Yankees navy + accent) ───────────────────────────────────
NYY_NAVY = "#003087"
NYY_GRAY = "#C4CED4"
NYY_WHITE = "#FFFFFF"
NYY_RED = "#E4002B"
BG_DARK = "#0D1117"
BG_CARD = "#161B22"
TEXT_LIGHT = "#E6EDF3"
TEXT_MUTED = "#8B949E"
ACCENT = "#1F6FEB"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_DARK,
        font=dict(family="Arial, sans-serif", color=TEXT_LIGHT, size=12),
        margin=dict(l=50, r=30, t=55, b=50),
    ))


# ── 4. FIGURE 1 — STRIKE ZONE HEATMAP (Whiff%) ────────────────────────────────
print("📊 Building Figure 1: Strike Zone Whiff% Heatmap...")


def build_heatmap(data, title_suffix="All Counts"):
    d = data.copy()
    x_bins = np.linspace(-1.5, 1.5, 20)
    z_bins = np.linspace(1.0, 4.5, 20)
    d["xbin"] = pd.cut(d["plate_x"], bins=x_bins, labels=False)
    d["zbin"] = pd.cut(d["plate_z"], bins=z_bins, labels=False)

    grid = d.groupby(["xbin","zbin"]).agg(
        swings=("is_swing","sum"),
        whiffs=("is_whiff","sum")
    ).reset_index()
    grid["whiff_pct"] = (grid["whiffs"] / grid["swings"].clip(lower=1) * 100).round(1)
    grid["xbin"] = grid["xbin"].astype(float)
    grid["zbin"] = grid["zbin"].astype(float)

    x_centers = [(x_bins[i]+x_bins[i+1])/2 for i in range(len(x_bins)-1)]
    z_centers = [(z_bins[i]+z_bins[i+1])/2 for i in range(len(z_bins)-1)]

    z_matrix = np.full((len(z_centers), len(x_centers)), np.nan)
    for _, row in grid.iterrows():
        xi = int(row["xbin"])
        zi = int(row["zbin"])
        if 0 <= xi < len(x_centers) and 0 <= zi < len(z_centers):
            z_matrix[zi][xi] = row["whiff_pct"]

    fig = go.Figure(go.Heatmap(
        z=z_matrix[::-1],
        x=x_centers,
        y=list(reversed(z_centers)),
        colorscale=[[0,"#003087"],[0.4,"#1F6FEB"],[0.7,"#F0A500"],[1.0,"#E4002B"]],
        zmin=0, zmax=80,
        colorbar=dict(title="Whiff%", ticksuffix="%", thickness=14,
                      tickfont=dict(color=TEXT_LIGHT)),
        hoverongaps=False,
        hovertemplate="Whiff: %{z:.0f}%<extra></extra>"
    ))

    sz_top, sz_bot = 3.5, 1.5
    fig.add_shape(type="rect", x0=-0.708, x1=0.708, y0=sz_bot, y1=sz_top,
                  line=dict(color=NYY_WHITE, width=2, dash="dash"))
    fig.add_shape(type="rect", x0=-1.0, x1=1.0, y0=sz_bot-0.25, y1=sz_top+0.25,
                  line=dict(color=NYY_GRAY, width=1, dash="dot"))
    fig.add_annotation(x=1.05, y=sz_top+0.25, text="Shadow", showarrow=False,
                       font=dict(color=NYY_GRAY, size=10), xanchor="left")
    fig.add_annotation(x=0.72, y=sz_top, text="Strike Zone", showarrow=False,
                       font=dict(color=NYY_WHITE, size=10), xanchor="left")

    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_DARK,
        font=dict(family="Arial, sans-serif", color=TEXT_LIGHT, size=12),
        margin=dict(l=50, r=30, t=55, b=50),
        title=f"Whiff% by Location — {title_suffix}",
        xaxis=dict(title="Horizontal (ft from center)", range=[-1.6, 1.6],
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Height (ft)", range=[0.9, 4.6],
                   gridcolor="rgba(255,255,255,0.05)"),
        height=500,
        annotations=fig.layout.annotations
    )
    return fig


fig1 = build_heatmap(df, "2025 Season")

# ── 5. FIGURE 2 — CHASE RATE BY PITCH TYPE IN SHADOW ZONE ─────────────────────
print("📊 Building Figure 2: Chase Rate by Pitch Type (Shadow Zone)...")

shadow_df = df[df["zone_label"] == "Shadow"].copy()

chase_by_pitch = (
    shadow_df.groupby("pitch_name")
    .agg(pitches=("is_swing", "count"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"))
    .reset_index()
)
chase_by_pitch["chase_rate"] = (chase_by_pitch["swings"] / chase_by_pitch["pitches"] * 100).round(1)
chase_by_pitch["whiff_rate"] = (chase_by_pitch["whiffs"] / chase_by_pitch["swings"].clip(lower=1) * 100).round(1)
chase_by_pitch = chase_by_pitch[chase_by_pitch["pitches"] >= 3].sort_values("chase_rate", ascending=True)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y=chase_by_pitch["pitch_name"],
    x=chase_by_pitch["chase_rate"],
    orientation="h",
    name="Chase Rate",
    marker=dict(color=NYY_NAVY, line=dict(color=ACCENT, width=1)),
    text=chase_by_pitch["chase_rate"].apply(lambda x: f"{x:.0f}%"),
    textposition="outside",
    textfont=dict(color=TEXT_LIGHT, size=11),
    hovertemplate="%{y}: Chase %{x:.1f}%<br>Pitches: %{customdata}<extra></extra>",
    customdata=chase_by_pitch["pitches"]
))
fig2.add_trace(go.Bar(
    y=chase_by_pitch["pitch_name"],
    x=chase_by_pitch["whiff_rate"],
    orientation="h",
    name="Whiff Rate",
    marker=dict(color=NYY_RED, opacity=0.85),
    text=chase_by_pitch["whiff_rate"].apply(lambda x: f"{x:.0f}%"),
    textposition="outside",
    textfont=dict(color=TEXT_LIGHT, size=11),
    hovertemplate="%{y}: Whiff %{x:.1f}%<extra></extra>",
))

fig2.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    title="Chase Rate & Whiff% by Pitch Type — Shadow Zone Only",
    barmode="group",
    xaxis=dict(title="Rate (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(title=""),
    height=420,
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
)
fig2.add_vline(x=50, line_dash="dot", line_color=NYY_GRAY, opacity=0.4,
               annotation_text="50% baseline", annotation_font_color=TEXT_MUTED,
               annotation_position="top right")

# ── 6. FIGURE 3 — PITCH SEQUENCES WITH HIGHEST WHIFF RATE ─────────────────────
print("📊 Building Figure 3: Pitch Sequences — Whiff Rate...")

# Build sequences: previous pitch → current pitch, result
df_sorted = df.sort_values(["game_date", "at_bat_number", "pitch_number"])
df_sorted["prev_pitch"] = df_sorted.groupby(["game_date", "at_bat_number"])["pitch_name"].shift(1)
seq_df = df_sorted.dropna(subset=["prev_pitch"]).copy()
seq_df["sequence"] = seq_df["prev_pitch"] + " → " + seq_df["pitch_name"]

seq_stats = (
    seq_df.groupby("sequence")
    .agg(count=("is_whiff", "count"), whiffs=("is_whiff", "sum"))
    .reset_index()
)
seq_stats["whiff_rate"] = (seq_stats["whiffs"] / seq_stats["count"] * 100).round(1)
seq_stats = seq_stats[seq_stats["count"] >= 3].sort_values("whiff_rate", ascending=False).head(12)

colors = [NYY_RED if r >= 40 else ACCENT if r >= 25 else NYY_NAVY
          for r in seq_stats["whiff_rate"]]

fig3 = go.Figure(go.Bar(
    x=seq_stats["sequence"],
    y=seq_stats["whiff_rate"],
    marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
    text=seq_stats.apply(lambda r: f"{r['whiff_rate']:.0f}%<br>n={r['count']}", axis=1),
    textposition="outside",
    textfont=dict(color=TEXT_LIGHT, size=10),
    hovertemplate="Sequence: %{x}<br>Whiff: %{y:.1f}%<br>Pitches: %{customdata}<extra></extra>",
    customdata=seq_stats["count"]
))
fig3.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    title="Top Pitch Sequences by Whiff Rate (min. 3 occurrences)",
    xaxis=dict(title="", tickangle=-35, gridcolor="rgba(255,255,255,0.03)"),
    yaxis=dict(title="Whiff%", ticksuffix="%", gridcolor="rgba(255,255,255,0.05)"),
    height=450,
)
fig3.add_hline(y=40, line_dash="dot", line_color=NYY_RED, opacity=0.5,
               annotation_text="High danger (40%+)", annotation_font_color=NYY_RED,
               annotation_position="top right")

# ── 7. FIGURE 4 — WHIFF% BY COUNT (matrix) ────────────────────────────────────
print("📊 Building Figure 4: Whiff% by Count...")

COUNT_ORDER = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]

count_stats = (
    df.groupby("count_label")
    .agg(swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"), pitches=("is_swing", "count"))
    .reset_index()
)
count_stats["whiff_rate"] = (count_stats["whiffs"] / count_stats["swings"].clip(lower=1) * 100).round(1)
count_stats["swing_rate"] = (count_stats["swings"] / count_stats["pitches"] * 100).round(1)
count_stats = count_stats[count_stats["count_label"].isin(COUNT_ORDER)]
count_stats["count_label"] = pd.Categorical(count_stats["count_label"], categories=COUNT_ORDER, ordered=True)
count_stats = count_stats.sort_values("count_label")

fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(go.Bar(
    x=count_stats["count_label"],
    y=count_stats["whiff_rate"],
    name="Whiff%",
    marker=dict(color=[
        NYY_RED if c in ["0-2", "1-2", "2-2", "3-2"] else ACCENT
        for c in count_stats["count_label"]
    ], opacity=0.9),
    text=count_stats["whiff_rate"].apply(lambda x: f"{x:.0f}%"),
    textposition="outside",
    textfont=dict(color=TEXT_LIGHT, size=10),
), secondary_y=False)

fig4.add_trace(go.Scatter(
    x=count_stats["count_label"],
    y=count_stats["swing_rate"],
    name="Swing%",
    mode="lines+markers",
    line=dict(color=NYY_GRAY, width=2, dash="dot"),
    marker=dict(size=7, color=NYY_GRAY),
    hovertemplate="Count %{x}<br>Swing: %{y:.1f}%<extra></extra>",
), secondary_y=True)

fig4.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    title="Whiff% & Swing% by Count — Pitcher's Advantage Counts Highlighted",
    height=420,
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    bargap=0.25,
)
fig4.update_yaxes(title_text="Whiff%", ticksuffix="%",
                  gridcolor="rgba(255,255,255,0.05)", secondary_y=False)
fig4.update_yaxes(title_text="Swing%", ticksuffix="%",
                  showgrid=False, secondary_y=True)
fig4.add_annotation(x="0-2", y=count_stats[count_stats["count_label"] == "0-2"]["whiff_rate"].values[0] + 8
if not count_stats[count_stats["count_label"] == "0-2"].empty else 50,
                    text="Pitcher's count", showarrow=True, arrowhead=2,
                    font=dict(color=NYY_RED, size=10), arrowcolor=NYY_RED)

# ── 8. KEY METRICS ─────────────────────────────────────────────────────────────
total_pitches = len(df)
total_swings = int(df["is_swing"].sum())
total_whiffs = int(df["is_whiff"].sum())
whiff_rate_pct = total_whiffs / max(total_swings, 1) * 100
shadow_pitches = int((df["zone_label"] == "Shadow").sum())
shadow_swings = int(df[df["zone_label"] == "Shadow"]["is_swing"].sum())
shadow_chase_rt = shadow_swings / max(shadow_pitches, 1) * 100
shadow_whiff_rt = df[df["zone_label"] == "Shadow"]["is_whiff"].sum() / max(
    df[df["zone_label"] == "Shadow"]["is_swing"].sum(), 1) * 100
games = df["game_date"].nunique()

# ── 9. CONVERT FIGS TO HTML ────────────────────────────────────────────────────
print("🔧 Rendering figures to HTML...")

cfg = dict(responsive=True, displayModeBar=False)
fig1_html = pio.to_html(fig1, include_plotlyjs='cdn', full_html=False, config=cfg)
fig2_html = pio.to_html(fig2, include_plotlyjs=False, full_html=False, config=cfg)
fig3_html = pio.to_html(fig3, include_plotlyjs=False, full_html=False, config=cfg)
fig4_html = pio.to_html(fig4, include_plotlyjs=False, full_html=False, config=cfg)

# ── 10. HTML TEMPLATE ──────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Anthony Volpe — Shadow Zone Report 2025</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0D1117;color:#E6EDF3;font-family:'Inter',Arial,sans-serif;font-size:14px;line-height:1.6}
  .page{max-width:1100px;margin:0 auto;padding:32px 24px}
  /* Header */
  .header{border-bottom:2px solid #003087;padding-bottom:20px;margin-bottom:28px;display:flex;align-items:flex-end;justify-content:space-between}
  .header-left h1{font-size:26px;font-weight:700;letter-spacing:-0.5px;color:#fff}
  .header-left h2{font-size:14px;font-weight:400;color:#8B949E;margin-top:4px}
  .header-right{text-align:right;font-size:12px;color:#8B949E}
  .header-right span{display:block;color:#C4CED4;font-weight:500}
  /* Metric cards */
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}
  .metric-card{background:#161B22;border:0.5px solid #30363D;border-radius:10px;padding:14px 16px}
  .metric-label{font-size:11px;color:#8B949E;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
  .metric-value{font-size:26px;font-weight:600;color:#fff}
  .metric-sub{font-size:11px;color:#8B949E;margin-top:3px}
  .metric-card.danger .metric-value{color:#E4002B}
  .metric-card.warning .metric-value{color:#F0A500}
  .metric-card.good .metric-value{color:#2EA043}
  /* Insight box */
  .insight{background:#161B22;border-left:3px solid #003087;border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:28px}
  .insight-title{font-size:11px;color:#8B949E;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
  .insight p{font-size:13px;color:#C4CED4;line-height:1.7}
  .insight strong{color:#fff}
  /* Chart sections */
  .section{margin-bottom:36px}
  .section-header{display:flex;align-items:center;gap:10px;margin-bottom:14px}
  .section-num{background:#003087;color:#fff;font-size:11px;font-weight:600;width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0}
  .section-title{font-size:15px;font-weight:600;color:#fff}
  .section-sub{font-size:12px;color:#8B949E;margin-left:32px;margin-bottom:12px;margin-top:-8px}
  .chart-wrap{background:#161B22;border:0.5px solid #30363D;border-radius:10px;padding:16px;overflow:hidden}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  /* Footer */
  .footer{border-top:1px solid #21262D;margin-top:40px;padding-top:16px;font-size:11px;color:#484F58;display:flex;justify-content:space-between}
  .tag{display:inline-block;background:#21262D;color:#8B949E;font-size:10px;padding:2px 8px;border-radius:20px;margin-right:4px}
  @media print{body{background:#fff;color:#000}.metric-card,.chart-wrap,.insight{border:1px solid #ccc}}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div class="header-left">
      <h1>Anthony Volpe — Swing Decision & Shadow Zone</h1>
      <h2>Plate Discipline Deep-Dive · 2025 Season · New York Yankees</h2>
    </div>
    <div class="header-right">
      <span>{{ games }} games analyzed</span>
      {{ start_date }} – {{ end_date }}<br>
      Data: MLB Statcast via pybaseball
    </div>
  </div>

  <!-- METRICS -->
  <div class="metrics">
    <div class="metric-card">
      <div class="metric-label">Total Pitches</div>
      <div class="metric-value">{{ total_pitches }}</div>
      <div class="metric-sub">{{ total_swings }} swings</div>
    </div>
    <div class="metric-card {% if whiff_rate > 30 %}danger{% elif whiff_rate > 22 %}warning{% else %}good{% endif %}">
      <div class="metric-label">Overall Whiff%</div>
      <div class="metric-value">{{ "%.1f"|format(whiff_rate) }}%</div>
      <div class="metric-sub">MLB avg ≈ 24%</div>
    </div>
    <div class="metric-card {% if shadow_chase > 55 %}danger{% elif shadow_chase > 45 %}warning{% else %}good{% endif %}">
      <div class="metric-label">Shadow Zone Swing%</div>
      <div class="metric-value">{{ "%.1f"|format(shadow_chase) }}%</div>
      <div class="metric-sub">{{ shadow_pitches }} pitches in shadow</div>
    </div>
    <div class="metric-card {% if shadow_whiff > 35 %}danger{% elif shadow_whiff > 25 %}warning{% else %}good{% endif %}">
      <div class="metric-label">Shadow Zone Whiff%</div>
      <div class="metric-value">{{ "%.1f"|format(shadow_whiff) }}%</div>
      <div class="metric-sub">On swings in shadow zone</div>
    </div>
  </div>

  <!-- INSIGHT -->
  <div class="insight">
    <div class="insight-title">Scout Insight</div>
    <p>
      Volpe's shadow zone swing rate of <strong>{{ "%.1f"|format(shadow_chase) }}%</strong> indicates
      he is being exploited at the edges of the strike zone. Pitchers with late-breaking
      offspeed sequences drive his whiff rate above <strong>{{ "%.1f"|format(shadow_whiff) }}%</strong>
      in the shadow zone — well above the MLB average for contact hitters.
      Addressing this will directly improve his walk rate and reduce two-strike vulnerability.
    </p>
  </div>

  <!-- CHARTS ROW 1: Heatmap + Chase by Pitch -->
  <div class="two-col">
    <div class="section">
      <div class="section-header">
        <div class="section-num">1</div>
        <div class="section-title">Whiff% Heatmap by Location</div>
      </div>
      <div class="section-sub">Dashed = Strike Zone &nbsp;|&nbsp; Dotted = Shadow Zone</div>
      <div class="chart-wrap">{{ fig1_html }}</div>
    </div>
    <div class="section">
      <div class="section-header">
        <div class="section-num">2</div>
        <div class="section-title">Chase & Whiff% by Pitch Type</div>
      </div>
      <div class="section-sub">Shadow zone pitches only (≥3 occurrences)</div>
      <div class="chart-wrap">{{ fig2_html }}</div>
    </div>
  </div>

  <!-- CHART: Sequences -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">3</div>
      <div class="section-title">Pitch Sequences — Highest Whiff Rate</div>
    </div>
    <div class="section-sub">
      Two-pitch sequences ranked by whiff%. Red bars = high danger (40%+). Minimum 3 occurrences.
    </div>
    <div class="chart-wrap">{{ fig3_html }}</div>
  </div>

  <!-- CHART: Count -->
  <div class="section">
    <div class="section-header">
      <div class="section-num">4</div>
      <div class="section-title">Whiff% & Swing% by Count</div>
    </div>
    <div class="section-sub">
      Red bars = two-strike counts. Dotted line = swing rate (secondary axis).
    </div>
    <div class="chart-wrap">{{ fig4_html }}</div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div>
      <span class="tag">CONFIDENTIAL</span>
      <span class="tag">NYY Analytics</span>
      <span class="tag">2025</span>
    </div>
    <div>Data sourced from MLB Statcast · pybaseball · Baseball Savant</div>
  </div>

</div>
</body>
</html>"""

# ── 11. RENDER & SAVE HTML ─────────────────────────────────────────────────────
print("💾 Saving HTML report...")

template = Template(HTML_TEMPLATE)
html_out = template.render(
    games=games,
    start_date=START_DATE,
    end_date=END_DATE,
    total_pitches=total_pitches,
    total_swings=total_swings,
    whiff_rate=whiff_rate_pct,
    shadow_pitches=shadow_pitches,
    shadow_chase=shadow_chase_rt,
    shadow_whiff=shadow_whiff_rt,
    fig1_html=fig1_html,
    fig2_html=fig2_html,
    fig3_html=fig3_html,
    fig4_html=fig4_html,
)

out_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(out_dir, "volpe_report.html")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_out)
print(f"✅  HTML saved → {html_path}")

# ── 12. EXPORT PDF (via kaleido — static per figure, assembled) ────────────────
print("📄 Exporting PDF (individual figures via kaleido)...")

try:
    import subprocess, tempfile, shutil

    # Save each figure as PNG, then assemble in HTML→PDF
    fig_paths = []
    for i, fig in enumerate([fig1, fig2, fig3, fig4], 1):
        p = os.path.join(out_dir, f"_fig{i}.png")
        fig.write_image(p, width=1050, height=fig.layout.height or 450, scale=2)
        fig_paths.append(p)

    PDF_TEMPLATE = f"""<!DOCTYPE html><html><head>
    <style>
      body{{margin:0;background:#0D1117;font-family:Arial,sans-serif;color:#E6EDF3}}
      .page{{max-width:1050px;margin:0 auto;padding:40px 32px}}
      h1{{font-size:22px;font-weight:700;color:#fff;border-bottom:2px solid #003087;padding-bottom:12px;margin-bottom:6px}}
      h2{{font-size:13px;font-weight:400;color:#8B949E;margin-bottom:24px}}
      .metrics{{display:flex;gap:12px;margin-bottom:24px}}
      .mc{{flex:1;background:#161B22;border:1px solid #30363D;border-radius:8px;padding:12px}}
      .ml{{font-size:10px;color:#8B949E;text-transform:uppercase;letter-spacing:0.5px}}
      .mv{{font-size:22px;font-weight:700;color:#fff;margin:4px 0}}
      .ms{{font-size:10px;color:#8B949E}}
      .insight{{background:#161B22;border-left:3px solid #003087;padding:12px 16px;margin-bottom:24px;border-radius:0 6px 6px 0;font-size:12px;color:#C4CED4;line-height:1.7}}
      .row{{display:flex;gap:16px;margin-bottom:16px}}
      .row img{{width:50%;border-radius:8px;background:#161B22}}
      img.full{{width:100%;border-radius:8px;background:#161B22;margin-bottom:16px}}
      .footer{{margin-top:32px;font-size:10px;color:#484F58;border-top:1px solid #21262D;padding-top:12px;display:flex;justify-content:space-between}}
    </style></head><body><div class="page">
    <h1>Anthony Volpe — Swing Decision & Shadow Zone · 2025</h1>
    <h2>New York Yankees · {games} games · {START_DATE} – {END_DATE} · Source: MLB Statcast</h2>
    <div class="metrics">
      <div class="mc"><div class="ml">Total Pitches</div><div class="mv">{total_pitches}</div><div class="ms">{total_swings} swings</div></div>
      <div class="mc"><div class="ml">Overall Whiff%</div><div class="mv">{whiff_rate_pct:.1f}%</div><div class="ms">MLB avg ≈ 24%</div></div>
      <div class="mc"><div class="ml">Shadow Swing%</div><div class="mv">{shadow_chase_rt:.1f}%</div><div class="ms">{shadow_pitches} shadow pitches</div></div>
      <div class="mc"><div class="ml">Shadow Whiff%</div><div class="mv">{shadow_whiff_rt:.1f}%</div><div class="ms">On swings in shadow</div></div>
    </div>
    <div class="insight">
      Volpe's shadow zone swing rate of <b>{shadow_chase_rt:.1f}%</b> indicates exploitation at the edges.
      Offspeed sequences drive whiff rate to <b>{shadow_whiff_rt:.1f}%</b> in the shadow zone.
      Improvement here will directly raise his walk rate and reduce two-strike exposure.
    </div>
    <div class="row">
      <img src="_fig1.png"/><img src="_fig2.png"/>
    </div>
    <img class="full" src="_fig3.png"/>
    <img class="full" src="_fig4.png"/>
    <div class="footer">
      <span>CONFIDENTIAL · NYY Analytics · 2025</span>
      <span>Data: MLB Statcast via pybaseball · Baseball Savant</span>
    </div>
    </div></body></html>"""

    pdf_html_path = os.path.join(out_dir, "_pdf_source.html")
    pdf_path = os.path.join(out_dir, "volpe_report.pdf")

    with open(pdf_html_path, "w") as f:
        f.write(PDF_TEMPLATE)

    # Try weasyprint first, then playwright, then skip
    try:
        import weasyprint

        weasyprint.HTML(filename=pdf_html_path).write_pdf(pdf_path)
        print(f"✅  PDF saved → {pdf_path}  (via weasyprint)")
    except ImportError:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(f"file://{pdf_html_path}")
                page.pdf(path=pdf_path, format="A4", print_background=True)
                browser.close()
            print(f"✅  PDF saved → {pdf_path}  (via playwright)")
        except ImportError:
            print("⚠️  PDF skipped — install weasyprint OR playwright:")
            print("     pip install weasyprint")
            print("     pip install playwright && playwright install chromium")

    # Cleanup temp files
    for p in fig_paths:
        if os.path.exists(p): os.remove(p)
    if os.path.exists(pdf_html_path): os.remove(pdf_html_path)

except Exception as e:
    print(f"⚠️  PDF export error: {e}")

# ── 13. OPEN HTML IN BROWSER ───────────────────────────────────────────────────
import webbrowser

webbrowser.open(f"file://{html_path}")
print("\n🎯 Done! Browser opened with the interactive HTML report.")
print(f"   HTML → {html_path}")
print(f"   PDF  → {os.path.join(out_dir, 'volpe_report.pdf')}")