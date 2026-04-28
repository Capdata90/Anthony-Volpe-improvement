"""
==============================================================
 ANTHONY VOLPE — MATCHUP EXPLORER & PITCH PREDICTION
 New York Yankees Analytics type | 2023-2025

 HOW TO RUN:
   streamlit run volpe_matchup_app.py
==============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pybaseball import statcast_batter, chadwick_register
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volpe Matchup Explorer | NYY Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLING ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0D1117; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Arial', sans-serif; }
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 0.5px solid #30363D;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #161B22; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #FFFFFF; border-bottom: 2px solid #003087; }
    .stDataFrame { border: 0.5px solid #30363D; border-radius: 8px; }
    .stSelectbox label, .stMultiSelect label { color: #C4CED4 !important; }
    .insight-box {
        background: #161B22;
        border-left: 3px solid #003087;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #C4CED4;
        font-size: 14px;
        line-height: 1.7;
    }
    .warning-box {
        background: #1C1A0F;
        border-left: 3px solid #F0A500;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #C4CED4;
        font-size: 14px;
    }
    .nYY-header {
        background: linear-gradient(90deg, #003087 0%, #161B22 100%);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 24px;
        border: 0.5px solid #003087;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
VOLPE_ID  = 683011
NYY_TEAM  = "NYY"

PITCH_NAMES = {
    "FF": "4-Seam Fastball", "SI": "Sinker",     "FC": "Cutter",
    "SL": "Slider",          "ST": "Sweeper",    "CU": "Curveball",
    "CH": "Changeup",        "FS": "Splitter",   "KC": "Knuckle-curve",
    "CS": "Slow Curve",      "SV": "Slurve",     "FO": "Forkball"
}

RESULT_COLORS = {
    "single":                    "#2EA043",
    "double":                    "#3FB950",
    "triple":                    "#56D364",
    "home_run":                  "#FFD700",
    "field_out":                 "#E4002B",
    "strikeout":                 "#B91C1C",
    "walk":                      "#1F6FEB",
    "hit_by_pitch":              "#58A6FF",
    "swinging_strike":           "#F0A500",
    "called_strike":             "#D29922",
    "foul":                      "#8B949E",
    "ball":                      "#30363D",
    "sac_fly":                   "#D2A679",
    "force_out":                 "#C0392B",
    "grounded_into_double_play": "#922B21",
    "field_error":               "#9B59B6",
}


# ── PITCHER NAME LOOKUP ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_pitcher_name_map(pitcher_ids: list) -> dict:
    """
    Returns {mlbam_id (int): 'First Last'} for every rival pitcher.
    Uses the Chadwick register — downloaded once and cached by pybaseball.
    Falls back to '#ID' when a pitcher is not found.
    """
    try:
        register = chadwick_register(save=False)
        reg = register.dropna(subset=["key_mlbam"]).copy()
        reg["key_mlbam"] = reg["key_mlbam"].astype(int)
        reg["full_name"] = (
            reg["name_first"].fillna("").str.strip()
            + " "
            + reg["name_last"].fillna("").str.strip()
        ).str.strip()
        lookup = reg.set_index("key_mlbam")["full_name"].to_dict()
    except Exception:
        lookup = {}

    result = {}
    for pid in pitcher_ids:
        try:
            pid_int = int(pid)
            name = lookup.get(pid_int, "").strip()
            result[pid_int] = name if name else f"#{pid_int}"
        except Exception:
            result[pid] = f"#{pid}"
    return result


# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_volpe_data():
    file_path = "volpe_data_2023_2025.csv"

    # PASO A: Intentar leer el archivo local primero (Súper rápido)
    try:
        df = pd.read_csv(file_path)
        df["game_date"] = pd.to_datetime(df["game_date"])
        # Aseguramos que los tipos de datos sean correctos tras la carga del CSV
        if "pitcher" in df.columns:
            df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").fillna(0).astype(int)
        print("✅ Datos cargados desde el CSV local.")
        return df
    except FileNotFoundError:
        # PASO B: Si el archivo no existe, descargamos de pybaseball (Solo una vez)
        st.warning("Generando CSV por primera vez... esto tardará un poco.")
        seasons = [
            ("2023-03-30", "2023-11-05"),
            ("2024-03-20", "2024-10-31"),
            ("2025-03-27", "2025-10-15"),
        ]
        frames = []
        for start, end in seasons:
            try:
                d = statcast_batter(start, end, player_id=VOLPE_ID)
                if not d.empty:
                    frames.append(d)
            except Exception as e:
                print(f"Error descargando temporada: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Guardamos para futuras ocasiones
        df.to_csv(file_path, index=False)
        print(f"¡Archivo {file_path} generado con éxito!")

        return df

    # MLB games only (R=regular, D=ALDS/NLDS, L=LCS, W=WS, F=wildcard)
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["R","D","L","W","F"])]

    # Remove Yankees pitchers (teammates)
    if "home_team" in df.columns and "away_team" in df.columns:
        df = df[~(
            ((df["inning_topbot"] == "Top")  & (df["home_team"] == NYY_TEAM)) |
            ((df["inning_topbot"] == "Bot") & (df["away_team"] == NYY_TEAM))
        )]

    df = df.dropna(subset=["plate_x","plate_z","pitch_type"])
    df["pitch_name"]  = df["pitch_type"].map(PITCH_NAMES).fillna(df["pitch_type"])
    df["count_label"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df["game_date"]   = pd.to_datetime(df["game_date"])
    df["season"]      = df["game_date"].dt.year
    df["is_playoff"]  = df["game_type"].isin(["D","L","W","F"]) if "game_type" in df.columns else False
    df["pitcher"]     = pd.to_numeric(df["pitcher"], errors="coerce")
    df = df.dropna(subset=["pitcher"])
    df["pitcher"]     = df["pitcher"].astype(int)
    return df.reset_index(drop=True)


# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["is_whiff"]     = df["description"].isin(["swinging_strike","swinging_strike_blocked"]).astype(int)
    df["is_swing"]     = df["description"].isin([
        "swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play"
    ]).astype(int)
    df["is_contact"]   = df["description"].isin(["foul","foul_tip","hit_into_play"]).astype(int)
    df["is_hit"]       = df["events"].isin(["single","double","triple","home_run"]).astype(int)
    df["runner_1b"]    = df["on_1b"].notna().astype(int)
    df["runner_2b"]    = df["on_2b"].notna().astype(int)
    df["runner_3b"]    = df["on_3b"].notna().astype(int)
    df["runners_on"]   = df["runner_1b"] + df["runner_2b"] + df["runner_3b"]
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["inning_late"]  = (df["inning"] >= 7).astype(int)
    return df


# ── ML MODEL ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_models(_df):
    df = engineer_features(_df).copy()
    features = [
        "balls","strikes","outs_when_up",
        "runner_1b","runner_2b","runner_3b",
        "p_throws_enc","inning","inning_late","runners_on"
    ]
    df_m  = df.dropna(subset=features + ["pitch_type"])
    cnts  = df_m["pitch_type"].value_counts()
    df_m  = df_m[df_m["pitch_type"].isin(cnts[cnts >= 10].index)]

    X  = df_m[features]
    le = LabelEncoder()
    y  = le.fit_transform(df_m["pitch_type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    xgb_m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, verbosity=0
    )
    xgb_m.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_m.predict(X_test))

    best  = xgb_m  if xgb_acc >= rf_acc else rf
    bname = "XGBoost" if xgb_acc >= rf_acc else "Random Forest"
    cm    = confusion_matrix(y_test, best.predict(X_test))

    return dict(rf=rf, xgb=xgb_m, rf_acc=rf_acc, xgb_acc=xgb_acc,
                le=le, features=features, best_model=best, best_name=bname,
                cm=cm, classes=le.classes_, feature_names=features)


def predict_pitch(models, balls, strikes, outs, r1, r2, r3, p_throws, inning):
    X = pd.DataFrame([{
        "balls":balls,"strikes":strikes,"outs_when_up":outs,
        "runner_1b":r1,"runner_2b":r2,"runner_3b":r3,
        "p_throws_enc":1 if p_throws=="R" else 0,
        "inning":inning,"inning_late":int(inning>=7),"runners_on":r1+r2+r3
    }])
    avg = (models["rf"].predict_proba(X)[0] + models["xgb"].predict_proba(X)[0]) / 2
    return sorted(
        [(PITCH_NAMES.get(c,c), round(p*100,1)) for c,p in zip(models["le"].classes_, avg)],
        key=lambda x: x[1], reverse=True
    )


# ── SPRAY CHART ────────────────────────────────────────────────────────────────
def draw_spray_chart(df_f):
    fig = go.Figure()
    theta = np.linspace(np.radians(45), np.radians(135), 100)
    r = 320
    fig.add_trace(go.Scatter(
        x=np.concatenate([[0],[r*np.cos(np.radians(45))],r*np.cos(theta),[r*np.cos(np.radians(135))],[0]]),
        y=np.concatenate([[0],[r*np.sin(np.radians(45))],r*np.sin(theta),[r*np.sin(np.radians(135))],[0]]),
        fill="toself", fillcolor="#1a2e1a",
        line=dict(color="#2ea043",width=1.5),
        mode="lines", showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[0,63.6,0,-63.6,0], y=[0,63.6,127.3,63.6,0],
        mode="lines", line=dict(color="#C4CED4",width=1.5),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(color="#FFFFFF",size=8,symbol="square"),
        showlegend=False, hoverinfo="skip"
    ))
    for angle in [45,135]:
        fig.add_trace(go.Scatter(
            x=[0,350*np.cos(np.radians(angle))],
            y=[0,350*np.sin(np.radians(angle))],
            mode="lines", line=dict(color="#C4CED4",width=1,dash="dash"),
            showlegend=False, hoverinfo="skip"
        ))

    hits = df_f.dropna(subset=["hc_x","hc_y"])
    if not hits.empty:
        fig.add_trace(go.Scatter(
            x=hits["hc_x"]-125, y=200-hits["hc_y"],
            mode="markers",
            marker=dict(
                color=[RESULT_COLORS.get(e,"#8B949E") for e in hits["events"].fillna("unknown")],
                size=9, opacity=0.85,
                line=dict(color="rgba(255,255,255,0.3)",width=0.5)
            ),
            customdata=np.stack([
                hits["events"].fillna("—"),
                hits["pitch_name"].fillna("—"),
                hits["release_speed"].fillna(0).round(1),
                hits["count_label"].fillna("—"),
                hits["game_date"].dt.strftime("%Y-%m-%d"),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pitch: %{customdata[1]} %{customdata[2]} mph<br>"
                "Count: %{customdata[3]} | Date: %{customdata[4]}<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        xaxis=dict(range=[-280,280],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(range=[-30,380],showgrid=False,zeroline=False,showticklabels=False,
                   scaleanchor="x",scaleratio=1),
        margin=dict(l=0,r=0,t=10,b=0), height=420,
    )
    return fig


# ── STRIKE ZONE ────────────────────────────────────────────────────────────────
def draw_strike_zone(df_f):
    fig = go.Figure()
    desc_map = {
        "swinging_strike":         ("Swinging Strike","#F0A500"),
        "swinging_strike_blocked": ("Swinging Strike","#F0A500"),
        "called_strike":           ("Called Strike",  "#E4002B"),
        "foul":                    ("Foul",           "#8B949E"),
        "foul_tip":                ("Foul Tip",       "#8B949E"),
        "ball":                    ("Ball",           "#1F6FEB"),
        "blocked_ball":            ("Ball",           "#1F6FEB"),
        "hit_into_play":           ("In Play",        "#2EA043"),
        "hit_by_pitch":            ("HBP",            "#58A6FF"),
    }
    for desc,(label,color) in desc_map.items():
        sub = df_f[df_f["description"]==desc]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["plate_x"], y=sub["plate_z"],
            mode="markers",
            marker=dict(color=color,size=8,opacity=0.75,
                        line=dict(color="rgba(255,255,255,0.2)",width=0.5)),
            name=label,
            customdata=np.stack([
                sub["pitch_name"].fillna("—"),
                sub["release_speed"].fillna(0).round(1),
                sub["count_label"].fillna("—"),
                sub["game_date"].dt.strftime("%Y-%m-%d"),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> %{customdata[1]} mph<br>"
                "Count: %{customdata[2]} | %{customdata[3]}<br>"
                "Location: (%{x:.2f}, %{y:.2f})<extra></extra>"
            )
        ))
    fig.add_shape(type="rect",x0=-0.708,x1=0.708,y0=1.5,y1=3.5,
                  line=dict(color="#FFFFFF",width=2,dash="dash"))
    fig.add_shape(type="rect",x0=-1.0,x1=1.0,y0=1.25,y1=3.75,
                  line=dict(color="#C4CED4",width=1,dash="dot"))
    fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3"),
        xaxis=dict(title="Horizontal (ft)",range=[-2,2],gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Height (ft)",range=[0.5,5],gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11)),
        margin=dict(l=40,r=10,t=10,b=40), height=420,
    )
    return fig


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def render_sidebar(df, pitcher_name_map):
    st.sidebar.markdown("""
    <div style='text-align:center; padding:12px 0 20px;'>
        <div style='font-size:16px; font-weight:700; color:#fff'>Volpe Analytics</div>
        <div style='font-size:11px; color:#8B949E'>NYY · 2023-2025</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### Matchup Filters")

    # Sorted list of "Last, First [ID]" — last name first so searching by last name works
    pitcher_display = sorted(
        [
            f"{name.split()[-1]}, {' '.join(name.split()[:-1])}  [{pid}]"
            if len(name.split()) > 1 else f"{name}  [{pid}]"
            for pid, name in pitcher_name_map.items()
        ]
    )

    selected_display = st.sidebar.selectbox(
        "Pitcher — type to search",
        options=pitcher_display,
        index=0,
        help="Type a last name to filter the list"
    )

    selected_pid  = int(selected_display.split("[")[-1].replace("]","").strip())
    selected_name = pitcher_name_map.get(selected_pid, f"#{selected_pid}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Game Context")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        balls   = st.selectbox("Balls",   [0,1,2,3],        index=0)
        outs    = st.selectbox("Outs",    [0,1,2],           index=0)
    with col2:
        strikes = st.selectbox("Strikes", [0,1,2],           index=0)
        inning  = st.selectbox("Inning",  list(range(1,13)), index=0)

    st.sidebar.markdown("**Runners on Base**")
    r1 = st.sidebar.checkbox("1st Base")
    r2 = st.sidebar.checkbox("2nd Base")
    r3 = st.sidebar.checkbox("3rd Base")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pitch Filter")
    all_pitches    = ["All"] + sorted(df["pitch_name"].dropna().unique().tolist())
    selected_pitch = st.sidebar.selectbox("Pitch Type", all_pitches)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Season")
    seasons = st.sidebar.multiselect(
        "Include Seasons", options=[2023,2024,2025], default=[2023,2024,2025]
    )
    include_playoffs = st.sidebar.checkbox("Include Playoffs", value=True)

    return dict(
        pitcher_id=selected_pid, pitcher_name=selected_name,
        balls=balls, strikes=strikes, outs=outs, inning=inning,
        r1=int(r1), r2=int(r2), r3=int(r3),
        pitch_type=selected_pitch, seasons=seasons,
        include_playoffs=include_playoffs,
    )


# ── FILTER HELPERS ─────────────────────────────────────────────────────────────
def apply_filters(df, f):
    d = df[df["pitcher"] == f["pitcher_id"]].copy()
    if f["seasons"]:
        d = d[d["season"].isin(f["seasons"])]
    if not f["include_playoffs"]:
        d = d[~d["is_playoff"]]
    if f["pitch_type"] != "All":
        d = d[d["pitch_name"] == f["pitch_type"]]
    return d


def apply_context_filters(df, f):
    d = apply_filters(df, f)
    d = d[(d["balls"]==f["balls"]) & (d["strikes"]==f["strikes"]) & (d["outs_when_up"]==f["outs"])]
    if f["r1"]: d = d[d["on_1b"].notna()]
    if f["r2"]: d = d[d["on_2b"].notna()]
    if f["r3"]: d = d[d["on_3b"].notna()]
    return d


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MATCHUP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_matchup(df, f):
    name      = f["pitcher_name"]
    count_str = f"{f['balls']}-{f['strikes']}"

    st.markdown(f"""
    <div class='nYY-header'>
        <div style='font-size:22px;font-weight:700;color:#fff'>Volpe vs {name}</div>
        <div style='font-size:13px;color:#C4CED4;margin-top:4px'>
            Historical Matchup · Count: {count_str} · {f['outs']} out(s) · Inning {f['inning']}
        </div>
    </div>""", unsafe_allow_html=True)

    df_p  = apply_filters(df, f)
    df_cx = apply_context_filters(df, f)
    df_e  = engineer_features(df_p)

    total_pa    = df_p["at_bat_number"].nunique() if "at_bat_number" in df_p.columns else len(df_p)
    total_pitch = len(df_p)
    whiff_rate  = df_e["is_whiff"].sum() / max(df_e["is_swing"].sum(),1) * 100
    hit_rate    = df_e["is_hit"].sum()   / max(total_pitch,1) * 100
    k_total     = df_p[df_p["events"]=="strikeout"]["events"].count()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Pitches",     f"{total_pitch:,}")
    c2.metric("Plate Appearances", f"{total_pa:,}")
    c3.metric("Whiff%",            f"{whiff_rate:.1f}%")
    c4.metric("Hit Rate",          f"{hit_rate:.1f}%")
    c5.metric("Strikeouts",        f"{k_total}")

    st.markdown("---")

    runners = [b for b,v in [("1B",f["r1"]),("2B",f["r2"]),("3B",f["r3"])] if v]
    runners_label = ", ".join(runners) if runners else "Bases empty"

    if len(df_cx) > 0:
        st.markdown(f"""<div class='insight-box'>
            <b>Context Filter Active</b> — Count {count_str}, {f['outs']} out(s), {runners_label}<br>
            Found <b>{len(df_cx)} pitches</b> matching this context out of {total_pitch} total vs {name}.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='warning-box'>
            No pitches found for Count {count_str}, {f['outs']} outs, {runners_label}.
            Showing all pitches vs {name}.
        </div>""", unsafe_allow_html=True)
        df_cx = df_p

    tab1,tab2,tab3 = st.tabs(["Spray Chart","Strike Zone","Pitch Breakdown"])

    with tab1:
        c_l,c_r = st.columns([2,1])
        with c_l:
            st.markdown("##### Ball in Play")
            st.plotly_chart(draw_spray_chart(df_cx), use_container_width=True)
        with c_r:
            st.markdown("##### Results")
            if "events" in df_cx.columns:
                ev = df_cx["events"].dropna().value_counts().reset_index()
                ev.columns = ["Event","Count"]
                fig_ev = go.Figure(go.Bar(
                    x=ev["Count"], y=ev["Event"], orientation="h",
                    marker=dict(color=[RESULT_COLORS.get(e,"#8B949E") for e in ev["Event"]]),
                    text=ev["Count"], textposition="outside",
                    textfont=dict(color="#E6EDF3")
                ))
                fig_ev.update_layout(
                    paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
                    font=dict(color="#E6EDF3",size=12),
                    margin=dict(l=10,r=30,t=10,b=10),height=380,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.03)")
                )
                st.plotly_chart(fig_ev, use_container_width=True)

    with tab2:
        c_l,c_r = st.columns([2,1])
        with c_l:
            st.markdown("##### Pitch Locations")
            st.plotly_chart(draw_strike_zone(df_cx), use_container_width=True)
        with c_r:
            st.markdown("##### Pitch Mix")
            pm = df_cx["pitch_name"].value_counts().reset_index()
            pm.columns = ["Pitch","Count"]
            pm["Pct"] = (pm["Count"]/pm["Count"].sum()*100).round(1)
            fig_pie = go.Figure(go.Pie(
                labels=pm["Pitch"],values=pm["Count"],hole=0.45,
                marker=dict(colors=px.colors.qualitative.Bold),
                textinfo="label+percent",textfont=dict(color="#E6EDF3",size=11),
            ))
            fig_pie.update_layout(
                paper_bgcolor="#0D1117",font=dict(color="#E6EDF3"),
                showlegend=False,margin=dict(l=10,r=10,t=10,b=10),height=380,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.markdown("##### Pitch-by-Pitch Results")
        cols = ["game_date","season","is_playoff","count_label",
                "pitch_name","release_speed","description","events","plate_x","plate_z"]
        avail = [c for c in cols if c in df_cx.columns]
        d_disp = df_cx[avail].copy()
        d_disp.columns = [c.replace("_"," ").title() for c in avail]
        st.dataframe(d_disp.sort_values("Game Date",ascending=False),
                     use_container_width=True, height=400)

        if "release_speed" in df_cx.columns:
            st.markdown("##### Velocity by Pitch Type")
            vel = df_cx.groupby("pitch_name")["release_speed"].agg(
                ["mean","min","max","count"]).reset_index()
            vel.columns = ["Pitch Type","Avg Velo","Min","Max","Count"]
            vel = vel[vel["Count"]>=2].sort_values("Avg Velo",ascending=False)
            fig_vel = go.Figure(go.Bar(
                x=vel["Pitch Type"],y=vel["Avg Velo"],
                marker=dict(color="#1F6FEB"),
                error_y=dict(type="data",
                             array=(vel["Max"]-vel["Avg Velo"]).tolist(),
                             arrayminus=(vel["Avg Velo"]-vel["Min"]).tolist(),
                             color="#8B949E"),
                text=vel["Avg Velo"].round(1),textposition="outside",
                textfont=dict(color="#E6EDF3")
            ))
            fig_vel.update_layout(
                paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3",size=12),
                margin=dict(l=40,r=20,t=20,b=40),height=320,
                xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                yaxis=dict(title="mph",gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_vel, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PITCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_prediction(df, models, f):
    st.markdown("""
    <div class='nYY-header'>
        <div style='font-size:22px;font-weight:700;color:#fff'>Pitch Prediction Model</div>
        <div style='font-size:13px;color:#C4CED4;margin-top:4px'>
            Random Forest + XGBoost Ensemble · Trained on 2023-2025 MLB data
        </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Random Forest Accuracy", f"{models['rf_acc']*100:.1f}%")
    c2.metric("XGBoost Accuracy",       f"{models['xgb_acc']*100:.1f}%")
    c3.metric("Best Model",             models["best_name"])

    st.markdown("---")
    st.markdown(f"### Live Prediction — Count {f['balls']}-{f['strikes']} · {f['outs']} out(s) · Inning {f['inning']}")

    pitcher_data = df[df["pitcher"]==f["pitcher_id"]]
    p_throws = (pitcher_data["p_throws"].mode()[0]
                if not pitcher_data.empty and "p_throws" in pitcher_data.columns else "R")

    probs = predict_pitch(models,f["balls"],f["strikes"],f["outs"],
                          f["r1"],f["r2"],f["r3"],p_throws,f["inning"])
    top_pitch, top_prob = probs[0]

    st.markdown(f"""<div class='insight-box'>
        <b>Model Recommendation</b><br>
        <b>{top_pitch}</b> — <b>{top_prob:.1f}% probability</b><br>
        <span style='color:#8B949E;font-size:12px'>
        Ensemble RF + XGBoost · {len(df):,} historical pitches
        </span>
    </div>""", unsafe_allow_html=True)

    c_pred,c_hist = st.columns(2)
    with c_pred:
        st.markdown("##### Pitch Probability Distribution")
        labels = [p[0] for p in probs]
        values = [p[1] for p in probs]
        bar_colors = ["#E4002B" if i==0 else "#1F6FEB" if i==1 else "#8B949E"
                      for i in range(len(probs))]
        fig_pred = go.Figure(go.Bar(
            y=labels[::-1],x=values[::-1],orientation="h",
            marker=dict(color=bar_colors[::-1]),
            text=[f"{v:.1f}%" for v in values[::-1]],
            textposition="outside",textfont=dict(color="#E6EDF3")
        ))
        fig_pred.update_layout(
            paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3",size=12),
            margin=dict(l=10,r=60,t=10,b=10),height=350,
            xaxis=dict(title="%",range=[0,65],gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)")
        )
        st.plotly_chart(fig_pred,use_container_width=True)

    with c_hist:
        st.markdown("##### Actual Pitch Mix vs This Pitcher")
        df_p = apply_filters(df,f)
        am   = df_p["pitch_name"].value_counts(normalize=True).reset_index()
        am.columns = ["Pitch","Pct"]
        am["Pct"] = (am["Pct"]*100).round(1)
        fig_am = go.Figure(go.Bar(
            y=am["Pitch"][::-1],x=am["Pct"][::-1],orientation="h",
            marker=dict(color="#003087"),
            text=[f"{v:.1f}%" for v in am["Pct"][::-1]],
            textposition="outside",textfont=dict(color="#E6EDF3")
        ))
        fig_am.update_layout(
            paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3",size=12),
            margin=dict(l=10,r=60,t=10,b=10),height=350,
            xaxis=dict(title="%",range=[0,80],gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)")
        )
        st.plotly_chart(fig_am,use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Importance (XGBoost)")
    FEAT_LABELS = {
        "balls":"Ball Count","strikes":"Strike Count","outs_when_up":"Outs",
        "runner_1b":"Runner on 1B","runner_2b":"Runner on 2B","runner_3b":"Runner on 3B",
        "p_throws_enc":"Pitcher Handedness","inning":"Inning",
        "inning_late":"Late Inning (7+)","runners_on":"Total Runners On"
    }
    fi = pd.Series(models["xgb"].feature_importances_,
                   index=models["feature_names"]).sort_values(ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=[FEAT_LABELS.get(f,f) for f in fi.index],
        orientation="h",
        marker=dict(color=fi.values,
                    colorscale=[[0,"#003087"],[0.5,"#1F6FEB"],[1.0,"#E4002B"]],
                    showscale=False),
        text=[f"{v:.3f}" for v in fi.values],
        textposition="outside",textfont=dict(color="#E6EDF3",size=11)
    ))
    fig_fi.update_layout(
        paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3",size=12),
        margin=dict(l=10,r=80,t=10,b=10),height=380,
        xaxis=dict(title="Importance",gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)")
    )
    st.plotly_chart(fig_fi,use_container_width=True)

    st.markdown("### Confusion Matrix — Best Model")
    cls = [PITCH_NAMES.get(c,c) for c in models["classes"]]
    fig_cm = go.Figure(go.Heatmap(
        z=models["cm"],x=cls,y=cls,
        colorscale=[[0,"#0D1117"],[0.5,"#1F6FEB"],[1.0,"#E4002B"]],
        text=models["cm"].astype(str),
        texttemplate="%{text}",textfont=dict(size=11,color="#E6EDF3"),showscale=True
    ))
    fig_cm.update_layout(
        paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3",size=11),
        xaxis=dict(title="Predicted",tickangle=-35),
        yaxis=dict(title="Actual"),
        margin=dict(l=80,r=20,t=20,b=80),height=420
    )
    st.plotly_chart(fig_cm,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FRONT OFFICE REPORT
# ══════════════════════════════════════════════════════════════════════════════
def page_front_office(df, models, f):
    name = f["pitcher_name"]

    st.markdown(f"""
    <div class='nYY-header'>
        <div style='font-size:22px;font-weight:700;color:#fff'>Front Office Report</div>
        <div style='font-size:13px;color:#C4CED4;margin-top:4px'>
            Executive Summary · Volpe vs {name} · 2023-2025
        </div>
    </div>""", unsafe_allow_html=True)

    df_p = apply_filters(df,f)
    df_e = engineer_features(df_p)
    if df_p.empty:
        st.warning("No data available for this pitcher.")
        return

    total_pitches = len(df_p)
    whiff_rate    = df_e["is_whiff"].sum()/max(df_e["is_swing"].sum(),1)*100
    k_total       = df_p[df_p["events"]=="strikeout"]["events"].count()
    bb_total      = df_p[df_p["events"]=="walk"]["events"].count()
    hr_total      = df_p[df_p["events"]=="home_run"]["events"].count()
    hits_total    = df_p[df_p["events"].isin(["single","double","triple","home_run"])]["events"].count()

    st.markdown("#### Key Performance Indicators")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Pitches", f"{total_pitches:,}")
    c2.metric("Whiff%",        f"{whiff_rate:.1f}%",
              delta=f"{whiff_rate-24:.1f}% vs MLB avg", delta_color="inverse")
    c3.metric("Strikeouts",    f"{k_total}")
    c4.metric("Walks",         f"{bb_total}")
    c5.metric("Home Runs",     f"{hr_total}")
    c6.metric("Hits",          f"{hits_total}")

    st.markdown("---")
    st.markdown("#### Pitcher Arsenal vs Volpe")
    c_l,c_r = st.columns(2)

    with c_l:
        ps = df_e.groupby("pitch_name").agg(
            pitches =("is_swing","count"), swings=("is_swing","sum"),
            whiffs  =("is_whiff","sum"),   avg_velo=("release_speed","mean"),
        ).reset_index()
        ps["whiff_pct"] = (ps["whiffs"]/ps["swings"].clip(lower=1)*100).round(1)
        ps["usage_pct"] = (ps["pitches"]/ps["pitches"].sum()*100).round(1)
        ps = ps[ps["pitches"]>=3].sort_values("whiff_pct",ascending=False)

        st.markdown("##### Whiff% by Pitch Type")
        fig_ar = go.Figure(go.Bar(
            x=ps["pitch_name"],y=ps["whiff_pct"],
            marker=dict(color=ps["whiff_pct"],
                        colorscale=[[0,"#003087"],[0.5,"#1F6FEB"],[1.0,"#E4002B"]],
                        showscale=False),
            text=ps["whiff_pct"].apply(lambda x:f"{x:.0f}%"),
            textposition="outside",textfont=dict(color="#E6EDF3"),
            customdata=ps[["usage_pct","avg_velo","pitches"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>Whiff: %{y:.1f}%<br>"
                "Usage: %{customdata[0]:.1f}%<br>"
                "Avg Velo: %{customdata[1]:.1f} mph<br>"
                "Pitches: %{customdata[2]}<extra></extra>"
            )
        ))
        fig_ar.add_hline(y=24,line_dash="dot",line_color="#F0A500",opacity=0.6,
                         annotation_text="MLB avg 24%",annotation_font_color="#F0A500")
        fig_ar.update_layout(
            paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3",size=12),
            margin=dict(l=40,r=20,t=20,b=40),height=320,
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(title="Whiff%",ticksuffix="%",gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_ar,use_container_width=True)

    with c_r:
        st.markdown("##### Outcomes by Count")
        cr = df_p.groupby("count_label")["events"].value_counts().unstack(fill_value=0)
        pos = ["single","double","triple","home_run","walk"]
        neg = ["strikeout","field_out","force_out","grounded_into_double_play"]
        cs  = pd.DataFrame({
            "Positive": cr[[c for c in pos if c in cr.columns]].sum(axis=1),
            "Negative": cr[[c for c in neg if c in cr.columns]].sum(axis=1),
        }).reset_index()
        fig_cs = go.Figure()
        fig_cs.add_trace(go.Bar(x=cs["count_label"],y=cs["Positive"],
                                name="Positive (H/BB)",marker=dict(color="#2EA043")))
        fig_cs.add_trace(go.Bar(x=cs["count_label"],y=cs["Negative"],
                                name="Negative (K/Out)",marker=dict(color="#E4002B")))
        fig_cs.update_layout(
            paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3",size=12),barmode="group",
            margin=dict(l=40,r=20,t=20,b=40),height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(title="Events",gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_cs,use_container_width=True)

    st.markdown("---")
    st.markdown("#### Performance Trend by Season")
    ss = df_e.groupby("season").agg(
        pitches=("is_whiff","count"),whiffs=("is_whiff","sum"),
        swings=("is_swing","sum"),hits=("is_hit","sum"),
    ).reset_index()
    ss["whiff_pct"] = (ss["whiffs"]/ss["swings"].clip(lower=1)*100).round(1)
    ss["hit_pct"]   = (ss["hits"]/ss["pitches"].clip(lower=1)*100).round(1)

    fig_tr = make_subplots(specs=[[{"secondary_y":True}]])
    fig_tr.add_trace(go.Bar(
        x=ss["season"],y=ss["whiff_pct"],name="Whiff%",
        marker=dict(color="#E4002B",opacity=0.8),
        text=ss["whiff_pct"].apply(lambda x:f"{x:.1f}%"),
        textposition="outside",textfont=dict(color="#E6EDF3")
    ), secondary_y=False)
    fig_tr.add_trace(go.Scatter(
        x=ss["season"],y=ss["hit_pct"],name="Hit%",
        mode="lines+markers",line=dict(color="#2EA043",width=2),marker=dict(size=8)
    ), secondary_y=True)
    fig_tr.update_layout(
        paper_bgcolor="#0D1117",plot_bgcolor="#161B22",
        font=dict(color="#E6EDF3",size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50,r=50,t=20,b=40),height=300,
    )
    fig_tr.update_yaxes(title_text="Whiff%",ticksuffix="%",
                        gridcolor="rgba(255,255,255,0.05)",secondary_y=False)
    fig_tr.update_yaxes(title_text="Hit%",ticksuffix="%",
                        showgrid=False,secondary_y=True)
    st.plotly_chart(fig_tr,use_container_width=True)

    st.markdown("---")
    st.markdown("#### Scouting Summary")
    top_pitch = ps.iloc[0]["pitch_name"] if not ps.empty else "N/A"
    top_whiff = ps.iloc[0]["whiff_pct"]  if not ps.empty else 0

    st.markdown(f"""<div class='insight-box'>
        <b>Volpe vs {name} — Executive Summary</b><br><br>
        In <b>{total_pitches}</b> historical pitches, {name} has generated a
        <b>{whiff_rate:.1f}% whiff rate</b> against Volpe
        ({'above' if whiff_rate>24 else 'below'} the MLB average of 24%).<br><br>
        The most effective pitch has been the <b>{top_pitch}</b>
        ({top_whiff:.1f}% whiff rate).
        Volpe has recorded <b>{k_total} strikeouts</b> and <b>{bb_total} walks</b>
        in this matchup, with <b>{hr_total} home run(s)</b>.<br><br>
        <span style='color:#8B949E;font-size:12px'>
        Model Accuracy: RF {models['rf_acc']*100:.1f}% · XGBoost {models['xgb_acc']*100:.1f}%
        · Best: {models['best_name']} · Data: 2023-2025 MLB regular season + playoffs
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Full Arsenal Statistics")
    st.dataframe(
        ps[["pitch_name","pitches","usage_pct","avg_velo","swings","whiffs","whiff_pct"]]
        .rename(columns={
            "pitch_name":"Pitch","pitches":"Total","usage_pct":"Usage%",
            "avg_velo":"Avg Velo","swings":"Swings","whiffs":"Whiffs","whiff_pct":"Whiff%"
        }).round(1),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.spinner("Loading Statcast data for Anthony Volpe (2023-2025)..."):
        df = load_volpe_data()

    if df.empty:
        st.error("Could not load data. Check your internet connection.")
        st.stop()

    df = engineer_features(df)

    with st.spinner("Loading pitcher names from Chadwick register..."):
        pitcher_ids      = df["pitcher"].dropna().unique().tolist()
        pitcher_name_map = build_pitcher_name_map(pitcher_ids)

    with st.spinner("Training prediction models..."):
        models = train_models(df)

    filters = render_sidebar(df, pitcher_name_map)

    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>
        <div>
            <div style='font-size:20px;font-weight:700;color:#fff'>
                Anthony Volpe — Matchup & Prediction System
            </div>
            <div style='font-size:12px;color:#8B949E'>
                New York Yankees Analytics · 2023-2025 · {len(df):,} pitches loaded
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "", ["Matchup Explorer","Pitch Prediction","Front Office Report"],
        horizontal=True, label_visibility="collapsed"
    )
    st.markdown("---")

    if page == "Matchup Explorer":
        page_matchup(df, filters)
    elif page == "Pitch Prediction":
        page_prediction(df, models, filters)
    else:
        page_front_office(df, models, filters)


if __name__ == "__main__":
    main()
