"""Neon theme for the Streamlit dashboard: CSS + Plotly template.

Two colour roles are kept separate on purpose:

* **Data colours** (``CATEGORICAL``) come from the data-viz skill's validated,
  colourblind-safe palette — used for the actual chart marks so series stay
  legible.
* **Neon chrome** (``NEON_*``) is used only for decoration: animated background,
  card glow, borders, KPI accents. It never encodes data.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- data colours (validated, CVD-safe; do not reorder) --------------------
CATEGORICAL = [
    "#3987e5",  # blue
    "#00a300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#9085e9",  # violet
    "#e66767",  # red
]

# --- neon chrome (decoration only) -----------------------------------------
NEON_CYAN = "#00f0ff"
NEON_MAGENTA = "#ff2bd6"
NEON_PURPLE = "#a855f7"
NEON_LIME = "#39ff14"

SURFACE = "#0d0f1a"
INK = "#e9ecff"
MUTED = "#8b90b5"
GRID = "rgba(139,144,181,0.15)"

# Status colours (reserved — never used for a data series).
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def register_plotly_template() -> str:
    """Register and return the name of a dark neon Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=CATEGORICAL,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family='system-ui, "Segoe UI", sans-serif', size=13),
        title=dict(font=dict(color=INK, size=16)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED)),
        legend=dict(font=dict(color=MUTED)),
        margin=dict(l=10, r=10, t=48, b=10),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    pio.templates["sda_neon"] = template
    return "sda_neon"


# --- CSS: animated neon chrome --------------------------------------------
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

:root {{
  --neon-cyan: {NEON_CYAN};
  --neon-magenta: {NEON_MAGENTA};
  --neon-purple: {NEON_PURPLE};
  --surface: {SURFACE};
  --ink: {INK};
  --muted: {MUTED};
}}

/* Animated aurora background */
.stApp {{
  background: radial-gradient(1200px 600px at 10% -10%, rgba(168,85,247,0.20), transparent 60%),
              radial-gradient(1000px 500px at 100% 0%, rgba(0,240,255,0.16), transparent 55%),
              radial-gradient(900px 700px at 50% 120%, rgba(255,43,214,0.14), transparent 60%),
              {SURFACE};
  background-attachment: fixed;
  animation: auroraShift 18s ease-in-out infinite alternate;
  color: {INK};
}}
@keyframes auroraShift {{
  0%   {{ background-position: 0% 0%, 100% 0%, 50% 100%; }}
  100% {{ background-position: 8% 6%, 92% 4%, 55% 96%; }}
}}

/* App title */
.neon-title {{
  font-family: 'Orbitron', system-ui, sans-serif;
  font-weight: 700;
  font-size: clamp(1.6rem, 3vw, 2.6rem);
  letter-spacing: 2px;
  text-align: center;
  color: #fff;
  text-shadow: 0 0 6px var(--neon-cyan), 0 0 18px var(--neon-magenta);
  animation: titlePulse 3.5s ease-in-out infinite;
  margin: 0.2rem 0 0.1rem;
}}
.neon-sub {{ text-align:center; color: var(--muted); margin-bottom: 1.2rem; letter-spacing:1px; }}
@keyframes titlePulse {{
  0%,100% {{ text-shadow: 0 0 6px var(--neon-cyan), 0 0 18px var(--neon-magenta); }}
  50%     {{ text-shadow: 0 0 12px var(--neon-cyan), 0 0 30px var(--neon-magenta); }}
}}

/* KPI cards — glass + neon glow, staggered entrance */
.kpi-card {{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(0,240,255,0.35);
  border-radius: 16px;
  padding: 18px 20px;
  backdrop-filter: blur(8px);
  box-shadow: 0 0 18px rgba(0,240,255,0.12), inset 0 0 12px rgba(168,85,247,0.06);
  transition: transform .25s cubic-bezier(.2,.8,.2,1), box-shadow .25s;
  animation: cardIn .6s cubic-bezier(.2,.8,.2,1) both;
}}
.kpi-card:hover {{
  transform: translateY(-6px) scale(1.02);
  box-shadow: 0 0 28px rgba(0,240,255,0.35), 0 0 48px rgba(255,43,214,0.18);
}}
@keyframes cardIn {{ from {{ opacity:0; transform: translateY(18px);}} to {{opacity:1; transform:none;}} }}
.kpi-label {{ color: var(--muted); font-size:.78rem; letter-spacing:1.5px; text-transform:uppercase; }}
.kpi-value {{
  font-family: 'Orbitron', sans-serif; font-size: 2.1rem; font-weight:700; color:#fff;
  text-shadow: 0 0 10px var(--neon-cyan); font-variant-numeric: tabular-nums;
}}
.kpi-value.magenta {{ text-shadow: 0 0 10px var(--neon-magenta); }}
.kpi-value.purple  {{ text-shadow: 0 0 10px var(--neon-purple); }}

/* Chart panels */
.block-container {{ padding-top: 1.2rem; }}
[data-testid="stPlotlyChart"], .stDataFrame {{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(168,85,247,0.25);
  border-radius: 16px; padding: 8px;
  box-shadow: 0 0 16px rgba(168,85,247,0.10);
  transition: box-shadow .3s, transform .3s;
}}
[data-testid="stPlotlyChart"]:hover {{
  box-shadow: 0 0 26px rgba(0,240,255,0.22); transform: translateY(-2px);
}}

/* Buttons & interactive widgets — neon motion */
.stButton>button, .stDownloadButton>button {{
  background: linear-gradient(135deg, rgba(0,240,255,0.15), rgba(255,43,214,0.15));
  border: 1px solid var(--neon-cyan); color:#fff; border-radius: 12px;
  letter-spacing:1px; transition: all .25s cubic-bezier(.2,.8,.2,1);
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  transform: translateY(-3px) scale(1.03);
  box-shadow: 0 0 18px var(--neon-cyan), 0 0 30px var(--neon-magenta);
  border-color: var(--neon-magenta);
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: rgba(13,15,26,0.85); backdrop-filter: blur(10px);
  border-right: 1px solid rgba(0,240,255,0.25);
}}
[data-testid="stSidebar"] * {{ color: {INK}; }}

/* Section headers */
h2, h3 {{ color:#fff !important; text-shadow: 0 0 8px rgba(0,240,255,0.35); letter-spacing:1px; }}
</style>
"""


def kpi_card_html(label: str, value: str, glow: str = "cyan", delay: float = 0.0) -> str:
    """Return HTML for one neon KPI card. ``glow`` ∈ {cyan, magenta, purple}."""
    cls = "" if glow == "cyan" else glow
    return (
        f'<div class="kpi-card" style="animation-delay:{delay:.2f}s">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {cls}">{value}</div></div>'
    )
