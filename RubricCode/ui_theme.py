"""
ui_theme.py — custom visual styling (glassmorphism + animations) for
Smart Kitchen Assistant. Kept separate from layout/orchestration code so
the CSS can be iterated on without touching app logic.
"""

import streamlit as st


def inject_custom_css():
    """Inject the app's custom glassmorphism theme. Call once, right after
    st.set_page_config()."""
    st.markdown(
        """
        <style>
        /* ============================================================
           ANIMATED BACKGROUND — slow-moving gradient, glass-friendly
           ============================================================ */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(-45deg, #0a0f1c, #0f1729, #0a1f1a, #0f1729);
            background-size: 400% 400%;
            animation: gradientShift 18s ease infinite;
        }

        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Subtle floating orbs behind content for depth, purely decorative */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            top: -10%;
            left: -10%;
            width: 40vw;
            height: 40vw;
            background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
            border-radius: 50%;
            animation: floatOrb 22s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: fixed;
            bottom: -10%;
            right: -10%;
            width: 35vw;
            height: 35vw;
            background: radial-gradient(circle, rgba(52,211,153,0.10) 0%, transparent 70%);
            border-radius: 50%;
            animation: floatOrb 26s ease-in-out infinite reverse;
            pointer-events: none;
            z-index: 0;
        }
        @keyframes floatOrb {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(5%, 8%) scale(1.1); }
        }

        /* ============================================================
           GLASSMORPHISM CARDS — frosted glass effect
           ============================================================ */
        .chefcoach-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.10);
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
            margin-bottom: 20px;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            animation: fadeSlideIn 0.5s ease-out;
        }
        .chefcoach-card:hover {
            transform: translateY(-3px);
            border-color: rgba(16, 185, 129, 0.35);
            box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.15);
        }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* Glass effect on Streamlit's own containers, forms, and expanders */
        [data-testid="stForm"],
        [data-testid="stExpander"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            transition: border-color 0.3s ease;
        }
        [data-testid="stForm"]:hover,
        [data-testid="stExpander"]:hover {
            border-color: rgba(16, 185, 129, 0.25) !important;
        }

        /* ============================================================
           METRIC CARDS — glass + lift-on-hover + glowing value
           Sized down from the original + made uniform across a row so
           KPI cards don't vary in height/width, and reduced further
           inside the narrow sidebar so values don't truncate to "...".
           ============================================================ */
        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        [data-testid="stHorizontalBlock"] > div {
            display: flex;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 12px 14px;
            min-height: 100px;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 28px rgba(16, 185, 129, 0.18);
        }
        [data-testid="stMetricValue"] {
            font-weight: 800 !important;
            font-size: 1.6rem !important;
            color: #10B981 !important;
            text-shadow: 0 0 18px rgba(16, 185, 129, 0.35);
            transition: text-shadow 0.3s ease;
            white-space: normal !important;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
        }
        [data-testid="stMetric"]:hover [data-testid="stMetricValue"] {
            text-shadow: 0 0 28px rgba(16, 185, 129, 0.55);
        }

        /* Sidebar metrics live in a much narrower column — shrink further
           so "Protein"/"Budget" values render fully instead of truncating. */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            min-height: 70px;
            padding: 8px 10px;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
            text-shadow: 0 0 10px rgba(16, 185, 129, 0.30);
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }

        /* ============================================================
           BUTTONS — glass, glow, shimmer sweep on hover
           ============================================================ */
        .stButton > button, .stFormSubmitButton > button {
            position: relative;
            overflow: hidden;
            background: rgba(16, 185, 129, 0.10);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(16, 185, 129, 0.30);
            border-radius: 10px;
            color: #E6EDF3;
            font-weight: 600;
            transition: all 0.25s ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: rgba(16, 185, 129, 0.20);
            border-color: rgba(16, 185, 129, 0.60);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.30);
            transform: translateY(-2px);
        }
        .stButton > button:active, .stFormSubmitButton > button:active {
            transform: translateY(0);
        }
        .stButton > button::before, .stFormSubmitButton > button::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(120deg, transparent, rgba(255,255,255,0.15), transparent);
            transition: left 0.5s ease;
        }
        .stButton > button:hover::before, .stFormSubmitButton > button:hover::before {
            left: 100%;
        }

        button[kind="primary"] {
            background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(52,211,153,0.15)) !important;
            border: 1px solid rgba(16, 185, 129, 0.5) !important;
        }
        button[kind="primary"]:hover {
            box-shadow: 0 0 28px rgba(16, 185, 129, 0.45) !important;
        }

        /* ============================================================
           TABS — glass pill style with animated active indicator
           ============================================================ */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 8px;
            transition: background 0.25s ease, color 0.25s ease;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: rgba(16, 185, 129, 0.15) !important;
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.20);
        }

        /* ============================================================
           PROGRESS BAR — animated glow fill
           ============================================================ */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #10B981, #34D399) !important;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }

        /* ============================================================
           ALERTS (success/warning/error/info) — glass treatment
           ============================================================ */
        div.element-container:has(div.stAlert) {
            border-radius: 12px !important;
        }
        div[data-testid="stAlert"] {
            backdrop-filter: blur(10px);
            border-radius: 12px !important;
            animation: fadeSlideIn 0.4s ease-out;
        }

        /* ============================================================
           DATA EDITOR / TABLES — subtle glass frame
           ============================================================ */
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* ============================================================
           SIDEBAR — deeper glass panel
           ============================================================ */
        [data-testid="stSidebar"] {
            background: rgba(10, 15, 28, 0.6) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        @media (prefers-color-scheme: dark) {
            .chefcoach-card {
                background: rgba(255, 255, 255, 0.04);
                border-color: rgba(255, 255, 255, 0.10);
            }
            [data-testid="stMetricValue"] {
                color: #34D399 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
