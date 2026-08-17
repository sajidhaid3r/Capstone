"""
ui_components.py — presentation layer for Smart Kitchen Assistant.

Turns a Gemini-generated recipe dict (already validated by RECIPE_SCHEMA in
gemini_client.py) into the actual Streamlit widgets: KPI metric row, macro
bar chart, budget progress bar, ingredient table, step expander, swaps
expander, and the session's protein trend line chart.
"""

import pandas as pd
import streamlit as st

from data_pipeline import macros_to_dataframe, history_to_dataframe


def render_recipe_flashcards(recipe: dict, protein_target: float, budget: float, key_suffix: str = "0"):
    if not recipe:
        st.error("The model didn't return a parseable recipe. Try again.")
        return

    # ---- KPI row: st.metric with deltas ----
    macros = recipe.get("macros", {})
    achieved_protein = macros.get("protein_g", 0)
    est_cost = recipe.get("estimated_cost_inr", 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Protein", f"{achieved_protein} g", delta=f"{achieved_protein - protein_target:+.0f} g vs target")
    k2.metric("Calories", f"{macros.get('calories', 0)} kcal")
    k3.metric("Est. Cost", f"₹{est_cost}", delta=f"₹{budget - est_cost:+.0f} vs budget")
    k4.metric("Carbs / Fat", f"{macros.get('carbs_g', 0)}g / {macros.get('fat_g', 0)}g")

    # ---- Macro bar chart (extra data viz alongside the KPI row) ----
    st.bar_chart(macros_to_dataframe(macros), use_container_width=True)
    render_budget_progress(est_cost, budget)

    st.divider()

    # ---- Card 1: Recipe name + ingredients (editable table) ----
    with st.container():
        st.subheader(f'🍳 {recipe.get("recipe_name", "Your Recipe")}')
        ing_df = pd.DataFrame(recipe.get("ingredients_used", []))
        if not ing_df.empty:
            st.data_editor(ing_df, use_container_width=True, hide_index=True, key=f"ingredients_editor_{key_suffix}")

    # ---- Card 2: Steps (expander) ----
    with st.expander("👨‍🍳 Step-by-step method", expanded=True):
        for i, step in enumerate(recipe.get("steps", []), start=1):
            st.markdown(f"**{i}.** {step}")

    # ---- Card 3: Swaps (expander, only if present) ----
    swaps = recipe.get("missing_or_optional_swaps", [])
    if swaps:
        with st.expander("🔁 Optional swaps"):
            for s in swaps:
                st.markdown(f"- **{s.get('original')} → {s.get('swap')}** — {s.get('reason')}")

    # ---- Card 4: Timing & Digestion guidance ----
    # Fix 3: guard against Gemini returning null for optional schema keys.
    timing = recipe.get("timing_and_digestion") or {}
    if timing:
        with st.container():
            st.subheader("🕐 Timing & Digestion")
            st.write(f"**Best time:** {timing.get('best_time', '—')}")
            st.write(f"**Why it works:** {timing.get('why_it_works', '—')}")
            st.info(f"💡 **Pro tip:** {timing.get('pro_tip', '—')}")


def render_budget_progress(estimated_cost: float, budget: float):
    """Visual progress bar showing spend against budget ceiling."""
    if budget <= 0:
        return
    fraction = min(estimated_cost / budget, 1.0)
    label = f"₹{estimated_cost} of ₹{budget} budget used"
    if estimated_cost > budget:
        label += " ⚠️ over budget"
    st.progress(fraction, text=label)


def render_history_trend(history: list[dict]):
    """Line chart of protein achieved over session history."""
    if len(history) < 2:
        return
    st.subheader("📈 Protein trend this session")
    st.line_chart(history_to_dataframe(history)["protein_g"])
