"""
data_pipeline.py — Pandas data pipeline for Smart Kitchen Assistant.

Every place ingredient lists, macros, or session history cross into or out
of a DataFrame goes through here, each wrapped in a defensive try/except
so a malformed AI response degrades to an empty/zeroed frame instead of
crashing the UI layer.
"""

import pandas as pd


def ingredients_to_dataframe(ingredients: list[str]) -> pd.DataFrame:
    """Wrap a raw ingredient list in a DataFrame for st.data_editor.

    Fix 4: always return at least one blank row so the editor stays visible
    even when all rows have been deleted, letting the user add new items.
    """
    try:
        clean = [str(i).strip() for i in (ingredients or []) if str(i).strip()]
        # Keep one empty sentinel row when list is empty so the editor
        # remains interactable rather than disappearing from the UI.
        if not clean:
            clean = [""]
        return pd.DataFrame({"ingredient": clean})
    except Exception:
        return pd.DataFrame({"ingredient": [""]})


def dataframe_to_ingredients(df: pd.DataFrame) -> list[str]:
    """Extract a clean, deduplicated ingredient list back out of an edited DataFrame."""
    try:
        if df is None or "ingredient" not in df.columns:
            return []
        series = df["ingredient"].dropna().astype(str).str.strip()
        series = series[series != ""]
        return list(dict.fromkeys(series.tolist()))  # de-dupe, preserve order
    except Exception:
        return []


def macros_to_dataframe(macros: dict) -> pd.DataFrame:
    """Wrap a macros dict {protein_g, carbs_g, fat_g} into chart-ready DataFrame."""
    try:
        return pd.DataFrame({
            "macro": ["Protein", "Carbs", "Fat"],
            "grams": [
                float(macros.get("protein_g", 0) or 0),
                float(macros.get("carbs_g", 0) or 0),
                float(macros.get("fat_g", 0) or 0),
            ],
        }).set_index("macro")
    except Exception:
        return pd.DataFrame({"macro": ["Protein", "Carbs", "Fat"], "grams": [0, 0, 0]}).set_index("macro")


def history_to_dataframe(history: list[dict]) -> pd.DataFrame:
    """Wrap the session's meal history into a DataFrame for the trend chart."""
    try:
        if not history:
            return pd.DataFrame(columns=["meal_number", "protein_g"]).set_index("meal_number")
        df = pd.DataFrame(history)
        return df.set_index("meal_number")
    except Exception:
        return pd.DataFrame(columns=["meal_number", "protein_g"]).set_index("meal_number")
