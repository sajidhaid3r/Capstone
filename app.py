"""
Smart Kitchen Assistant - CodeChef
---------------------------------------------
Camera + voice powered meal-engineering app built for the MirAI Capstone.
Combines problem statements #11 (Macro Engine), #14 (Fridge-to-Feast), and a
flashcard-output element inspired by #8 (Voice-Notes to Flashcards).

Tech stack (strictly per rubric): Streamlit, Gemini API (Vision + Audio),
Pandas, native Streamlit components only.
"""

import os
import json
import time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ---- Minimal .env loader (no third-party dependency) ----
def _load_env_file():
    """Read KEY=VALUE pairs from a .env file next to this script into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env_file()


# =============================================================================
# SECTION 1 — GEMINI CLIENT (prompts, schemas, API calls)
# =============================================================================

MODEL_NAME = "gemini-3.5-flash"  # confirmed available for new API keys (2026-08)

SYSTEM_PROMPT = """You are ChefCoach, an expert sports nutritionist and chef AI
embedded inside the Smart Kitchen Assistant app. You are not a generic chatbot —
you are a tailored meal-engineering engine for a specific user with a specific
protein target and budget.

Rules you always follow:
1. Only use ingredients the user actually has available. Never invent ingredients
   that weren't detected or spoken by the user, unless suggesting an optional swap.
2. Hit the user's stated protein target as closely as possible; be honest in the
   macro estimate rather than inflating numbers to look better.
3. Respect the user's budget ceiling.
4. Write timing and digestion guidance in the tone of a sports nutritionist:
   general, educational, evidence-informed language (e.g. "commonly recommended",
   "many nutrition coaches suggest") — never absolute medical claims.

Example of your voice (for tone calibration only — do not reuse this content):
User context: ingredients=[eggs, spinach], protein_target_g=40, budget_inr=100
Your style: "Recipe Name: Spinach & Egg Power Scramble. Why it works: eggs are a
complete protein and pair well with spinach's iron for absorption. Best time:
30-45 min post-workout, when many nutrition coaches suggest fast-digesting
protein." — direct, specific, never vague or generic-chatbot-sounding.
"""

# --- Response schemas (native JSON mode — Gemini enforces these directly,
# no manual markdown-fence stripping needed) --------------------------------

INGREDIENT_LIST_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(type=types.Type.STRING),
)

VOICE_PARSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "ingredients": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
        "protein_target_g": types.Schema(type=types.Type.NUMBER, nullable=True),
    },
    required=["ingredients"],
)

RECIPE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "recipe_name": types.Schema(type=types.Type.STRING),
        "ingredients_used": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "name": types.Schema(type=types.Type.STRING),
                    "quantity": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        "missing_or_optional_swaps": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "original": types.Schema(type=types.Type.STRING),
                    "swap": types.Schema(type=types.Type.STRING),
                    "reason": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        "steps": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
        "macros": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "protein_g": types.Schema(type=types.Type.NUMBER),
                "carbs_g": types.Schema(type=types.Type.NUMBER),
                "fat_g": types.Schema(type=types.Type.NUMBER),
                "calories": types.Schema(type=types.Type.NUMBER),
            },
        ),
        "estimated_cost_inr": types.Schema(type=types.Type.NUMBER),
        "timing_and_digestion": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "best_time": types.Schema(type=types.Type.STRING),
                "why_it_works": types.Schema(type=types.Type.STRING),
                "pro_tip": types.Schema(type=types.Type.STRING),
            },
        ),
    },
    required=["recipe_name", "ingredients_used", "steps", "macros", "estimated_cost_inr"],
)


class GeminiCallError(Exception):
    """Raised when a Gemini call fails cleanly, so the UI layer can show a
    friendly st.error instead of letting an uncaught exception crash the app
    or print a raw traceback to the terminal."""
    pass


_client = None


def configure_gemini(api_key: str):
    """Initialize the google-genai client (from st.secrets['GEMINI_API_KEY'])."""
    global _client
    _client = genai.Client(api_key=api_key)


def _require_client():
    if _client is None:
        raise GeminiCallError("Gemini client not configured — check GEMINI_API_KEY in secrets.")
    return _client


def run_gemini_operation(api_call_fn):
    """
    Executes an API call function. If a key is rate-limited (status code 429 or RESOURCE_EXHAUSTED),
    it automatically switches to the next available API key, shows a toast notification,
    and retries the call.
    """
    # Load keys GEMINI_API_KEY_1 to GEMINI_API_KEY_6 from environment or secrets
    keys = []
    for i in range(1, 7):
        val = os.getenv(f"GEMINI_API_KEY_{i}")
        if not val:
            try:
                val = st.secrets.get(f"GEMINI_API_KEY_{i}", None)
            except Exception:
                val = None
        if val:
            keys.append((i, val))

    # Fallback to standard GEMINI_API_KEY if no indexed keys are found
    if not keys:
        val = os.getenv("GEMINI_API_KEY")
        if not val:
            try:
                val = st.secrets.get("GEMINI_API_KEY", None)
            except Exception:
                val = None
        if val:
            keys.append((1, val))

    if not keys:
        st.toast("Try again within 3 seconds.")
        raise GeminiCallError("Gemini client not configured — check GEMINI_API_KEY in secrets.")

    if "active_key_index" not in st.session_state:
        st.session_state.active_key_index = 0

    start_idx = st.session_state.active_key_index
    num_keys = len(keys)

    for attempt in range(num_keys):
        curr_pos = (start_idx + attempt) % num_keys
        key_num, api_key = keys[curr_pos]

        try:
            client = genai.Client(api_key=api_key)
            return api_call_fn(client)
        except Exception as e:
            err_str = str(e).lower()
            # Retry on quota/rate-limit errors, model-not-available errors,
            # AND transient server-side errors (503 UNAVAILABLE, 500 INTERNAL,
            # 502 BAD_GATEWAY — "model experiencing high demand" is a 503).
            # These aren't strictly "key exhausted", but rotating keys is a
            # reasonable retry strategy for them too since different keys can
            # land on different backend replicas.
            is_retriable_error = (
                "429" in err_str or
                "resource_exhausted" in err_str or
                "quota" in err_str or
                "exhausted" in err_str or
                "404" in err_str or
                "not_found" in err_str or
                "not found" in err_str or
                "no longer available" in err_str or
                "503" in err_str or
                "unavailable" in err_str or
                "500" in err_str or
                "internal" in err_str or
                "502" in err_str or
                "bad_gateway" in err_str or
                "high demand" in err_str or
                "deadline" in err_str or
                "timeout" in err_str
            )

            if is_retriable_error:
                next_pos = (curr_pos + 1) % num_keys
                st.session_state.active_key_index = next_pos
                next_key_num, _ = keys[next_pos]

                # Distinguish the reason in the message so it's honest about
                # what actually happened (quota vs. server overload), while
                # still literally saying TRY AGAIN as requested.
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    reason = f"API Key {key_num} limit reached"
                else:
                    reason = "Gemini is temporarily unavailable"

                if attempt < num_keys - 1:
                    st.toast(f"⚠️ TRY AGAIN — {reason}. Switching to API Key {next_key_num}...")
                    time.sleep(1)  # brief pause helps transient 503s clear before retry
                    continue
                else:
                    # All 6 keys exhausted / all attempts failed.
                    st.toast("⚠️ TRY AGAIN — all API keys failed or Gemini is overloaded.")
                    raise GeminiCallError(
                        "TRY AGAIN — all 6 API keys were tried and Gemini is still unavailable "
                        "(rate-limited or experiencing high demand). Please wait a minute and retry."
                    ) from e
            else:
                # Unexpected error — raise immediately
                raise e


def detect_ingredients_from_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> list[str]:
    """
    Vision path: fridge photo -> list of detected ingredient names.
    Request:  [prompt: str, image bytes]
    Response: JSON array of lowercase ingredient strings (schema-enforced).
    """
    def _call(client):
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        prompt = (
            "List every distinct food ingredient visible in this fridge photo. "
            "Return lowercase ingredient names only."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=INGREDIENT_LIST_SCHEMA,
            ),
        )
        return json.loads(response.text)

    try:
        return run_gemini_operation(_call)
    except GeminiCallError:
        raise
    except Exception as e:
        raise GeminiCallError(f"Vision detection failed: {e}") from e


def parse_voice_input(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
    """
    Audio path: spoken ingredients + protein target -> structured dict.
    Request:  [prompt: str, audio bytes]
    Response: {"ingredients": [str], "protein_target_g": number|null}
    """
    # Defensive check: reject empty/near-empty recordings (accidental tap,
    # stream cut short) before spending an API call on unusable audio.
    MIN_AUDIO_BYTES = 2000  # a fraction-of-a-second clip won't clear this
    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        raise GeminiCallError("Recording was too short or empty — try again and speak for at least 2-3 seconds.")

    def _call(client):
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        prompt = (
            "The user is speaking in any language (Hindi, English, or a mix). "
            "Extract two things: (1) a list of ingredients they have, and (2) their protein target in grams for today. "
            "IMPORTANT: Always return ingredient names in standard English, regardless of what language the user speaks. "
            "Translate common names — for example: 'bhindi' → 'okra', 'aloo' → 'potato', 'gobhi' → 'cauliflower', "
            "'pyaaz' → 'onion', 'tamatar' → 'tomato', 'kaddu' → 'pumpkin', 'matar' → 'green peas', "
            "'parwal' → 'pointed gourd', 'chawal' → 'rice', 'dal' → 'lentils', 'paneer' → 'paneer'. "
            "Use the widely recognized English culinary name for every ingredient. "
            "If a protein target isn't mentioned, return null for that field."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=VOICE_PARSE_SCHEMA,
            ),
        )
        return json.loads(response.text)

    try:
        return run_gemini_operation(_call)
    except GeminiCallError:
        raise
    except Exception as e:
        raise GeminiCallError(f"Voice parsing failed: {e}") from e


def generate_recipe(ingredients: list[str], protein_target_g: float,
                     budget_inr: float, meal_type: str) -> dict:
    """
    Core AI Integration step. Uses f-strings to inject dynamic user context
    into the prompt, and enforces a strict JSON schema (native response_schema,
    not manual parsing) so the UI layer can render flashcards deterministically.

    Request:  prompt (f-string with live ingredients/target/budget/meal_type)
    Response: RECIPE_SCHEMA-shaped JSON — recipe_name, ingredients_used,
              missing_or_optional_swaps, steps, macros, estimated_cost_inr,
              timing_and_digestion
    """
    ingredients_str = ", ".join(ingredients) if ingredients else "no ingredients specified"

    prompt = f"""
The user has these ingredients available: {ingredients_str}.
Today's protein target: {protein_target_g} grams.
Budget ceiling: ₹{budget_inr}.
Meal type: {meal_type}.

Design ONE recipe using primarily what they already have, that gets as close
as possible to the protein target within budget. Suggest at most 2 optional
swaps if something would meaningfully improve the macro fit. Include timing
and digestion guidance appropriate for this meal type.
"""

    def _call(client):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RECIPE_SCHEMA,
            ),
        )
        parsed = json.loads(response.text)
        if not parsed:
            raise GeminiCallError("Model returned an empty recipe — try again.")
        return parsed

    try:
        return run_gemini_operation(_call)
    except GeminiCallError:
        raise
    except Exception as e:
        raise GeminiCallError(f"Recipe generation failed: {e}") from e


# =============================================================================
# SECTION 2 — DATA PIPELINE (Pandas DataFrame helpers, defensive checks)
# =============================================================================

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


# SECTION 3 — FLASHCARD UI (rendering the recipe as native Streamlit cards)
# =============================================================================



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


# =============================================================================
# SECTION 4 — MAIN APP (Streamlit UI orchestration)
# =============================================================================

st.set_page_config(
    page_title="Smart Kitchen Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom Visual Styling Block (Glassmorphism + Animations) ----
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

# ---- Session state initialization (prevents memory loss on rerun) ----
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "recipe_data" not in st.session_state:
    st.session_state.recipe_data = None
if "history" not in st.session_state:
    st.session_state.history = []
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "chat_log" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "type": str, "content": any}
    st.session_state.chat_log = []
# Fix 1: initialize form defaults so protein_target/budget/meal_type are always
# bound, even on reruns where the sidebar form was never submitted.
if "protein_target" not in st.session_state:
    st.session_state.protein_target = 60
if "budget" not in st.session_state:
    st.session_state.budget = 150
if "meal_type" not in st.session_state:
    st.session_state.meal_type = "Breakfast"

# ---- Chat session management (init) ----
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "Chat 1"

# ---- API config ----
try:
    has_key = False
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None
    if api_key:
        configure_gemini(api_key)
        has_key = True
    else:
        # Check if any indexed key (GEMINI_API_KEY_1..6) exists
        for idx in range(1, 7):
            val = os.getenv(f"GEMINI_API_KEY_{idx}")
            if not val:
                try:
                    val = st.secrets.get(f"GEMINI_API_KEY_{idx}", None)
                except Exception:
                    val = None
            if val:
                configure_gemini(val)
                has_key = True
                break
    st.session_state.api_configured = has_key
except Exception:
    st.session_state.api_configured = False

# Ensure current chat entry exists
if st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.current_chat_id] = {
        "ingredients": [],
        "recipe_data": None,
        "history": [],
        "show_camera": False,
        "show_uploader": False,
        "chat_log": [],
        "protein_target": 60,
        "budget": 150,
        "meal_type": "Breakfast",
    }

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.title("🥗 Smart Kitchen")
st.sidebar.caption("Camera + voice powered meal engineering.")

st.sidebar.divider()

# ---- Create New Chat ----
if st.sidebar.button("➕  Create New Chat", use_container_width=True, type="primary"):
    # Save current session into the dict before switching
    st.session_state.chat_sessions[st.session_state.current_chat_id] = {
        "ingredients": st.session_state.ingredients,
        "recipe_data": st.session_state.recipe_data,
        "history": st.session_state.history,
        "show_camera": st.session_state.show_camera,
        "show_uploader": st.session_state.show_uploader,
        "chat_log": st.session_state.chat_log,
        "protein_target": st.session_state.protein_target,
        "budget": st.session_state.budget,
        "meal_type": st.session_state.meal_type,
    }
    new_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.current_chat_id = new_id
    st.session_state.chat_sessions[new_id] = {
        "ingredients": [],
        "recipe_data": None,
        "history": [],
        "show_camera": False,
        "show_uploader": False,
        "chat_log": [],
        "protein_target": 60,
        "budget": 150,
        "meal_type": "Breakfast",
    }
    # Reset active session vars
    st.session_state.ingredients = []
    st.session_state.recipe_data = None
    st.session_state.history = []
    st.session_state.show_camera = False
    st.session_state.show_uploader = False
    st.session_state.chat_log = []
    st.session_state.protein_target = 60
    st.session_state.budget = 150
    st.session_state.meal_type = "Breakfast"
    st.rerun()

# ---- Recent Chats ----
st.sidebar.subheader("Recent Chats")

chat_keys = list(reversed(list(st.session_state.chat_sessions.keys())))
with st.sidebar.expander("▼  All sessions", expanded=True):
    for chat_id in chat_keys:
        is_active = (chat_id == st.session_state.current_chat_id)
        session_data = st.session_state.chat_sessions[chat_id]
        # Build a preview label: show recipe name if available, else ingredient count
        recipe = session_data.get("recipe_data")
        ings = session_data.get("ingredients", [])
        if recipe and recipe.get("recipe_name"):
            preview = recipe["recipe_name"]
        elif ings:
            preview = ", ".join(ings[:3]) + ("..." if len(ings) > 3 else "")
        else:
            preview = "New session"
        label = f"{'🟢 ' if is_active else ''}{chat_id}"
        help_txt = preview
        if st.button(label, key=f"chat_btn_{chat_id}", use_container_width=True, help=help_txt):
            if not is_active:
                # Save current state first
                st.session_state.chat_sessions[st.session_state.current_chat_id] = {
                    "ingredients": st.session_state.ingredients,
                    "recipe_data": st.session_state.recipe_data,
                    "history": st.session_state.history,
                    "show_camera": st.session_state.show_camera,
                    "show_uploader": st.session_state.show_uploader,
                    "chat_log": st.session_state.chat_log,
                    "protein_target": st.session_state.protein_target,
                    "budget": st.session_state.budget,
                    "meal_type": st.session_state.meal_type,
                }
                # Load selected session
                loaded = st.session_state.chat_sessions[chat_id]
                st.session_state.current_chat_id = chat_id
                st.session_state.ingredients = loaded.get("ingredients", [])
                st.session_state.recipe_data = loaded.get("recipe_data", None)
                st.session_state.history = loaded.get("history", [])
                st.session_state.show_camera = loaded.get("show_camera", False)
                st.session_state.show_uploader = loaded.get("show_uploader", False)
                st.session_state.chat_log = loaded.get("chat_log", [])
                st.session_state.protein_target = loaded.get("protein_target", 60)
                st.session_state.budget = loaded.get("budget", 150)
                st.session_state.meal_type = loaded.get("meal_type", "Breakfast")
                st.rerun()

st.sidebar.divider()

# ---- Today's targets form ----
with st.sidebar.form("target_form"):
    st.subheader("Today's targets")
    # Use session-state values as widget defaults so the form always reflects
    # the last saved targets after rerun (fixes UnboundLocalError on first load).
    protein_target = st.slider("Protein target (g)", 10, 200,
                               value=st.session_state.protein_target, step=5)
    budget = st.number_input("Budget ceiling (₹)", min_value=0,
                             value=st.session_state.budget, step=10)
    meal_type = st.selectbox("Meal type",
                             ["Breakfast", "Lunch", "Dinner", "Post-workout snack"],
                             index=["Breakfast", "Lunch", "Dinner", "Post-workout snack"].index(
                                 st.session_state.meal_type))
    submitted_targets = st.form_submit_button("Save targets", use_container_width=True)

if submitted_targets:
    st.session_state.protein_target = protein_target
    st.session_state.budget = budget
    st.session_state.meal_type = meal_type
    st.sidebar.success("Targets saved for this session.")

if not st.session_state.api_configured:
    st.sidebar.warning("⚠️ Add GEMINI_API_KEY to .env to enable AI features.")

with st.sidebar.container():
    s1, s2 = st.columns(2)
    s1.metric("Protein", f"{st.session_state.protein_target} g")
    s2.metric("Budget", f"₹{st.session_state.budget}")
    st.caption(f"Meal: {st.session_state.meal_type}")

# ---- Main area header ----
col_title, col_session = st.columns([3, 1])
with col_title:
    st.title("What’s Brewing on the Cold Shelf?")
with col_session:
    st.caption(f"🟢 Active Session: {st.session_state.current_chat_id}")

# ---- Chat log display: show full conversation history for the current session ----
if st.session_state.chat_log:
    st.markdown("### 💬 Session History")
    for i, entry in enumerate(st.session_state.chat_log):
        role = entry.get("role", "assistant")
        etype = entry.get("type", "text")
        content = entry.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🥗"):
                if etype == "ingredients":
                    st.markdown(f"**🔍 Detected ingredients:** {', '.join(content) if isinstance(content, list) else content}")
                elif etype == "recipe":
                    if isinstance(content, dict):
                        st.markdown(f"**🍳 Generated Recipe:** `{content.get('recipe_name', 'Recipe')}`")
                        macros = content.get('macros', {})
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Protein", f"{macros.get('protein_g', 0)} g")
                        c2.metric("Calories", f"{macros.get('calories', 0)} kcal")
                        c3.metric("Cost", f"₹{content.get('estimated_cost_inr', 0)}")
                        c4.metric("Carbs/Fat", f"{macros.get('carbs_g', 0)}g / {macros.get('fat_g', 0)}g")
                        with st.expander("👨‍🍳 Full recipe details", expanded=False):
                            render_recipe_flashcards(
                                content,
                                entry.get("protein_target", 60),
                                entry.get("budget", 150),
                                key_suffix=f"history_{i}",
                            )
                    else:
                        st.markdown(str(content))
                else:
                    st.markdown(str(content))
    st.divider()

# ---- Input tabs (vision / audio / manual) ----
tab_camera, tab_voice, tab_manual = st.tabs(["📷 Scan Fridge", "🎤 Speak It", "✏️ Type Manually"])

with tab_camera:
    st.caption("Take a photo or upload an image of your fridge or pantry shelf. Gemini Vision will identify what's usable.")
    if not st.session_state.show_camera and not st.session_state.show_uploader:
        col1, col2 = st.columns(2)
        with col1:
            # Fix 2 (corrected): a button click triggers exactly ONE script
            # rerun, and the if/elif chain above was already evaluated
            # BEFORE this line sets the flag — so without an explicit
            # st.rerun() here, the camera/uploader widget doesn't appear
            # until a second, unrelated interaction forces another rerun.
            # That was the exact cause of the "have to click twice" bug.
            # Calling st.rerun() immediately re-executes the script with
            # the flag already set, so the widget appears after one click.
            if st.button("Capture Picture", use_container_width=True):
                st.session_state.show_camera = True
                st.rerun()
        with col2:
            if st.button("Upload Picture", use_container_width=True):
                st.session_state.show_uploader = True
                st.rerun()
    elif st.session_state.show_camera:
        photo = st.camera_input("Fridge photo", key="fridge_cam")
        if photo is not None and st.session_state.api_configured:
            if st.button("Detect ingredients from photo", use_container_width=True):
                with st.spinner("Analyzing photo..."):
                    try:
                        detected = detect_ingredients_from_image(
                            photo.getvalue(), mime_type=photo.type or "image/jpeg"
                        )
                        st.session_state.ingredients = detected
                        # Log to chat history
                        st.session_state.chat_log.append({"role": "user", "type": "text", "content": "📷 Captured a fridge photo"})
                        st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": detected})
                        st.success(f"Detected: {', '.join(detected) if detected else 'nothing recognizable — try a clearer photo'}")
                    except GeminiCallError as e:
                        st.error(f"Couldn't analyze that photo: {e}")
        if st.button("Close Camera", use_container_width=True):
            st.session_state.show_camera = False
            st.rerun()
    elif st.session_state.show_uploader:
        uploaded_file = st.file_uploader("Choose a photo of your fridge", type=["jpg", "jpeg", "png"], key="fridge_upload")
        if uploaded_file is not None and st.session_state.api_configured:
            st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
            if st.button("Detect ingredients from uploaded photo", use_container_width=True):
                with st.spinner("Analyzing uploaded photo..."):
                    try:
                        mime = uploaded_file.type if uploaded_file.type else "image/jpeg"
                        detected = detect_ingredients_from_image(
                            uploaded_file.getvalue(), mime_type=mime
                        )
                        st.session_state.ingredients = detected
                        # Log to chat history
                        st.session_state.chat_log.append({"role": "user", "type": "text", "content": f"📤 Uploaded photo: `{uploaded_file.name}`"})
                        st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": detected})
                        st.success(f"Detected: {', '.join(detected) if detected else 'nothing recognizable — try a clearer photo'}")
                    except GeminiCallError as e:
                        st.error(f"Couldn't analyze that photo: {e}")
        if st.button("Close Uploader", use_container_width=True):
            st.session_state.show_uploader = False
            st.rerun()

with tab_voice:
    st.caption("Say what's in your fridge and your protein target, e.g. "
               '"I have eggs, spinach, paneer, I want 60 grams of protein." — Hindi/English mix works too!')

    voice_method = st.radio(
        "Input method",
        ["🎤 Record live", "📂 Upload audio file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    audio = None
    audio_mime = "audio/wav"

    if voice_method == "🎤 Record live":
        st.info("If the recorder shows an error, **refresh the page** or use the upload option below.", icon="ℹ️")
        recorded = st.audio_input("🎤 Record your voice note", key="voice_recorder")
        if recorded is not None:
            audio = recorded
            audio_mime = getattr(recorded, "type", None) or "audio/wav"
    else:
        uploaded_audio = st.file_uploader(
            "Upload a voice recording (WAV, MP3, WebM, OGG, M4A)",
            type=["wav", "mp3", "webm", "ogg", "m4a"],
            key="voice_upload",
        )
        if uploaded_audio is not None:
            audio = uploaded_audio
            ext = uploaded_audio.name.rsplit(".", 1)[-1].lower()
            mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "webm": "audio/webm",
                        "ogg": "audio/ogg", "m4a": "audio/mp4"}
            audio_mime = mime_map.get(ext, "audio/wav")
            st.audio(uploaded_audio)

    if audio is not None and st.session_state.api_configured:
        if st.button("Parse voice input", use_container_width=True):
            with st.spinner("Transcribing and extracting..."):
                try:
                    audio_bytes = audio.getvalue()
                    parsed = parse_voice_input(audio_bytes, mime_type=audio_mime)
                    st.session_state.ingredients = parsed.get("ingredients", [])
                    if parsed.get("protein_target_g"):
                        st.session_state.protein_target = parsed["protein_target_g"]
                    if st.session_state.ingredients:
                        # Log to chat history
                        st.session_state.chat_log.append({"role": "user", "type": "text", "content": "🎤 Spoke ingredient list"})
                        st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": st.session_state.ingredients})
                        st.success(f"Heard: {', '.join(st.session_state.ingredients)}")
                    else:
                        st.warning("Didn't catch any ingredients — try speaking more clearly or closer to the mic.")
                except GeminiCallError as e:
                    st.error(f"Couldn't process that recording: {e}")
    elif audio is not None and not st.session_state.api_configured:
        st.warning("Add your GEMINI_API_KEY to .env to enable voice parsing.")

with tab_manual:
    st.caption("Prefer typing? Enter ingredients comma-separated.")
    manual_text = st.text_input("Ingredients", placeholder="eggs, spinach, paneer")
    if st.button("Use this list", use_container_width=True):
        st.session_state.ingredients = [i.strip() for i in manual_text.split(",") if i.strip()]
        if st.session_state.ingredients:
            st.session_state.chat_log.append({"role": "user", "type": "text", "content": f"✏️ Typed ingredients: {manual_text}"})
            st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": st.session_state.ingredients})

# ---- Confirmed ingredients: editable table, generation via st.form only ----
if st.session_state.ingredients:
    st.divider()
    st.subheader("Confirmed ingredients")

    with st.form("recipe_generation_form"):
        ing_df = ingredients_to_dataframe(st.session_state.ingredients)
        edited_df = st.data_editor(ing_df, num_rows="dynamic", use_container_width=True, key="ingredient_editor")
        generate_clicked = st.form_submit_button(
            "🍳 Generate recipe", type="primary", use_container_width=True,
            disabled=not st.session_state.api_configured,
        )

    if generate_clicked:
        st.session_state.ingredients = dataframe_to_ingredients(edited_df)
        if not st.session_state.ingredients:
            st.warning("Add at least one ingredient before generating.")
        else:
            with st.spinner("ChefCoach is engineering your meal..."):
                try:
                    recipe = generate_recipe(
                        ingredients=st.session_state.ingredients,
                        protein_target_g=st.session_state.protein_target,
                        budget_inr=st.session_state.budget,
                        meal_type=st.session_state.meal_type,
                    )
                    st.session_state.recipe_data = recipe
                    st.session_state.history.append({
                        "meal_number": len(st.session_state.history) + 1,
                        "protein_g": recipe.get("macros", {}).get("protein_g", 0),
                    })
                    # Log recipe to chat history
                    st.session_state.chat_log.append({
                        "role": "user",
                        "type": "text",
                        "content": f"🍳 Generate recipe — {st.session_state.meal_type} | {st.session_state.protein_target}g protein | ₹{st.session_state.budget} budget"
                    })
                    st.session_state.chat_log.append({
                        "role": "assistant",
                        "type": "recipe",
                        "content": recipe,
                        "protein_target": st.session_state.protein_target,
                        "budget": st.session_state.budget,
                    })
                except GeminiCallError as e:
                    st.error(f"Recipe generation failed: {e}. Please try again.")

# ---- Output: flashcards ----
if st.session_state.recipe_data:
    st.divider()
    st.subheader("Your recipe")
    render_recipe_flashcards(
        st.session_state.recipe_data,
        st.session_state.protein_target,
        st.session_state.budget,
        key_suffix="current",
    )

render_history_trend(st.session_state.history)

st.divider()
st.caption("Built for the MirAI School of Technology Capstone · Streamlit + Gemini API")
