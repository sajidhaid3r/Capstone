"""
Smart Kitchen Assistant — single-file build
---------------------------------------------
Camera + voice powered meal-engineering app built for the MirAI Capstone.
Combines problem statements #11 (Macro Engine), #14 (Fridge-to-Feast), and a
flashcard-output element inspired by #8 (Voice-Notes to Flashcards).

Tech stack (strictly per rubric): Streamlit, Gemini API (Vision + Audio),
Pandas, native Streamlit components only.

SDK note: built on `google-genai` (the current, GA, actively-maintained
Gemini SDK). The older `google-generativeai` package is deprecated and its
2.0-series models have been shut down — do not revert to it.
"""

import os
import json
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
# SECTION 1 — GEMINI CLIENT & FALLBACK ROUTER
# =============================================================================

MODEL_NAME = "gemini-2.5-flash"

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

# --- Response schemas ---
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
    """Raised when a Gemini call fails cleanly across all fallback attempts."""
    pass


# Load keys matching GEMINI_API_KEY_1 through GEMINI_API_KEY_6
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
]
# Fallback to single key if only GEMINI_API_KEY is present
if not any(API_KEYS) and os.getenv("GEMINI_API_KEY"):
    API_KEYS = [os.getenv("GEMINI_API_KEY")]

API_KEYS = [k.strip() for k in API_KEYS if k and k.strip()]


def call_gemini_with_fallback(contents, config=None):
    """
    Iterates through available API keys until a successful API call is made.
    Automatically skips keys that hit 429 RESOURCE_EXHAUSTED limits.
    """
    if not API_KEYS:
        raise GeminiCallError("No Gemini API keys found. Check your .env configuration.")

    from google.genai.errors import APIError
    last_exception = None

    for index, key in enumerate(API_KEYS):
        try:
            client = genai.Client(api_key=key)
            response = client.interactions.create(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            return response
        except APIError as e:
            last_exception = e
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                st.toast(f"⚠️ Key #{index + 1} quota exhausted. Retrying with Key #{index + 2}...", icon="🔄")
                continue
            else:
                raise GeminiCallError(f"API Error: {e}") from e
        except Exception as e:
            last_exception = e
            continue

    raise GeminiCallError(f"All {len(API_KEYS)} API key(s) failed or exceeded quota: {last_exception}")


def detect_ingredients_from_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> list[str]:
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        prompt = (
            "List every distinct food ingredient visible in this fridge photo. "
            "Return lowercase ingredient names only."
        )
        response = call_gemini_with_fallback(
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=INGREDIENT_LIST_SCHEMA,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise GeminiCallError(f"Vision detection failed: {e}") from e


def parse_voice_input(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
    MIN_AUDIO_BYTES = 2000
    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        raise GeminiCallError("Recording was too short or empty — try again and speak for at least 2-3 seconds.")

    try:
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
        response = call_gemini_with_fallback(
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=VOICE_PARSE_SCHEMA,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise GeminiCallError(f"Voice parsing failed: {e}") from e


def generate_recipe(ingredients: list[str], protein_target_g: float,
                     budget_inr: float, meal_type: str) -> dict:
    try:
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
        response = call_gemini_with_fallback(
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
    except GeminiCallError:
        raise
    except Exception as e:
        raise GeminiCallError(f"Recipe generation failed: {e}") from e


# =============================================================================
# SECTION 2 — DATA PIPELINE
# =============================================================================

def ingredients_to_dataframe(ingredients: list[str]) -> pd.DataFrame:
    try:
        clean = [str(i).strip() for i in (ingredients or []) if str(i).strip()]
        if not clean:
            clean = [""]
        return pd.DataFrame({"ingredient": clean})
    except Exception:
        return pd.DataFrame({"ingredient": [""]})


def dataframe_to_ingredients(df: pd.DataFrame) -> list[str]:
    try:
        if df is None or "ingredient" not in df.columns:
            return []
        series = df["ingredient"].dropna().astype(str).str.strip()
        series = series[series != ""]
        return list(dict.fromkeys(series.tolist()))
    except Exception:
        return []


def macros_to_dataframe(macros: dict) -> pd.DataFrame:
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
    try:
        if not history:
            return pd.DataFrame(columns=["meal_number", "protein_g"]).set_index("meal_number")
        df = pd.DataFrame(history)
        return df.set_index("meal_number")
    except Exception:
        return pd.DataFrame(columns=["meal_number", "protein_g"]).set_index("meal_number")


# =============================================================================
# SECTION 3 — FLASHCARD UI
# =============================================================================

def render_recipe_flashcards(recipe: dict, protein_target: float, budget: float, key_suffix: str = "0"):
    if not recipe:
        st.error("The model didn't return a parseable recipe. Try again.")
        return

    macros = recipe.get("macros", {})
    achieved_protein = macros.get("protein_g", 0)
    est_cost = recipe.get("estimated_cost_inr", 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Protein", f"{achieved_protein} g", delta=f"{achieved_protein - protein_target:+.0f} g vs target")
    k2.metric("Calories", f"{macros.get('calories', 0)} kcal")
    k3.metric("Est. Cost", f"₹{est_cost}", delta=f"₹{budget - est_cost:+.0f} vs budget")
    k4.metric("Carbs / Fat", f"{macros.get('carbs_g', 0)}g / {macros.get('fat_g', 0)}g")

    st.bar_chart(macros_to_dataframe(macros), use_container_width=True)
    render_budget_progress(est_cost, budget)

    st.divider()

    with st.container():
        st.subheader(f'🍳 {recipe.get("recipe_name", "Your Recipe")}')
        ing_df = pd.DataFrame(recipe.get("ingredients_used", []))
        if not ing_df.empty:
            st.data_editor(ing_df, use_container_width=True, hide_index=True, key=f"ingredients_editor_{key_suffix}")

    with st.expander("👨‍🍳 Step-by-step method", expanded=True):
        for i, step in enumerate(recipe.get("steps", []), start=1):
            st.markdown(f"**{i}.** {step}")

    swaps = recipe.get("missing_or_optional_swaps", [])
    if swaps:
        with st.expander("🔁 Optional swaps"):
            for s in swaps:
                st.markdown(f"- **{s.get('original')} → {s.get('swap')}** — {s.get('reason')}")

    timing = recipe.get("timing_and_digestion") or {}
    if timing:
        with st.container():
            st.subheader("🕐 Timing & Digestion")
            st.write(f"**Best time:** {timing.get('best_time', '—')}")
            st.write(f"**Why it works:** {timing.get('why_it_works', '—')}")
            st.info(f"💡 **Pro tip:** {timing.get('pro_tip', '—')}")


def render_budget_progress(estimated_cost: float, budget: float):
    if budget <= 0:
        return
    fraction = min(estimated_cost / budget, 1.0)
    label = f"₹{estimated_cost} of ₹{budget} budget used"
    if estimated_cost > budget:
        label += " ⚠️ over budget"
    st.progress(fraction, text=label)


def render_history_trend(history: list[dict]):
    if len(history) < 2:
        return
    st.subheader("📈 Protein trend this session")
    st.line_chart(history_to_dataframe(history)["protein_g"])


# =============================================================================
# SECTION 4 — MAIN APP
# =============================================================================

st.set_page_config(
    page_title="Smart Kitchen Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        color: #10B981 !important;
    }
    div.element-container:has(div.stAlert) {
        border-radius: 10px !important;
    }
    .chefcoach-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
    }
    @media (prefers-color-scheme: dark) {
        .chefcoach-card {
            background-color: #1e293b;
            border-color: #334155;
        }
        [data-testid="stMetricValue"] {
            color: #34D399 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# State initialization
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "recipe_data" not in st.session_state:
    st.session_state.recipe_data = None
if "history" not in st.session_state:
    st.session_state.history = []
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "protein_target" not in st.session_state:
    st.session_state.protein_target = 60
if "budget" not in st.session_state:
    st.session_state.budget = 150
if "meal_type" not in st.session_state:
    st.session_state.meal_type = "Breakfast"

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "Chat 1"

st.session_state.api_configured = len(API_KEYS) > 0

if st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.current_chat_id] = {
        "ingredients": [],
        "recipe_data": None,
        "history": [],
        "show_camera": False,
        "show_uploader": False,
        "chat_log": [],
    }

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.title("🥗 Smart Kitchen")
st.sidebar.caption("Camera + voice powered meal engineering.")

st.sidebar.divider()

if st.sidebar.button("➕  Create New Chat", use_container_width=True, type="primary"):
    st.session_state.chat_sessions[st.session_state.current_chat_id] = {
        "ingredients": st.session_state.ingredients,
        "recipe_data": st.session_state.recipe_data,
        "history": st.session_state.history,
        "show_camera": st.session_state.show_camera,
        "show_uploader": st.session_state.show_uploader,
        "chat_log": st.session_state.chat_log,
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
    }
    st.session_state.ingredients = []
    st.session_state.recipe_data = None
    st.session_state.history = []
    st.session_state.show_camera = False
    st.session_state.show_uploader = False
    st.session_state.chat_log = []
    st.rerun()

st.sidebar.subheader("Recent Chats")
chat_keys = list(reversed(list(st.session_state.chat_sessions.keys())))
with st.sidebar.expander("▼  All sessions", expanded=True):
    for chat_id in chat_keys:
        is_active = (chat_id == st.session_state.current_chat_id)
        session_data = st.session_state.chat_sessions[chat_id]
        recipe = session_data.get("recipe_data")
        ings = session_data.get("ingredients", [])
        if recipe and recipe.get("recipe_name"):
            preview = recipe["recipe_name"]
        elif ings:
            preview = ", ".join(ings[:3]) + ("..." if len(ings) > 3 else "")
        else:
            preview = "New session"
        label = f"{'🟢 ' if is_active else ''}{chat_id}"
        if st.button(label, key=f"chat_btn_{chat_id}", use_container_width=True, help=preview):
            if not is_active:
                st.session_state.chat_sessions[st.session_state.current_chat_id] = {
                    "ingredients": st.session_state.ingredients,
                    "recipe_data": st.session_state.recipe_data,
                    "history": st.session_state.history,
                    "show_camera": st.session_state.show_camera,
                    "show_uploader": st.session_state.show_uploader,
                    "chat_log": st.session_state.chat_log,
                }
                loaded = st.session_state.chat_sessions[chat_id]
                st.session_state.current_chat_id = chat_id
                st.session_state.ingredients = loaded.get("ingredients", [])
                st.session_state.recipe_data = loaded.get("recipe_data", None)
                st.session_state.history = loaded.get("history", [])
                st.session_state.show_camera = loaded.get("show_camera", False)
                st.session_state.show_uploader = loaded.get("show_uploader", False)
                st.session_state.chat_log = loaded.get("chat_log", [])
                st.rerun()

st.sidebar.divider()

with st.sidebar.form("target_form"):
    st.subheader("Today's targets")
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
    st.sidebar.warning("⚠️ Add GEMINI_API_KEY_1 to .env to enable AI features.")

with st.sidebar.container():
    s1, s2 = st.columns(2)
    s1.metric("Protein", f"{st.session_state.protein_target} g")
    s2.metric("Budget", f"₹{st.session_state.budget}")
    st.caption(f"Meal: {st.session_state.meal_type}")

col_title, col_session = st.columns([3, 1])
with col_title:
    st.title("What’s Brewing on the Cold Shelf?")
with col_session:
    st.caption(f"🟢 Active Session: {st.session_state.current_chat_id}")

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

tab_camera, tab_voice, tab_manual = st.tabs(["📷 Scan Fridge", "🎤 Speak It", "✏️ Type Manually"])

with tab_camera:
    st.caption("Take a photo or upload an image of your fridge or pantry shelf. Gemini Vision will identify what's usable.")
    if not st.session_state.show_camera and not st.session_state.show_uploader:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Capture Picture", use_container_width=True):
                st.session_state.show_camera = True
        with col2:
            if st.button("Upload Picture", use_container_width=True):
                st.session_state.show_uploader = True
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
                        st.session_state.chat_log.append({"role": "user", "type": "text", "content": "📷 Captured a fridge photo"})
                        st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": detected})
                        st.success(f"Detected: {', '.join(detected) if detected else 'nothing recognizable — try a clearer photo'}")
                    except GeminiCallError as e:
                        st.error(f"Couldn't analyze that photo: {e}")
        if st.button("Close Camera", use_container_width=True):
            st.session_state.show_camera = False
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
                        st.session_state.chat_log.append({"role": "user", "type": "text", "content": f"📤 Uploaded photo: `{uploaded_file.name}`"})
                        st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": detected})
                        st.success(f"Detected: {', '.join(detected) if detected else 'nothing recognizable — try a clearer photo'}")
                    except GeminiCallError as e:
                        st.error(f"Couldn't analyze that photo: {e}")
        if st.button("Close Uploader", use_container_width=True):
            st.session_state.show_uploader = False

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
                        st.session_state.chat_log.append({"role": "user", "type": "text", "content": "🎤 Spoke ingredient list"})
                        st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": st.session_state.ingredients})
                        st.success(f"Heard: {', '.join(st.session_state.ingredients)}")
                    else:
                        st.warning("Didn't catch any ingredients — try speaking more clearly or closer to the mic.")
                except GeminiCallError as e:
                    st.error(f"Couldn't process that recording: {e}")
    elif audio is not None and not st.session_state.api_configured:
        st.warning("Add your GEMINI_API_KEY_1 to .env to enable voice parsing.")

with tab_manual:
    st.caption("Prefer typing? Enter ingredients comma-separated.")
    manual_text = st.text_input("Ingredients", placeholder="eggs, spinach, paneer")
    if st.button("Use this list", use_container_width=True):
        st.session_state.ingredients = [i.strip() for i in manual_text.split(",") if i.strip()]
        if st.session_state.ingredients:
            st.session_state.chat_log.append({"role": "user", "type": "text", "content": f"✏️ Typed ingredients: {manual_text}"})
            st.session_state.chat_log.append({"role": "assistant", "type": "ingredients", "content": st.session_state.ingredients})

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
