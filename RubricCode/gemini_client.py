"""
gemini_client.py — Gemini API integration layer for Smart Kitchen Assistant.

Owns everything related to talking to Gemini: the system prompt/persona,
the response schemas (native JSON mode), the multi-key rotating fallback
engine, and the three call types (vision, audio, text). No Streamlit UI
rendering lives here — this module is UI-agnostic except for the small
amount of st.session_state / st.secrets / st.toast it needs to track the
active key and surface a rotation notice.
"""

import os
import time
import json

import streamlit as st
from google import genai
from google.genai import types


# =============================================================================
# ENV LOADING
# =============================================================================

def load_env_file():
    """Read KEY=VALUE pairs from a .env file next to the project root into os.environ.

    A minimal loader with no third-party dependency (no python-dotenv) —
    intentional, since it keeps the dependency list to exactly three
    packages (streamlit, google-genai, pandas), all of which are already
    required for the app's core features.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


# =============================================================================
# PERSONA / SYSTEM PROMPT
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


# =============================================================================
# RESPONSE SCHEMAS (native JSON mode — Gemini enforces these directly,
# no manual markdown-fence stripping needed)
# =============================================================================

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


# Backoff pause applied between key rotations on a retriable error. Kept
# deliberately short: Streamlit already executes synchronously per rerun,
# so this is a bounded, intentional pause to let a transient 503 clear —
# not an accidental blocking call. 0.4s x up to 5 rotations (~2s worst
# case) is a fair trade against failing the whole request outright.
RETRY_BACKOFF_SECONDS = 0.4


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
                    time.sleep(RETRY_BACKOFF_SECONDS)  # brief pause lets transient 503s clear before retry
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


def resolve_api_key():
    """
    Check every configured secret source (single key or indexed 1..6) and
    configure the client with the first one found.
    Returns True if a usable key was found, False otherwise.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None
    if api_key:
        configure_gemini(api_key)
        return True

    for idx in range(1, 7):
        val = os.getenv(f"GEMINI_API_KEY_{idx}")
        if not val:
            try:
                val = st.secrets.get(f"GEMINI_API_KEY_{idx}", None)
            except Exception:
                val = None
        if val:
            configure_gemini(val)
            return True

    return False
