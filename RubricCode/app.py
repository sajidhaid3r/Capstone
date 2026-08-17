"""
Smart Kitchen Assistant - CodeChef
---------------------------------------------
Camera + voice powered meal-engineering app built for the MirAI Capstone.
Combines problem statements #11 (Macro Engine), #14 (Fridge-to-Feast), and a
flashcard-output element inspired by #8 (Voice-Notes to Flashcards).

Tech stack (strictly per rubric): Streamlit, Gemini API (Vision + Audio),
Pandas, native Streamlit components only.

This file is the orchestration layer only — it wires together session
state, the sidebar, and the input/output tabs. The actual Gemini calls
live in gemini_client.py, the Pandas transforms live in data_pipeline.py,
the custom CSS lives in ui_theme.py, and the flashcard/chart rendering
lives in ui_components.py.
"""

import streamlit as st

from gemini_client import (
    GeminiCallError,
    load_env_file,
    resolve_api_key,
    detect_ingredients_from_image,
    parse_voice_input,
    generate_recipe,
)
from data_pipeline import ingredients_to_dataframe, dataframe_to_ingredients
from ui_theme import inject_custom_css
from ui_components import render_recipe_flashcards, render_history_trend

load_env_file()


# =============================================================================
# PAGE CONFIG + THEME
# =============================================================================

st.set_page_config(
    page_title="Smart Kitchen Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# =============================================================================
# SESSION STATE INITIALIZATION (prevents memory loss on rerun)
# =============================================================================

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
    st.session_state.api_configured = resolve_api_key()
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
    st.title("What's Brewing on the Cold Shelf?")
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
                        st.error(f"Couldn't analyze that uploaded photo: {e}")
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
