```
   _____                      __     __ __ _  __       __
  / ___/____ ___  ____ ______/ /_   / //_/(_)/ /______/ /_  ___  ____
  \__ \/ __ `__ \/ __ `/ ___/ __/  / ,<  / // __/ ___/ __ \/ _ \/ __ \
 ___/ / / / / / / /_/ / /  / /_   / /| |/ // /_/ /__/ / / /  __/ / / /
/____/_/ /_/ /_/\__,_/_/   \__/  /_/ |_/_/ \__/\___/_/ /_/\___/_/ /_/

 > Assistant.exe
 > Camera + voice powered meal engineering, tuned to your macros.
```

[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://capstonehaid3r.streamlit.app/)
[![Gemini API](https://img.shields.io/badge/Gemini-8E75B2?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://python.org)

---

## `$ whoami`

**Smart Kitchen Assistant** is a Streamlit + Gemini capstone project built for
the MirAI School of Technology B.Tech Capstone. It combines **three problem
statements from the official 30-project directory** into one coherent
product:

- **Problem #11 — Hyper-Local Macro Engine:** the app takes a daily protein
  target and a strict budget, then generates a meal plan tuned to hit both.
- **Problem #14 — The Fridge-to-Feast Generator:** users snap or upload a
  photo of their fridge; Gemini Vision identifies the ingredients and the
  app outputs a step-by-step recipe.
- **Problem #8 — Voice-Notes to Flashcards:** inspired this project's output
  format — instead of a wall of AI-generated text, the recipe, macros,
  swaps, and timing guidance are rendered as structured flashcards, the
  same way #8 turns a chaotic voice note into a structured study guide.

This is a permitted combination per the capstone's own submission rules:
*"You may pivot or combine ideas, but the core technical requirements
remain the same."*

Feature-wise, this means:
- 📷 **Fridge scanning** (Gemini Vision, camera + file upload) — snap or upload a photo, get ingredients
- 🎤 **Voice input** (Gemini Audio, native `st.audio_input` + file upload) — speak your ingredients and protein goal
- 🍳 **AI-engineered recipes** — generated to hit your macro + budget targets
- 🗂️ **Flashcard output** — recipe, macros, swaps, and dietitian-style timing
  advice, rendered as clean cards instead of a wall of text
- 🔁 **6-key API fallback system** — automatically rotates across multiple
  Gemini API keys on rate-limit or server-overload errors, so a single
  exhausted key never breaks the demo

## `$ live_demo`

🔗 **Live app:** https://capstonehaid3r.streamlit.app/

## `$ features --list`

```
[✓] Camera input + file upload -> Gemini Vision ingredient detection
[✓] Mic recorder + audio upload -> Gemini Audio transcription + target extraction
[✓] st.form + st.form_submit_button -> batched inputs, no redundant API calls
[✓] st.session_state -> ingredients, targets, recipe & history persist
[✓] st.data_editor  -> editable ingredient list + recipe table
[✓] st.metric        -> protein/cost KPIs with deltas
[✓] st.bar_chart / st.line_chart / st.progress -> macro + trend + budget viz
[✓] Native Gemini JSON schema mode -> deterministic, parseable AI output
[✓] 6-key rotating fallback -> survives quota limits AND 503 server overload
```

## `$ architecture`

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full data-flow diagram,
API request/response schemas, and the fallback-system design.

## `$ setup --local`

```bash
git clone <your-repo-url>
cd smart-kitchen-assistant
pip install -r requirements.txt
```

> ⚠️ **Fixed dependency issue:** `requirements.txt` now pins `streamlit>=1.40.0`.
> Earlier this was pinned to `1.33.0`, which does **not** include
> `st.audio_input` (GA'd in Streamlit 1.40) — the app's voice tab would throw
> an `AttributeError` and crash the entire script on that version. If you're
> redeploying, update this in your GitHub repo and Streamlit Cloud will
> rebuild automatically on the next push.

Create a `.env` file in the project root with up to 6 keys for the fallback
system (at minimum, one is required):

```
GEMINI_API_KEY_1=your-first-key
GEMINI_API_KEY_2=your-second-key
GEMINI_API_KEY_3=your-third-key
GEMINI_API_KEY_4=your-fourth-key
GEMINI_API_KEY_5=your-fifth-key
GEMINI_API_KEY_6=your-sixth-key
```

```bash
python -m streamlit run app.py
```

## `$ deploy --cloud`

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo
3. Set `app.py` as the entry point
4. Add your Gemini key(s) under **App settings → Secrets** (same
   `GEMINI_API_KEY_1..6` naming, or a single `GEMINI_API_KEY`)
5. Deploy

**This project is already live:** https://capstonehaid3r.streamlit.app/

## `$ tech_stack`

| Layer | Tool |
|---|---|
| Frontend/UI | Streamlit `>=1.40.0` (native components only, incl. `st.audio_input`) |
| AI | Google Gemini API via `google-genai>=1.19.0` — Vision + Audio + Text, native JSON schema output |
| Data | `pandas>=2.2.0` |
| Env | Zero-dependency `.env` loader written in-house (`gemini_client.load_env_file`) — no `python-dotenv` needed, keeping the dependency list to exactly three packages |
| Resilience | 6-key rotating fallback across quota (429) and server-overload (503) errors |
| Deployment | Streamlit Community Cloud |

## `$ project_structure`

```
Capstone-main/
├── app.py            # orchestration only: page config, session state, sidebar, tabs
├── gemini_client.py   # Gemini persona, JSON schemas, key-rotation fallback, 3 call types
├── data_pipeline.py   # Pandas DataFrame helpers (ingredients, macros, history)
├── ui_theme.py         # custom glassmorphism CSS, isolated from app logic
├── ui_components.py    # flashcard/KPI/chart rendering
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

Split out of what was originally a single ~1,230-line `app.py` so each concern
(AI integration, data transforms, styling, presentation, orchestration) can be
read, tested, and changed independently — see `ARCHITECTURE.md` for the full
module responsibility table.

## `$ author`

Built by `MD SAJID HAIDER` for the MirAI School of Technology Capstone.

[![LinkedIn Post](https://img.shields.io/badge/LinkedIn-Submission%20Post-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)]([<https://www.linkedin.com/posts/md-sajid-haider-20869631a_miraischooloftechnology-streamlit-geminiapi-activity-7495319881899319296-e4v6?utm_source=share&utm_medium=member_android&rcm=ACoAAFDnKqgBO8wkGOAz1vDVxmwd53teqj0jXkI>])

