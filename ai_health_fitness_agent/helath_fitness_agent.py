import streamlit as st
from phi.agent import Agent
from phi.model.ollama import Ollama
from ollama import Client
from config import config
from pathlib import Path
import json, io, time
import matplotlib.pyplot as plt
from typing import Dict, Any
import re
from io import BytesIO
import random

# Must be the first Streamlit command
st.set_page_config(
    page_title="AI Health & Fitness Planner",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Load External Styles --------------------
STYLE_PATH = Path(__file__).parent / "assets" / "style.css"
if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)
else:
    st.warning("Style sheet missing: assets/style.css")

# -------------------- Backend / Client Helpers --------------------

def get_ollama_client():
    """Create an Ollama client (HF Space or local)."""
    try:
        host = config.get_ollama_config()["host"]
        client = Client(host=host)
        if config.is_huggingface_configured() and config.HF_API_KEY:
            # Inject Bearer token for HF Space
            if hasattr(client, "_client") and hasattr(client._client, "headers"):
                client._client.headers["Authorization"] = (
                    f"Bearer {config.HF_API_KEY}"
                )
        return client
    except Exception as e:
        print(f"Client init error: {e}")
        return None


def verify_ollama_auth(client, retries: int = 3, delay: float = 1.2) -> bool:
    for attempt in range(1, retries + 1):
        try:
            client.list()
            if attempt > 1:
                st.info(f"Connected after retry {attempt - 1}.")
            return True
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ["403", "Not authenticated"]):
                st.error("Auth failed (403). Check HF_API_KEY / HF_SPACE_URL.")
                return False
            if attempt < retries:
                st.warning(f"Connection issue (attempt {attempt}/{retries}). Retrying...")
                time.sleep(delay * attempt)
            else:
                st.error(f"Connection error: {e}")
                return False
    return False

# -------------------- UI Components --------------------

def render_stepper(current_step: int):
    labels = ["Profile", "Generate", "Q&A"]
    cols = st.columns(len(labels))
    for i, lbl in enumerate(labels, start=1):
        cls = "step"
        if current_step > i:
            cls += " done"
        elif current_step == i:
            cls += " active"
        cols[i - 1].markdown(
            f"<div class='{cls}'>{i}. {lbl}</div>", unsafe_allow_html=True
        )


def render_hero():
    st.markdown(
        """
        <div class='hero'>
            <h1>AI Health & Fitness Planner</h1>
            <p style='color:#cbd5e1;font-size:.95rem;line-height:1.4;'>
                Create personalized, actionable nutrition and training plans powered by AI.
                Track metrics, iterate fast, stay consistent.
            </p>
            <div style='margin-top:.8rem;display:flex;gap:.65rem;flex-wrap:wrap;'>
                <span class='card tight' style='border:1px solid #334155;color:#93c5fd;font-size:.65rem;font-weight:600;'>Personalized</span>
                <span class='card tight' style='border:1px solid #334155;color:#fbbf24;font-size:.65rem;font-weight:600;'>Evidence Based</span>
                <span class='card tight' style='border:1px solid #334155;color:#34d399;font-size:.65rem;font-weight:600;'>Adaptive</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def compute_stats(weight: float, height: float, activity_level: str):
    bmi = round(weight / ((height / 100) ** 2), 2) if height and weight else 0
    activity_map = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extremely Active": 1.9,
    }
    # Simplified Mifflin-St Jeor approximation (sex-neutral placeholder age=30)
    bmr = 10 * weight + 6.25 * height - 5 * 30
    tdee = int(bmr * activity_map.get(activity_level, 1.2))
    return bmi, tdee


def profile_completion(p: dict) -> int:
    required_keys = ["age", "height", "weight", "activity_level", "dietary_pref", "goal"]
    filled = sum(1 for k in required_keys if p.get(k) not in (None, "", 0))
    return int((filled / len(required_keys)) * 100)

def render_profile_form():  # type: ignore[override]
    st.subheader("Profile")
    st.caption("Provide accurate details. This is guidance only and not medical advice.")
    with st.container():
        c1, c2, c3 = st.columns(3)
        age = c1.number_input(
            "Age",
            10,
            100,
            30,
            help="Used to approximate metabolic rate (simplified)."
        )
        height = c2.number_input(
            "Height (cm)",
            100.0,
            250.0,
            175.0,
            step=0.1,
            help="Standing height in centimeters."
        )
        weight = c3.number_input(
            "Weight (kg)",
            20.0,
            300.0,
            70.0,
            step=0.1,
            help="Current body weight in kilograms."
        )
        c4, c5, c6 = st.columns(3)
        activity_level = c4.selectbox(
            "Activity Level",
            [
                "Sedentary",
                "Lightly Active",
                "Moderately Active",
                "Very Active",
                "Extremely Active",
            ],
            help="General weekly movement & training volume."
        )
        dietary_pref = c5.selectbox(
            "Diet Preference",
            [
                "Vegetarian",
                "Keto",
                "Gluten Free",
                "Low Carb",
                "Dairy Free",
            ],
            help="Main nutritional pattern or restriction."
        )
        goal = c6.selectbox(
            "Primary Goal",
            [
                "Lose Weight",
                "Gain Muscle",
                "Endurance",
                "Stay Fit",
                "Strength Training",
            ],
            help="Top-level training / nutrition objective."
        )
    bmi, tdee = compute_stats(weight, height, activity_level)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Key Indicators")
    completion = profile_completion({
        "age": age,
        "height": height,
        "weight": weight,
        "activity_level": activity_level,
        "dietary_pref": dietary_pref,
        "goal": goal,
    })
    pc_col, metric_col = st.columns([2, 1])
    with pc_col:
        st.progress(completion, text=f"Profile completion: {completion}%")
    with metric_col:
        if completion < 100:
            st.info("Fill all fields for best personalization.")
        else:
            st.success("Profile complete.")
    st.markdown(
        "<div class='metric-grid'>"
        + "".join(
            [
                f"<div class='metric-tile'><h4>BMI</h4><div class='metric-val'>{bmi}</div></div>",
                f"<div class='metric-tile'><h4>Est. TDEE</h4><div class='metric-val'>{tdee}</div></div>",
                f"<div class='metric-tile'><h4>Goal</h4><div class='metric-val'>{goal.split()[0]}</div></div>",
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    return {
        "age": age,
        "height": height,
        "weight": weight,
        "activity_level": activity_level,
        "dietary_pref": dietary_pref,
        "goal": goal,
        "bmi": bmi,
        "tdee": tdee,
        "_completion": completion,
    }


@st.cache_data(show_spinner=True)
def run_agent(agent: Agent, profile_text: str):
    return agent.run(profile_text)

# -------------------- Display Helpers --------------------

def format_plan_display(text: str) -> str:
    if not text:
        return "_No content_"
    # Basic normalization: ensure bullet points render
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    formatted = []
    for ln in lines:
        if ln.startswith(('- ', '* ')):
            formatted.append(ln)
        elif ln[0:2].isdigit() and ln[2] == '.':  # numbered list already
            formatted.append(ln)
        else:
            # Convert plain sentences into bullet points for readability
            if len(ln.split()) > 3:
                formatted.append(f"- {ln}")
            else:
                formatted.append(ln)
    return "\n".join(formatted)


def ensure_plan_edit_state():
    if "edit_diet" not in st.session_state:
        st.session_state.edit_diet = False
    if "edit_fitness" not in st.session_state:
        st.session_state.edit_fitness = False


# --- Copy utility & enhanced editable block ---

def copy_supported_code_block(text: str):
    # Use st.code which shows copy icon in Streamlit >=1.25
    st.code(text if text else "", language="markdown")


def render_editable_plan(title: str, plan_key: str, field: str, edit_flag: str):  # override
    plan = st.session_state.get(plan_key, {})
    st.markdown(f"### {title}")
    ensure_plan_edit_state()
    if not st.session_state.get(edit_flag):
        content = format_plan_display(plan.get(field, ""))
        copy_supported_code_block(content)
        btn_cols = st.columns([1,1,1,2,6])
        if btn_cols[0].button(f"Edit {title}"):
            st.session_state[edit_flag] = True
            st.session_state[f"_buffer_{plan_key}_{field}"] = plan.get(field, "")
            st.rerun()
        if btn_cols[1].button("Copy", key=f"copy_{plan_key}_{field}"):
            st.success("Copied (or use the code block icon).")
        if btn_cols[2].button("Refresh", key=f"refresh_{plan_key}_{field}"):
            if plan_key == 'dietary_plan':
                regenerate_meal_plan(); st.rerun()
            elif plan_key == 'fitness_plan':
                regenerate_fitness_routine(); st.rerun()
        if btn_cols[3].button("Export CSV", key=f"export_{plan_key}_{field}"):
            if plan_key == 'dietary_plan' and st.session_state.get('parsed_meal_plan'):
                import pandas as pd
                csv = pd.DataFrame(st.session_state.parsed_meal_plan).to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="meal_plan.csv", mime="text/csv")
            elif plan_key == 'fitness_plan' and st.session_state.get('parsed_fitness_routine'):
                import pandas as pd
                csv = pd.DataFrame(st.session_state.parsed_fitness_routine).to_csv(index=False)
                st.download_button("Download CSV", csv, file_name="fitness_routine.csv", mime="text/csv")
            else:
                st.info("No structured data to export yet.")
    else:
        buffer_key = f"_buffer_{plan_key}_{field}"
        current_val = st.session_state.get(buffer_key, plan.get(field, ""))
        new_val = st.text_area(f"Edit {title}", value=current_val, height=260)
        action_cols = st.columns([1,1,6])
        if action_cols[0].button("Save", key=f"save_{plan_key}"):
            plan[field] = new_val
            st.session_state[plan_key] = plan
            st.session_state[edit_flag] = False
            st.success(f"{title} updated.")
            st.rerun()
        if action_cols[1].button("Cancel", key=f"cancel_{plan_key}"):
            st.session_state[edit_flag] = False
            st.info("Edit cancelled.")
            st.rerun()

# Disable legacy simple parser & renderer so they do not shadow enhanced version
# (renamed with _legacy suffix earlier). If duplicate still exists below, rename it.
# >>> BEGIN: Deactivate old simple parser if still present <<<
# (Search marker) If a later definition of parse_structured_meal_plan exists it should be renamed.
# >>> END <<<

# --- Dynamic model listing & regeneration helpers ---

def list_model_names() -> list[str]:
    try:
        client = get_ollama_client()
        if not client:
            return []
        raw = client.list()
        if isinstance(raw, dict) and 'models' in raw:
            return [m.get('name','') for m in raw['models'] if m.get('name')]
        if isinstance(raw, list):
            return [m.get('name','') for m in raw if isinstance(m, dict) and m.get('name')]
        return []
    except Exception:
        return []


def _profile_to_text(p: dict) -> str:
    return (
        f"Age: {p['age']}\nWeight: {p['weight']}kg\nHeight: {p['height']}cm\n"
        f"Activity Level: {p['activity_level']}\nDiet: {p['dietary_pref']}\nGoal: {p['goal']}\n"
        f"BMI: {p['bmi']}  TDEE: {p['tdee']}\n"
    )


def regenerate_meal_plan():
    if not st.session_state.get('agents') or not st.session_state.get('profile'):
        return
    try:
        p = st.session_state.profile
        txt = _profile_to_text(p)
        resp = run_with_retry(lambda: st.session_state.agents['diet'].run(txt))
        diet_txt = getattr(resp, 'content', '') or str(resp)
        st.session_state.dietary_plan['meal_plan'] = diet_txt
        st.success('Dietary plan refreshed.')
    except Exception as e:
        st.error(f'Regeneration failed: {e}')


def regenerate_fitness_routine():
    if not st.session_state.get('agents') or not st.session_state.get('profile'):
        return
    try:
        p = st.session_state.profile
        txt = _profile_to_text(p)
        resp = run_with_retry(lambda: st.session_state.agents['fitness'].run(txt))
        fit_txt = getattr(resp, 'content', '') or str(resp)
        st.session_state.fitness_plan['routine'] = fit_txt
        st.success('Fitness routine refreshed.')
    except Exception as e:
        st.error(f'Regeneration failed: {e}')

# -------------------- Main App Flow --------------------

def inject_high_contrast():
    # Adjusted: keep original gradient background; only enhance borders/text for accessibility.
    if st.session_state.get("high_contrast"):
        st.markdown(
            """<style>
            /* High contrast tweaks without removing gradient */
            .metric-tile, .card, .qa-box, .plan-card, .stButton>button {border-color:#f1f5f9 !important;}
            .step {border-color:#f1f5f9 !important;}
            h1,h2,h3,h4,h5,h6, p, span, label, div {text-shadow:0 0 2px rgba(0,0,0,.6);}
            </style>""",
            unsafe_allow_html=True,
        )


def get_cached_agents(llm_model):
    if "agent_cache" not in st.session_state:
        st.session_state.agent_cache = {}
    key = f"agents::{getattr(llm_model,'id','default')}"
    if key not in st.session_state.agent_cache:
        st.session_state.agent_cache[key] = {
            "diet": Agent(
                name="Dietary Expert",
                role="Provides personalized dietary recommendations",
                model=llm_model,
                instructions=[
                    "Design a daily structured meal plan with macro rationale",
                    "Explain succinctly why composition supports the goal",
                    "Keep meals realistic and culturally neutral",
                ],
            ),
            "fitness": Agent(
                name="Fitness Expert",
                role="Provides personalized fitness recommendations",
                model=llm_model,
                instructions=[
                    "Provide phased warm-up, main sets, cool-down",
                    "Include progressive overload suggestion",
                    "Emphasize form & recovery cues",
                ],
            ),
        }
    return st.session_state.agent_cache[key]


def build_macro_chart(tdee: int) -> io.BytesIO:
    if not tdee:
        tdee = 2000
    macros = {"Protein": 0.3, "Carbs": 0.45, "Fats": 0.25}
    cals = {k: int(v * tdee) for k, v in macros.items()}
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.pie(cals.values(), labels=[f"{k}\n{v} cal" for k, v in cals.items()], autopct="%1.0f%%", startangle=90, textprops={"color": "#f1f5f9"})
    ax.set_title("Macro Calorie Split", color="#f1f5f9")
    fig.patch.set_alpha(0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf


def build_plan_markdown(diet: Dict[str, Any], fitness: Dict[str, Any]) -> str:
    return (
        "# Personalized Health & Fitness Plan\n\n"
        "## Dietary Plan\n"
        f"Why it works:\n{diet.get('why_this_plan_works','')}\n\n"
        f"Meals:\n{diet.get('meal_plan','')}\n\n"
        "Important Considerations:\n" + diet.get("important_considerations", "") + "\n\n"
        "## Fitness Plan\n"
        f"Goals:\n{fitness.get('goals','')}\n\n"
        f"Routine:\n{fitness.get('routine','')}\n\n"
        "Tips:\n" + fitness.get("tips", "") + "\n"
    )


PROFILE_CACHE = Path(__file__).parent / "profile_cache.json"

def save_profile(profile: Dict[str, Any]):
    try:
        PROFILE_CACHE.write_text(json.dumps(profile, indent=2))
    except Exception:
        pass

def load_cached_profile():
    if PROFILE_CACHE.exists():
        try:
            return json.loads(PROFILE_CACHE.read_text())
        except Exception:
            return None
    return None

def ensure_edit_buffers():
    if "diet_edit" not in st.session_state:
        st.session_state.diet_edit = ""
    if "fitness_edit" not in st.session_state:
        st.session_state.fitness_edit = ""

def stream_answer(agent: Agent, context: str) -> str:
    # Fallback: generate full then stream out
    resp = agent.run(context)
    content = getattr(resp, "content", str(resp))
    placeholder = st.empty()
    out = ""
    for chunk in content.split(" "):
        out += chunk + " "
        placeholder.markdown(out)
        time.sleep(0.015)
    return content

def run_with_retry(callable_fn, attempts: int = 4, base_delay: float = 1.0, show_first: bool = False):
    """Retry wrapper for transient 503/timeout errors with jitter and clearer messages.
    Params:
        callable_fn: function to execute
        attempts: max attempts
        base_delay: initial backoff seconds
        show_first: if False suppress warning on first failure
    """
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return callable_fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            transient = ("503" in msg) or ("timeout" in msg.lower()) or ("temporarily" in msg.lower())
            if i == attempts or not transient:
                st.error(f"Q&A failed: {msg}")
                raise
            delay = base_delay * (2 ** (i - 1)) * random.uniform(0.85, 1.3)
            if show_first or i > 1:
                st.warning(f"Transient issue (attempt {i}/{attempts}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
    if last_err:
        raise last_err

@st.cache_data(show_spinner=False)
def cached_generation(profile_payload: str, model_id: str):
    """Cache heavy plan generation for identical profile + model."""
    return {"_cached": True}


# -------------------- Main App Flow --------------------

def main():  # type: ignore[override]
    if "ui_step" not in st.session_state:
        st.session_state.ui_step = 1
        st.session_state.profile = None
        st.session_state.dietary_plan = {}
        st.session_state.fitness_plan = {}
        st.session_state.qa_pairs = []
        st.session_state.plans_generated = False

    with st.sidebar:
        st.markdown("### Options")
        available_models = list_model_names()
        fallback_models = [getattr(config, "DEFAULT_MODEL", "llama3.2:1b"), "llama3.1:3b", "llama3.1:latest"]
        model_options = available_models or fallback_models
        default_index = 0
        if getattr(config, 'DEFAULT_MODEL', None) in model_options:
            default_index = model_options.index(getattr(config, 'DEFAULT_MODEL'))
        model_choice = st.selectbox("Model", model_options, index=default_index)
        st.session_state["selected_model"] = model_choice.strip()
        st.checkbox("High Contrast", key="high_contrast", on_change=inject_high_contrast)
        st.checkbox("Auto-load last profile", key="auto_load_profile")
        uploaded = st.file_uploader("Load Profile JSON", type=["json"], label_visibility="collapsed")
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                st.session_state.profile = data
                st.success("Profile loaded")
                st.session_state.ui_step = 2
            except Exception as e:
                st.error(f"Load error: {e}")
        st.caption(
            "Environment: HF Space" if config.is_huggingface_configured() else "Environment: Local Ollama"
        )
        if st.button("Download Profile", disabled=not st.session_state.get("profile")):
            prof = json.dumps(st.session_state.get("profile", {}), indent=2)
            st.download_button("Save Profile JSON", prof, file_name="profile.json")
        if available_models:
            st.caption(f"Loaded {len(available_models)} model tags from host.")

    inject_high_contrast()
    # Auto-load cached profile
    if (
        st.session_state.get("ui_step") == 1
        and st.session_state.get("auto_load_profile")
        and not st.session_state.get("profile")
    ):
        cached = load_cached_profile()
        if cached:
            st.session_state.profile = cached
            st.info("Loaded cached profile.")

    client = None
    llm_model = None
    if st.session_state.ui_step >= 2:
        client = get_ollama_client()
        if not client or not verify_ollama_auth(client):
            return
        requested_id = st.session_state.get("selected_model") or getattr(config, "DEFAULT_MODEL", "llama3.2:1b")
        # Directly use requested id (no fallback override) assuming user provides valid model
        llm_model = Ollama(id=requested_id, client=client)
        agents = get_cached_agents(llm_model)
        st.session_state.agents = agents
        # Removed warm-up ping to avoid errors on some backends
    if st.session_state.ui_step == 1:
        st.markdown("### Step 1 · Profile Setup")
        profile = render_profile_form()
        btn_disabled = profile.get("_completion", 0) < 100
        col_next, col_note = st.columns([1, 3])
        with col_next:
            proceed = st.button(
                "Next ▶", use_container_width=True, disabled=btn_disabled
            ) if hasattr(st, "button") else st.button("Next ▶")
        with col_note:
            if btn_disabled:
                st.caption("Complete all fields to proceed. This is not medical advice.")
            else:
                st.caption("Ready to generate plans.")
        if proceed and not btn_disabled:
            st.session_state.profile = {k: v for k, v in profile.items() if not k.startswith("_")}
            st.session_state.ui_step = 2
            st.rerun()

    elif st.session_state.ui_step == 2:
        st.markdown("### Step 2 · Generate Plans")
        if st.button("🔄 Back", use_container_width=True):
            st.session_state.ui_step = 1
            st.rerun()
        st.write("Review your profile below; generate when ready.")
        p = st.session_state.profile
        # Replaced JSON with styled review
        render_profile_review(p)
        if st.button("🚀 Generate Personalized Plan", type="primary", use_container_width=True):
            profile_text = (
                f"Age: {p['age']}\nWeight: {p['weight']}kg\nHeight: {p['height']}cm\n"
                f"Activity Level: {p['activity_level']}\nDiet: {p['dietary_pref']}\nGoal: {p['goal']}\n"
                f"BMI: {p['bmi']}  TDEE: {p['tdee']}\n"
            )
            with st.spinner("Generating dietary plan..."):
                try:
                    diet_resp = run_with_retry(lambda: agents["diet"].run(profile_text))
                    diet_txt = getattr(diet_resp, "content", "") or str(diet_resp)
                except Exception as e:
                    st.error(f"Diet generation error: {e}")
                    return
            with st.spinner("Generating fitness plan..."):
                try:
                    fit_resp = run_with_retry(lambda: agents["fitness"].run(profile_text))
                    fit_txt = getattr(fit_resp, "content", "") or str(fit_resp)
                except Exception as e:
                    st.error(f"Fitness generation error: {e}")
                    return
            st.session_state.dietary_plan = {
                "why_this_plan_works": "Goal-aligned caloric & macro structure",
                "meal_plan": diet_txt,
                "important_considerations": "- Hydration\n- Fiber diversity\n- Electrolyte balance\n- Portion mindfulness",
            }
            st.session_state.fitness_plan = {
                "goals": "Progressive functional strength & metabolic health",
                "routine": fit_txt,
                "tips": "- Log sessions\n- Deload every 6-8 weeks\n- Sleep 7-9h\n- Prioritize form",
            }
            st.session_state.plans_generated = True
            st.session_state.ui_step = 3
            st.success("Plans generated.")
            st.rerun()

    if st.session_state.ui_step == 3:
        ensure_edit_buffers()
        st.markdown("### Step 3 · Your Plans & Q&A")
        if st.button("🔄 Back", use_container_width=True):
            st.session_state.ui_step = 2
            st.rerun()
        display_dietary_plan(st.session_state.dietary_plan)
        display_fitness_plan(st.session_state.fitness_plan)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("Ask Follow-up Questions")
        q_col, b_col = st.columns([4,1])
        question = q_col.text_input("Your question")
        if b_col.button("Ask", use_container_width=True) and question:
            with st.spinner("Generating answer..."):
                try:
                    diet_txt = st.session_state.dietary_plan.get('meal_plan','')[:2500]
                    fit_txt = st.session_state.fitness_plan.get('routine','')[:2500]
                    context = (
                        "You are an assistant answering follow-up questions about a user's personalized plans. "
                        "Provide a concise, actionable answer (<= 130 words) referencing diet and fitness where relevant.\n\n"
                        f"Dietary Plan (truncated):\n{diet_txt}\n\n"
                        f"Fitness Plan (truncated):\n{fit_txt}\n\n"
                        f"User Question: {question}\n"
                    )
                    qa_agent = Agent(model=llm_model, show_tool_calls=False, markdown=True, instructions=[
                        "Answer clearly",
                        "If insufficient info, state assumption briefly",
                        "Keep tone encouraging but factual"
                    ])
                    resp = run_with_retry(lambda: qa_agent.run(context))
                    answer = getattr(resp, "content", "No response.")
                    st.session_state.qa_pairs.insert(0, (question, answer))
                except Exception:
                    pass
        if st.session_state.qa_pairs:
            st.markdown("#### Q&A History")
            for q, a in st.session_state.qa_pairs[:8]:
                st.markdown(f"<div class='qa-box'><div class='q'>Q: {q}</div><div>{a}</div></div>", unsafe_allow_html=True)
        # after plans shown add export PDF button
        if st.session_state.get('dietary_plan'):
            export_col1, export_col2 = st.columns([1,2])
            if export_col1.button("📄 Export Plan PDF"):
                full_text = build_plan_markdown(st.session_state.dietary_plan, st.session_state.fitness_plan)
                pdf_bytes = _generate_pdf_from_text(full_text)
                export_col1.download_button("Download PDF", data=pdf_bytes, file_name="plan.pdf", mime="application/pdf")


def render_profile_review(p: dict):
    """Visual summary chips instead of raw JSON."""
    if not p:
        st.warning("No profile loaded.")
        return
    st.markdown("#### Profile Review")
    wrap = st.container()
    with wrap:
        cols = st.columns([2,1])
        with cols[0]:
            st.markdown("<div class='profile-review'>", unsafe_allow_html=True)
            items = [
                ("Age: ", f"{p.get('age','')} yrs"),
                ("Height: ", f"{p.get('height','')} cm"),
                ("Weight: ", f"{p.get('weight','')} kg"),
                ("Activity: ", p.get('activity_level','')),
                ("Diet: ", p.get('dietary_pref','')),
                ("Goal: ", p.get('goal','')),
                ("BMI: ", p.get('bmi','')),
                ("TDEE: ", f"{p.get('tdee','')} kcal"),
            ]
            chips = []
            for label, val in items:
                chips.append(
                    f"<div class='profile-chip'><span class='chip-label'>{label}</span><span class='chip-val'>{val}</span></div>"
                )
            st.markdown("<div class='chip-grid'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cols[1]:
            try:
                buf = build_macro_chart(p.get('tdee', 0))
                st.image(buf, caption="Macro Split")
            except Exception:
                pass
    st.caption("Confirm details then generate your plans.")

def display_dietary_plan(plan_content):  # type: ignore[override]
    with st.container():
        st.markdown("<div class='plan-card'>", unsafe_allow_html=True)
        st.markdown("#### Dietary Plan")
        top_cols = st.columns([2,1])
        with top_cols[0]:
            st.markdown("**Why it works**")
            st.info(plan_content.get("why_this_plan_works", "Not available"))
            meal_raw = plan_content.get("meal_plan", "")
            tabs = st.tabs(["Structured", "Raw / Edit"])
            with tabs[0]:
                try:
                    parsed = parse_structured_meal_plan(meal_raw)  # type: ignore[name-defined]
                except NameError:
                    st.warning("Parser unavailable, using basic fallback.")
                    parsed = fallback_basic_parse(meal_raw)
                if not parsed.get('meals'):
                    parsed = fallback_basic_parse(meal_raw)
                if parsed.get('variants_flat'):
                    st.session_state.parsed_meal_plan = parsed['variants_flat']
                render_structured_meals(parsed)
            with tabs[1]:
                render_editable_plan("Meals", "dietary_plan", "meal_plan", "edit_diet")
        with top_cols[1]:
            with st.expander("Key Considerations", expanded=False):
                considerations = format_plan_display(
                    plan_content.get("important_considerations", "")
                )
                st.markdown(considerations)
        st.markdown("</div>", unsafe_allow_html=True)

def display_fitness_plan(plan_content):  # restored & enhanced
    with st.container():
        st.markdown("<div class='plan-card'>", unsafe_allow_html=True)
        st.markdown("#### Fitness Plan")
        top_cols = st.columns([2,1])
        with top_cols[0]:
            st.markdown("**Goals**")
            st.success(plan_content.get("goals", "Not specified"))
            routine_raw = plan_content.get("routine", "")
            tabs = st.tabs(["Structured", "Raw / Edit"])
            with tabs[0]:
                parsed = parse_structured_fitness_routine(routine_raw)
                if not parsed.get('flat'):
                    parsed = fallback_basic_fitness(routine_raw)
                if parsed.get('flat'):
                    st.session_state.parsed_fitness_routine = parsed['flat']
                render_structured_routine(parsed)
            with tabs[1]:
                render_editable_plan("Routine", "fitness_plan", "routine", "edit_fitness")
        with top_cols[1]:
            with st.expander("Tips", expanded=False):
                tips_fmt = format_plan_display(plan_content.get("tips", ""))
                st.markdown(tips_fmt)
        st.markdown("</div>", unsafe_allow_html=True)

def parse_structured_fitness_routine(raw: str):
    """Heuristic parser for fitness routine text.
    Extended to handle:
    - Phase headings (e.g., **Phase 1: ...**)
    - Bullet / plus prefixed exercise lines ("- + Squats: 3 sets of 10 reps")
    - Patterns like "Squats: 3 sets of 10 reps", including notes in parentheses
    - Generic colon lines like "Warm-up: 5-minute walk" or "Main Sets: 20 minutes, 3 times a week"
    - Existing formats (Name - 3 x 10, 3 sets x 10 Bench Press, 3x10 Bench Press, etc.)
    Returns { 'sections': { section: [ {exercise, sets, reps, note} ] }, 'flat': [...] }"""
    if not raw:
        return {"sections": {}, "flat": []}
    # Pre-clean lines (preserve order)
    raw_lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    lines = []
    for ln in raw_lines:
        norm = re.sub(r"^[-*]\s*\+\s*", "", ln.strip())
        norm = re.sub(r"^[+*-]\s*", "", norm)
        lines.append(norm)
    section_re = re.compile(r"^(day\s*\d+|monday|tuesday|wednesday|thursday|friday|saturday|sunday|upper body|lower body|full body|push|pull|legs|core|conditioning)[:\-]?$", re.I)
    phase_re = re.compile(r"^phase\s*\d+\b.*", re.I)
    bold_heading_re = re.compile(r"^(?:\*\*)?(walking|bodyweight exercises|warm-up|cool-down|mobility|cardio|strength|supplements)(?:\*\*)?[:\-]?", re.I)
    exercise_re = re.compile(r"^(?P<name>.+?)[\-:–]\s*(?P<sets>\d+)\s*[xX×]\s*(?P<reps>\d+(?:-\d+)?)(?:\s*(?P<note>(?!\d+\s*x).+))?$")
    alt_ex_re = re.compile(r"^(?P<sets>\d+)\s*sets?\s*[xX×]\s*(?P<reps>\d+(?:-\d+)?)[\s:-]+(?P<name>.+?)\s*(?P<note>(?:@.+)?)$")
    name_first_re = re.compile(r"^(?P<name>.+?)\s+(?P<sets>\d+)\s*[xX×]\s*(?P<reps>\d+(?:-\d+)?)(?:\s+(?P<note>.+))?$")
    sets_first_re = re.compile(r"^(?P<sets>\d+)\s*[xX×]\s*(?P<reps>\d+(?:-\d+)?)[\s:-]+(?P<name>.+?)\s*(?P<note>(?:@.+)?)$")
    name_sets_of_reps_re = re.compile(r"^(?P<name>[A-Za-z][A-Za-z '\-/()]*?):\s*(?P<sets>\d+)\s*sets?(?:\s*of)?\s*(?P<reps>\d+(?:-\d+)?)\s*reps?(?:\s*(?P<note>\(.*\).*)|\s*(?P<note2>(?!\d+\s*sets).+))?$")
    name_space_sets_of_reps_re = re.compile(r"^(?P<name>[A-Za-z][A-ZaZ '\-/()]*?)\s+(?P<sets>\d+)\s*sets?(?:\s*of)?\s*(?P<reps>\d+(?:-\d+)?)\s*reps?(?:\s*(?P<note>\(.*\).*)|\s*(?P<note2>(?!\d+\s*sets).+))?$")
    # NEW generic colon pattern (captures Warm-up: 5-minute walk or Main Sets: 20 minutes, 3 times a week)
    generic_colon_re = re.compile(r"^(?P<name>[A-Za-z][A-Za-z '\-/()]*?):\s*(?P<detail>.+)$")
    frequency_re = re.compile(r"(\d+)\s*(?:times|x)\s*(?:per|a)?\s*week", re.I)
    minutes_re = re.compile(r"(\d+)\s*(?:minute|min)s?", re.I)

    sections: dict[str, list[dict]] = {}
    current = None

    def push_ex(match):
        name = match.group('name').strip().strip(':').strip()
        sets = int(match.group('sets'))
        reps = match.group('reps')
        note = (match.group('note') or match.group('note2') or '').strip()
        sections[current].append({'exercise': name, 'sets': sets, 'reps': reps, 'note': note})

    for ln in lines:
        plain = ln.strip().strip('*')
        low = plain.lower().rstrip(':')
        # Phase or explicit section heading
        if phase_re.match(low) or section_re.match(low) or (ln.isupper() and len(ln.split()) <= 5) or bold_heading_re.match(low):
            current = plain.rstrip(':')
            sections.setdefault(current, [])
            continue
        # If we have not yet set a section, default to first Phase or Generic
        if current is None:
            current = 'Session'
            sections.setdefault(current, [])
        m = (
            exercise_re.match(plain) or
            alt_ex_re.match(plain) or
            name_first_re.match(plain) or
            sets_first_re.match(plain) or
            name_sets_of_reps_re.match(plain) or
            name_space_sets_of_reps_re.match(plain)
        )
        if m:
            push_ex(m); continue
        # Generic colon style (Warm-up: 5-minute walk or Main Sets: 20 minutes, 3 times a week)
        gm = generic_colon_re.match(plain)
        if gm:
            detail = gm.group('detail').strip()
            freq_m = frequency_re.search(detail)
            min_m = minutes_re.search(detail)
            sets_val = None
            reps_val = None
            note_val = ''
            # Heuristic: if we have minutes AND frequency create reps string
            if min_m and freq_m:
                reps_val = f"{min_m.group(1)} min, {freq_m.group(1)}x/week"
            elif min_m:
                reps_val = f"{min_m.group(1)} min"
            elif freq_m:
                reps_val = f"{freq_m.group(1)}x/week"
            else:
                reps_val = detail
            sections[current].append({'exercise': gm.group('name').strip(), 'sets': sets_val, 'reps': reps_val, 'note': note_val})
            continue
        overload_keywords = ['progressive overload', 'increase', 'rest', 'cool-down', 'warm-up']
        if sections[current] and any(k in low for k in overload_keywords):
            sections[current][-1]['note'] = (sections[current][-1].get('note','') + ' ' + plain).strip()

    flat = []
    for sec, items in sections.items():
        for it in items:
            flat.append({'section': sec, **it})
    return {"sections": sections, "flat": flat}

# --- Meal Plan Parsing & Rendering (restored) ---
MEAL_HEADING_RE = re.compile(r"^(breakfast|lunch|dinner|snack|snacks|pre[- ]?workout|post[- ]?workout)[:\-]?$", re.I)


def parse_structured_meal_plan(raw: str):
    """Heuristic meal plan parser returning dict with macros, meals, variants_flat.
    Each meal maps to list of entries {variant, raw_title, ingredients, macros}."""
    if not raw:
        return {"macros": [], "meals": {}, "variants_flat": []}
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    meals: dict[str, list[dict]] = {}
    current = None
    variants_flat = []
    for ln in lines:
        low = ln.lower().strip('*').rstrip(':')
        # Detect meal heading (**Breakfast** or Breakfast:)
        if (ln.startswith('**') and ln.endswith('**')) or MEAL_HEADING_RE.match(low):
            name = low.strip('*').capitalize().rstrip(':')
            meals.setdefault(name, [])
            current = name
            continue
        if current is None:
            # first lines before heading -> skip
            continue
        # Bullet / numbered variant or plain ingredient line
        cleaned = re.sub(r"^([*\-+]|\d+\.)\s+", "", ln).strip()
        if not cleaned:
            continue
        entry = {
            "variant": f"Option {len(meals[current]) + 1}",
            "raw_title": cleaned,
            "days": [],
            "details": [],
            "calories": None,
            "protein": None,
            "carbs": None,
            "fat": None,
            "ingredients": [cleaned],
        }
        meals[current].append(entry)
        variants_flat.append({"meal": current, **entry})
    return {"macros": [], "meals": meals, "variants_flat": variants_flat}


def fallback_basic_parse(raw: str):
    """Very loose fallback: groups all lines under a single 'Meal Plan' section."""
    if not raw:
        return {"macros": [], "meals": {}, "variants_flat": []}
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    entry = {
        "variant": "Option 1",
        "raw_title": "Option 1",
        "days": [],
        "details": [],
        "calories": None,
        "protein": None,
        "carbs": None,
        "fat": None,
        "ingredients": lines,
    }
    return {"macros": [], "meals": {"Meal Plan": [entry]}, "variants_flat": [{"meal": "Meal Plan", **entry}]}


def render_structured_meals(parsed: dict):
    meals = parsed.get('meals', {})
    if not meals:
        st.info("No structured meals parsed.")
        return
    for meal_name, variants in meals.items():
        with st.expander(meal_name, expanded=False):
            for v in variants:
                st.markdown(f"**{v['variant']}**: ")
                for ing in v.get('ingredients', []):
                    st.markdown(f"- {ing}")

def fallback_basic_fitness(raw: str):
    """Very permissive fallback: capture any 'number x number' patterns as exercises."""
    if not raw:
        return {"sections": {}, "flat": []}
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    ex_re = re.compile(r"(?P<sets>\d+)\s*[xX×]\s*(?P<reps>\d+(?:-\d+)?)")
    collected = []
    for ln in lines:
        m = ex_re.search(ln)
        if m:
            name = ex_re.sub("", ln).strip(" -:") or ln
            collected.append({'section': 'Session', 'exercise': name, 'sets': int(m.group('sets')), 'reps': m.group('reps'), 'note': ''})
    return {"sections": {"Session": collected} if collected else {}, "flat": collected}


def render_structured_routine(parsed: dict):
    sections = parsed.get('sections', {})
    if not sections:
        st.info("No structured routine parsed.")
        return
    for sec, items in sections.items():
        with st.expander(sec, expanded=False):
            for ex in items:
                sets = ex.get('sets')
                reps = ex.get('reps')
                note = ex.get('note','')
                if sets and reps:
                    line = f"- **{ex['exercise']}**: {sets} x {reps}"
                elif reps:
                    line = f"- **{ex['exercise']}**: {reps}"
                else:
                    line = f"- **{ex['exercise']}**"
                if note:
                    line += f"  {note}"
                st.markdown(line)
