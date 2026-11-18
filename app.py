# Fall Case Review Assistant (Streamlit). Run locally with: streamlit run app.py
# Claude helped me write this code (see README for what was AI-assisted and what I checked myself).
import json
import os
from html import escape as esc
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import AGE_BANDS, CATEGORY_LABELS, DEFAULT_CHAT_MODEL, LLM, PLACES, SearchIndex, run_agent


# ======================================================================
# HTML + CSS HELPERS (plain functions that return strings; colors and fonts match the portfolio page)
# ======================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@600;700&display=swap');
:root {
  --ink:#14233a; --muted:#475970; --blue:#1f4e9c; --blue-deep:#0b1f3a; --blue-mid:#143a70; --amber:#e0a030;
  --amber-soft:#fdf0d0; --bg:#f3f6fb; --card:#ffffff; --line:#d6e0ee; --radius:14px; --red:#b3261e; --red-soft:#fdecea;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:"Inter", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.stApp { background: var(--bg); }
html, body, .stApp, .stMarkdown, .stTextArea textarea, button, label, p, li { font-family: var(--sans); color: var(--ink); }
.block-container { max-width: 1040px; padding: 1rem 1.2rem 4rem 1.2rem; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { font-family: var(--serif); color: var(--blue-deep); }
.stMarkdown p, .stMarkdown li { font-size: 1.05rem; line-height: 1.65; }
a { color: var(--blue); }
:focus-visible { outline: 3px solid var(--amber) !important; outline-offset: 2px; }

/* hero */
.hero { color:#eef3fb; border-radius:20px; padding:36px 32px 30px; margin:6px 0 18px;
  background: radial-gradient(520px 360px at 85% 25%, rgba(224,160,48,.30), rgba(224,160,48,0) 70%), linear-gradient(180deg,#0b1f3a,#10294d); }
.hero .eyebrow { letter-spacing:.14em; text-transform:uppercase; font-size:.78rem; color:var(--amber); font-weight:600; margin:0 0 10px; }
.hero h1 { font-family:var(--serif); color:#fff; font-size:clamp(1.9rem,5vw,3rem); line-height:1.1; margin:0 0 12px; padding:0; }
.hero .lede { color:#d3dff2; font-size:clamp(1rem,2.4vw,1.2rem); max-width:60ch; margin:0 0 16px; line-height:1.55; }
.pill { display:inline-block; font-size:.85rem; padding:4px 12px; border-radius:999px; margin:0 8px 6px 0;
  background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.28); color:#e8eefa; }

/* cards and notices */
.card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:18px 20px; margin:0 0 14px; }
.card h3 { margin:0 0 6px; font-size:1.2rem; }
.card p:last-child { margin-bottom:0; }
.card.note { background:var(--amber-soft); border-color:#efd391; }
.notice { border-radius:var(--radius); padding:14px 18px; margin:0 0 14px; border:1px solid; line-height:1.55; }
.notice.safety { background:var(--red-soft); border-color:#e8b4af; color:#5a1410; border-left:6px solid var(--red); }
.notice.info { background:#e8f0fb; border-color:#bcd0ee; color:#0b1f3a; }
.notice.warn { background:var(--amber-soft); border-color:#efd391; color:#4a3300; }
.notice b { font-weight:600; }
.steps { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); margin:0 0 14px; }
.steps .card { margin:0; border-top:5px solid var(--blue); }
.steps .card.amber { border-top-color:var(--amber); }

/* summary */
.headline { font-family:var(--serif); font-size:1.25rem; color:var(--blue-deep); margin:0 0 10px; line-height:1.35; }
ul.items { margin:0 0 10px; padding-left:1.2rem; }
ul.items li { margin-bottom:8px; }
.cite { display:inline-block; font-size:.78rem; font-weight:600; background:#e8f0fb; color:var(--blue-deep); border:1px solid #bcd0ee;
  border-radius:999px; padding:0 8px; margin-left:4px; white-space:nowrap; }
.src { font-size:.85rem; color:var(--muted); }

/* case cards */
.case { background:var(--card); border:1px solid var(--line); border-left:6px solid var(--blue); border-radius:var(--radius); padding:14px 18px 12px; margin:0 0 6px; }
.case .top { display:flex; flex-wrap:wrap; gap:6px 12px; align-items:baseline; margin-bottom:8px; }
.case .num { font-family:var(--serif); font-weight:700; font-size:1.15rem; color:var(--blue-deep); }
.case .meta { font-size:.9rem; color:var(--muted); }
.case .narr { background:#eef3fa; border-radius:10px; padding:10px 12px; font-size:.98rem; line-height:1.55; margin:0 0 10px; overflow-wrap:anywhere; }
.case .narr small { display:block; color:var(--muted); font-size:.75rem; letter-spacing:.06em; text-transform:uppercase; margin-bottom:3px; }
.tagrow { display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; }
.tag { display:inline-block; font-weight:600; font-size:.88rem; padding:3px 12px; border-radius:999px; background:var(--amber-soft); border:1px solid #efd391; color:#4a3300; }
.tag.rule { background:#eef3fa; border-color:var(--line); color:var(--blue-deep); }
.tag.unv { background:#fff; border:1px dashed var(--muted); color:var(--muted); }
.quote { font-size:.92rem; color:var(--muted); }
.chip { display:inline-block; font-size:.8rem; padding:1px 10px; border-radius:999px; border:1px solid var(--line); background:#fff; color:var(--muted); }

/* trace */
table.trace { width:100%; border-collapse:collapse; font-size:.92rem; }
table.trace th, table.trace td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
table.trace th { background:#eaf0f9; font-weight:600; color:var(--blue-deep); }
.kind { font-size:.75rem; font-weight:600; border-radius:999px; padding:1px 8px; color:#fff; }
.kind.code { background:var(--blue); } .kind.llm { background:#8a4f7d; } .kind.ml { background:#2e7d5b; }
.ok { color:#1d6b3d; font-weight:600; } .bad { color:var(--red); font-weight:600; } .mid { color:#8a5a00; font-weight:600; }

/* streamlit widgets */
div[data-testid="stFormSubmitButton"] button, button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background:var(--amber) !important; color:#14233a !important; border:0 !important; border-radius:999px !important; font-weight:600 !important; padding:.55rem 1.5rem !important; }
div[data-testid="stFormSubmitButton"] button:hover { background:#f0b445 !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { border-radius:999px !important; border:2px solid var(--blue) !important; background:#fff !important; color:var(--blue-deep) !important; font-weight:500 !important; }
.stTextArea textarea { font-size:1.05rem !important; line-height:1.5; border-radius:12px; border:2px solid var(--line); background:#fff; }
.stTextArea textarea:focus { border-color:var(--blue); }
.stTabs [data-baseweb="tab"] { font-size:1.05rem; padding:10px 14px; }
.stTabs [aria-selected="true"] { color:var(--blue-deep); font-weight:600; }
div[data-testid="stForm"] { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px; }
.foot { color:var(--muted); font-size:.9rem; border-top:1px solid var(--line); margin-top:26px; padding-top:14px; }

@media (max-width: 640px) {
  .block-container { padding: .6rem .7rem 3rem .7rem; }
  .hero { padding: 26px 18px 22px; border-radius:16px; }
  .case { padding:12px 13px 10px; }
  .card { padding:14px 15px; }
  table.trace th:nth-child(4), table.trace td:nth-child(4) { display:none; }
  .stTabs [data-baseweb="tab"] { padding: 8px 8px; font-size: .95rem; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto !important; transition:none !important; } }
"""

MODE_NAMES = {"facility": "Facility staff", "family": "Family or caregiver", "self": "Myself (older adult)"}


def hero():
    return ("<div class='hero'><p class='eyebrow'>Falls in older adults</p><h1>Fall Case Review Assistant</h1>"
            "<p class='lede'>Describe a fall in your own words. The assistant finds similar real emergency-department notes, "
            "shows what they have in common, and suggests questions to ask. It is a review aid, not a diagnosis.</p>"
            "<span class='pill'>Public NEISS 2024 data, adults 65+</span><span class='pill'>Not medical advice</span>"
            "<span class='pill'>Works on phone and desktop</span></div>")


def empty_state():
    return ("<div class='steps'>"
            "<div class='card'><h3>1. Describe</h3><p>Say what happened in a sentence or two: where, what the person was doing, and anything they noticed.</p></div>"
            "<div class='card amber'><h3>2. Compare</h3><p>The assistant searches 57,209 real emergency-department fall notes for the most similar ones and shows each note word for word.</p></div>"
            "<div class='card'><h3>3. Review</h3><p>You get a short summary and a checklist of questions. Every item points to the cases it came from, so you can check it.</p></div>"
            "</div>"
            "<div class='notice info'><b>What this is not.</b> It does not diagnose, predict who will fall, or replace a clinician. "
            "The notes come from emergency-department visits, so they show how serious falls happened, not how likely any fall is.</div>")


def safety(flags, mode):
    base = ("If someone is hurt, cannot get up, hit their head, or you are unsure, contact a doctor or call emergency services "
            "(911 in the U.S.). Do not wait for this tool.")
    if flags:
        listed = esc("; ".join(flags))
        who = "Follow your facility's post-fall protocol and clinical assessment. " if mode == "facility" else ""
        return f"<div class='notice safety' role='alert'><b>Your description mentions: {listed}.</b> Falls with these signs can need medical attention. {who}{esc(base)}</div>"
    if mode == "facility":
        return ""
    return f"<div class='notice safety'><b>Safety first.</b> {esc(base)}</div>"


def notice(kind, text, title=""):
    head = f"<b>{esc(title)}</b> " if title else ""
    return f"<div class='notice {kind}'>{head}{esc(text)}</div>"


def _cites(ids):
    return "".join(f"<span class='cite'>case #{esc(i[1:])}</span>" for i in ids)


def summary(summary_dict, ai_note=True):
    if not summary_dict:
        return ""
    parts = [f"<div class='card'><h3>What the similar cases show</h3><p class='headline'>{esc(summary_dict['headline'])}</p>"]
    if summary_dict["what_cases_show"]:
        parts.append("<ul class='items'>" + "".join(f"<li>{esc(i['text'])}{_cites(i['cases'])}</li>" for i in summary_dict["what_cases_show"]) + "</ul>")
    parts.append("</div>")
    if summary_dict["checks"]:
        parts.append("<div class='card note'><h3>Questions and things to check</h3><ul class='items'>"
                     + "".join(f"<li>{esc(i['text'])}{_cites(i['cases'])}</li>" for i in summary_dict["checks"]) + "</ul>")
        src = ("Drafted by an AI model from the cases below and checked by code. Review before you rely on it."
               if summary_dict["source"] == "ai" else "Fixed wording based on the factors found in the cases (no AI drafting).")
        parts.append(f"<p class='src'>{esc(src)}</p></div>")
    return "".join(parts)


def case_card(row):
    tag_cls = {"ai": "", "rule": "rule", "unverified": "unv"}[row["tag_source"]]
    tag_note = {"ai": "AI tag, quote checked", "rule": "Keyword-rule tag", "unverified": "AI tag, quote not verified"}[row["tag_source"]]
    quote = f"<span class='quote'>from the note: &ldquo;{esc(row['quote'])}&rdquo;</span>" if row["quote"] else ""
    age = f"{int(row['age_years'])}" if row["age_years"] == row["age_years"] else "age n/a"
    sim_kind = "meaning" if row.get("similarity_kind") == "embedding" else "keyword"
    return (f"<div class='case'><div class='top'><span class='num'>Case #{int(row['rank'])}</span>"
            f"<span class='meta'>NEISS {esc(str(row['case_number']))} &middot; age {esc(age)}, {esc(str(row['sex_label']).lower())} &middot; "
            f"{esc(str(row['location_label']))} &middot; {esc(str(row['disposition_label']))}</span>"
            f"<span class='chip'>{sim_kind} match {row['similarity']:.2f}</span></div>"
            f"<div class='narr'><small>Original note, word for word</small>{esc(str(row['narrative']))}</div>"
            f"<div class='tagrow'><span class='tag {tag_cls}'>{esc(CATEGORY_LABELS[row['tag']])}</span>"
            f"<span class='chip'>{tag_note}</span>{quote}</div></div>")


def group_card(ctx, show_admission):
    line = ""
    if show_admission:
        line = (f"<p>Among all {ctx['group_size']:,} notes in this group, {ctx['group_admitted_rate'] * 100:.0f}% ended in hospital admission, "
                f"compared with {ctx['overall_admitted_rate'] * 100:.0f}% of all notes. This describes past emergency-department visits. "
                "It does not say what will happen to anyone.</p>")
    return (f"<div class='card'><h3>Story group: {esc(ctx['label'])}</h3>"
            f"<p>{round(ctx['coherence'] * 100)}% of the matched cases fall in this group. Story groups come from a clustering model (K-Means) "
            f"and are loose, so treat them as a rough guide.</p>{line}</div>")


def trace_table(trace):
    status_cls = {"done": "ok", "fallback": "bad", "warning": "mid", "skipped": "mid"}
    rows = "".join(
        f"<tr><td>{esc(t['step'])}</td><td><span class='kind {t['kind']}'>{ {'code': 'code', 'llm': 'AI', 'ml': 'ML'}[t['kind']] }</span></td>"
        f"<td class='{status_cls.get(t['status'], '')}'>{esc(t['status'])}</td><td>{esc(t['note'])}</td><td>{t['seconds']}s</td></tr>" for t in trace)
    return ("<table class='trace'><tr><th>Step</th><th>Type</th><th>Result</th><th>Detail</th><th>Time</th></tr>" + rows + "</table>")


def footer():
    return ("<div class='foot'>This tool searches public emergency-department notes (U.S. CPSC NEISS, 2024). It does not diagnose, predict, or give medical advice. "
            "Do not type names, addresses, or other identifying details: your description is sent to an AI service to produce the results and is not saved by this app." "</div>")

# ======================================================================
# THE APP
# ======================================================================
DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Fall Case Review Assistant", page_icon="🌙", layout="centered")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

EXAMPLES = {
    "facility": "Resident found on the bathroom floor at 3 a.m. after getting up to use the toilet without calling for help. Complains of hip pain.",
    "family": "My dad tripped on the edge of the rug in the hallway and hit his head on the door frame. He takes blood thinners.",
    "self": "I stood up quickly from my chair, felt dizzy, and fell in the kitchen. I landed on my side.",
}


@st.cache_resource(show_spinner="Loading 57,209 fall notes and building the search index (first load only)...")
def load_assets():
    df = pd.read_csv(DATA / "falls.csv.gz")  # notes plus each note's story group and map position
    info = json.loads((DATA / "ml_info.json").read_text())  # ML metrics, group profiles, group labels
    groups = df[["case_number", "group", "x", "y"]]
    return df, SearchIndex(df), groups, info, info["labels"]


def get_llm():
    """The key comes from Streamlit secrets or an environment variable. It is never written in the code."""
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        model = st.secrets.get("OPENAI_MODEL", DEFAULT_CHAT_MODEL)
    except Exception:  # no secrets file: fine, fall back to the environment
        model = DEFAULT_CHAT_MODEL
    key = key or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", model)
    return LLM(api_key=key, chat_model=model)


def use_example():
    st.session_state["description"] = EXAMPLES[mode_key(st.session_state.get("mode_label", MODE_NAMES["facility"]))]


def mode_key(label):
    return {v: k for k, v in MODE_NAMES.items()}[label]


def story_map(groups, cases):
    sample = groups.sample(4000, random_state=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample["x"], y=sample["y"], mode="markers", name="All notes (sample)",
                             marker=dict(size=4, color="#c3d1e6"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cases["x"], y=cases["y"], mode="markers+text", text=[f"#{r}" for r in cases["rank"]],
                             textposition="top center", name="Matched cases",
                             marker=dict(size=13, color="#e0a030", line=dict(width=2, color="#0b1f3a")),
                             hovertext=cases["group_label"], hoverinfo="text"))
    fig.update_layout(height=330, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="white", plot_bgcolor="white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.05),
                      font=dict(family="Inter, sans-serif", color="#14233a"))
    return fig


def summary_text(result):
    lines = [f"Fall Case Review Assistant - {MODE_NAMES[result['mode']]}", f"Description: {result['description']}", ""]
    s = result["summary"]
    if s:
        lines += [s["headline"], ""] + [f"- {i['text']} ({', '.join(i['cases'])})" for i in s["what_cases_show"]]
        lines += ["", "Questions and things to check:"] + [f"- {i['text']} ({', '.join(i['cases'])})" for i in s["checks"]]
    lines += ["", "Cases used (NEISS 2024):"] + [f"{r.case_id} = NEISS {r.case_number}" for r in result["cases"].itertuples()]
    lines += ["", "Not medical advice. Built from public emergency-department notes."]
    return "\n".join(lines)


def show_results(result, groups):
    status = result["status"]
    if result["flags"] or result["mode"] != "facility":
        st.markdown(safety(result["flags"], result["mode"]), unsafe_allow_html=True)
    if status == "needs_input":
        st.markdown(notice("warn", "Please write a sentence or two about what happened (where, what the person was doing).", "A little more detail, please."), unsafe_allow_html=True)
        return
    if status == "no_results":
        st.markdown(notice("warn", "No notes matched that description, so nothing is shown rather than guessing. Try plain words such as 'tripped on stairs' or remove the filters.", "No matching cases."), unsafe_allow_html=True)
        return
    if result["degraded"]:
        st.markdown(notice("warn", "The AI service was not fully available, so some steps used simpler code-only methods. Results are still real notes, but wording is more basic.", "Running in reduced mode."), unsafe_allow_html=True)
    elif not result["ai_available"]:
        st.markdown(notice("info", "No AI key is set, so this run used keyword search and fixed wording only.", "Offline mode."), unsafe_allow_html=True)
    for w in result["warnings"]:
        st.markdown(notice("warn", w), unsafe_allow_html=True)
    if status == "weak_match":
        st.markdown(notice("warn", "I could not find a strong match, so I am not writing a summary. The closest notes are listed below for you to read yourself.", "Not enough to go on."), unsafe_allow_html=True)
    if result["summary"]:
        st.markdown(summary(result["summary"]), unsafe_allow_html=True)

    cases = result["cases"]
    st.subheader("The cases behind this")
    st.caption("Each note is the original emergency-department text. Case numbers in the summary refer to these.")
    cases = cases.assign(similarity_kind=result["similarity_kind"])
    for i, row in cases.iterrows():
        st.markdown(case_card(row), unsafe_allow_html=True)
        if st.button("This tag looks wrong", key=f"flag_{row['case_id']}_{row['case_number']}"):
            st.session_state.setdefault("flags", []).append({"case": str(row["case_number"]), "tag": row["tag"]})
        if any(f["case"] == str(row["case_number"]) for f in st.session_state.get("flags", [])):
            st.caption("Thanks, noted for this session. Tags are a best guess.")

    if result["group_context"]:
        st.markdown(group_card(result["group_context"], show_admission=result["mode"] != "self"), unsafe_allow_html=True)
        with st.expander("See where these cases sit on the story map"):
            st.plotly_chart(story_map(groups, cases), use_container_width=True, config={"displayModeBar": False})
            st.caption("Each dot is a note. Notes with similar wording sit close together. Amber dots are your matched cases.")
    with st.expander("How this answer was produced (step by step)"):
        st.markdown(trace_table(result["trace"]), unsafe_allow_html=True)
    st.download_button("Download this summary (text)", summary_text(result), file_name="fall_review_summary.txt")


# ---------- page ----------
st.markdown(hero(), unsafe_allow_html=True)
df, index, groups, info, labels = load_assets()
llm = get_llm()
tab_check, tab_how, tab_about = st.tabs(["Check a fall", "How it works", "About the data and models"])

with tab_check:
    mode_label = st.radio("Who is using this?", list(MODE_NAMES.values()), key="mode_label", horizontal=True)
    mode = mode_key(mode_label)
    with st.form("describe"):
        st.text_area("What happened?", key="description", height=130, max_chars=1200,
                     placeholder="Example: " + EXAMPLES[mode])
        with st.expander("Optional filters"):
            c1, c2 = st.columns(2)
            place = c1.selectbox("Place", list(PLACES), help="Leave on 'Any place' and the assistant may pick one from your description.")
            age_band = c2.selectbox("Age band of the cases", list(AGE_BANDS))
        go_btn = st.form_submit_button("Find similar cases", type="primary")
    st.button("Use an example", on_click=use_example)
    st.caption("Do not type names or addresses. Your text is sent to an AI service and is not saved by this app.")

    if go_btn:
        with st.status("Working through the steps...", expanded=True) as box:
            def show_step(item):
                icon = {"done": "✅", "fallback": "⚠️", "warning": "⚠️", "skipped": "➖"}.get(item["status"], "•")
                box.write(f"{icon} {item['step']}: {item['note']}")
            try:
                st.session_state["result"] = run_agent(st.session_state.get("description", ""), mode, place, age_band,
                                                       llm, index, groups, info, labels, on_step=show_step)
                box.update(label="Done", state="complete", expanded=False)
            except Exception as err:  # last-resort guard so the page never shows a stack trace
                st.session_state["result"] = None
                box.update(label="Something went wrong", state="error")
                st.markdown(notice("warn", f"An unexpected error stopped the search ({type(err).__name__}). Please try again or shorten the description.", "Sorry."), unsafe_allow_html=True)
    result = st.session_state.get("result")
    if result:
        show_results(result, groups)
    elif not go_btn:
        st.markdown(empty_state(), unsafe_allow_html=True)

with tab_how:
    st.markdown("### The assistant runs a fixed, logged sequence")
    st.markdown(
        "1. **Check the input** (code): looks for red-flag words such as head strike or blood thinners.\n"
        "2. **Plan the search** (AI): suggests other ways the notes might phrase the same event.\n"
        "3. **Keyword search** (code): TF-IDF over all 57,209 notes.\n"
        "4. **Rerank by meaning** (AI embeddings): reorders the best 60 candidates.\n"
        "5. **Check match quality** (code): if matches are weak, it says so and writes no summary.\n"
        "6. **Tag each case** (AI + check): every tag must come with a quote that really appears in the note.\n"
        "7. **Story groups** (machine learning): places cases in K-Means groups.\n"
        "8. **Write the summary** (AI) and 9. **verify it** (code): every claim must cite a retrieved case and avoid medical-advice wording; one retry, then fixed wording.")
    st.markdown("### When it cannot answer")
    st.markdown("- No key or AI outage: runs on code only and says so.\n- Weak match: shows the closest notes, no summary.\n- Bad AI output: rejected and replaced by fixed wording.\n- Red-flag words: a safety notice appears before anything else.")

with tab_about:
    m = info["metrics"]
    st.markdown("### Data")
    st.markdown("Public NEISS 2024 injury notes for adults 65 and over, 57,209 falls. Each note is a short emergency-department narrative.")
    st.markdown("### Story groups (non-AI machine learning)")
    st.markdown("Notes are cleaned, turned into TF-IDF word weights, reduced with SVD to 50 dimensions, and grouped with K-Means (k=8).")
    st.markdown(f"- Silhouette score: **{m['silhouette_by_k']['8']:.3f}** (low: groups overlap)\n- Stability across random seeds (ARI): **{m['seed_stability_mean']:.2f}**\n"
                f"- Admission AUC from group alone: **{m['auc_admission_from_group_only']:.2f}** vs age alone **{m['auc_admission_from_age_only']:.2f}**")
    st.markdown("**Limits.** The groups are loose and only moderately stable. They describe how notes are worded, not a person's risk. "
                "The data covers people who reached an emergency department, so minor falls are missing.")
    rows = [{"Group": labels[str(g['group'])]["label"], "Notes": g["size"], "Admitted": f"{g['admitted_rate'] * 100:.0f}%"} for g in info["groups"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.markdown(footer(), unsafe_allow_html=True)
