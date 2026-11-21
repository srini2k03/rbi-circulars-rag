"""Streamlit page: check a bank policy clause by clause against RBI notifications.

Talks to the API's ``/comply/*`` endpoints at ``RBIRAG_API_URL``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from rbirag.ui_client import (
    VERDICT_ORDER,
    VERDICT_STYLE,
    findings_rows,
    get_json,
    post_file,
    verdict_badge,
    verdict_counts,
)

API_URL = os.environ.get("RBIRAG_API_URL", "http://localhost:8000")
SAMPLE = Path(os.environ.get("RBIRAG_SAMPLE_POLICY", "data/samples/kyc_policy_sample.md"))
BACKEND_LABELS = {
    "gemini": "Gemini (free tier)",
    "groq": "Groq (free tier)",
    "ollama": "Ollama (local)",
    "fake": "Offline demo (no LLM; always 'insufficient evidence')",
}

st.set_page_config(page_title="Policy compliance check", page_icon=":bank:", layout="wide")
st.title("Check a policy against RBI regulations")


def status() -> dict[str, Any]:
    try:
        return get_json(API_URL, "/comply/status")
    except Exception as exc:  # API down
        st.error(f"API unreachable at {API_URL}: {exc}")
        st.stop()
        raise  # unreachable; for type checkers


info = status()
st.caption(
    f"Each clause gets a verdict (conflict, gap, compliant, insufficient evidence or not "
    f"applicable) with the notifications relied on and a confidence. Judged as of "
    f"{info['as_of']}, for {info['entity'].lower()}, against "
    f"{info['notifications']:,} RBI notifications. Not legal advice."
)
if not info["loaded"]:
    if info.get("error"):
        st.error(f"The regulation corpus is not available: {info['error']}")
        st.stop()
    st.info("Loading the regulation corpus...")

# -- options ---------------------------------------------------------------------------
available = [b for b, ok in info["backends"].items() if ok]
default = info["default_backend"]
with st.sidebar:
    st.header("Options")
    system = st.radio(
        "System",
        ["agent", "baseline"],
        captions=["plans its own searches and reads", "one search, one comparison"],
    )
    backend = st.selectbox(
        "Model",
        available,
        index=available.index(default) if default in available else 0,
        format_func=lambda b: BACKEND_LABELS.get(b, b),
    )
    max_clauses = st.slider(
        "Clauses to check",
        1,
        int(info["max_clauses_per_check"]),
        min(8, info["max_clauses_per_check"]),
    )
    st.caption(
        f"The public demo checks at most {info['max_clauses_per_check']} clauses per run on "
        "free-tier models; the agent makes ~5 model calls per clause."
    )

# -- input -----------------------------------------------------------------------------
upload, paste, sample = st.tabs(["Upload a file", "Paste text", "Sample policy"])
policy: tuple[str, bytes] | None = None
with upload:
    f = st.file_uploader("Policy document", type=["pdf", "txt", "md"])
    if f is not None:
        policy = (f.name, f.getvalue())
with paste:
    text = st.text_area("Policy text (numbered clauses work best)", height=220)
    if text.strip() and policy is None:
        policy = ("pasted_policy.md", text.encode("utf-8"))
with sample:
    if SAMPLE.exists():
        sample_text = SAMPLE.read_text(encoding="utf-8")
        use_sample = st.checkbox(f"Use the sample policy ({SAMPLE.name})", value=policy is None)
        with st.expander("Show the sample", expanded=False):
            st.code(sample_text[:5000], language="markdown")
        if use_sample and policy is None:
            policy = (SAMPLE.name, sample_text.encode("utf-8"))
    else:
        st.write("No sample policy bundled.")

if policy is None:
    st.stop()
name, content = policy

code, clauses = post_file(API_URL, "/comply/clauses/upload", filename=name, content=content)
if code != 200:
    st.error(f"Could not read the policy: {clauses.get('detail')}")
    st.stop()
found = clauses["clauses"]
with st.expander(f"{len(found)} clauses found in {name}", expanded=False):
    st.dataframe(
        [
            {
                "clause": c["label"],
                "section": " > ".join(c["heading_path"]),
                "text": c["text"][:200],
            }
            for c in found
        ],
        hide_index=True,
        use_container_width=True,
    )

if st.button(f"Check {min(max_clauses, len(found))} clause(s)", type="primary"):
    code, job = post_file(
        API_URL,
        "/comply/check/upload",
        filename=name,
        content=content,
        data={"system": system, "backend": backend, "max_clauses": str(max_clauses)},
    )
    if code != 202:
        st.error(f"Could not start the check: {job.get('detail')}")
        st.stop()
    st.session_state["job_id"] = job["job_id"]

job_id = st.session_state.get("job_id")
if not job_id:
    st.stop()


# -- progress and results --------------------------------------------------------------
def render_finding(f: dict[str, Any]) -> None:
    p, clause = f["prediction"], f["clause"]
    label = clause.get("number") or clause["clause_id"]
    conf = p.get("confidence")
    title = f"{label}: {VERDICT_STYLE.get(p['verdict'], ('', p['verdict']))[1]}" + (
        "" if conf is None else f" (confidence {conf:.2f})"
    )
    with st.expander(title, expanded=p["verdict"] in {"conflict", "gap"}):
        st.markdown(verdict_badge(p["verdict"]))
        st.markdown(f"> {' '.join(clause['text'].split())}")
        if p.get("error"):
            st.error(f"This clause failed to run: {p['error']}")
        if p.get("rationale"):
            st.write(p["rationale"])
        for src in f.get("sources", []):
            st.markdown(
                f"**[{src['doc_id']}]({src['url']})** {src['title']} "
                f"({src.get('date') or 'undated'})"
            )
            for q in src.get("quotes", []):
                st.markdown(f"> _{q}_")
        if p.get("notes"):
            st.caption("Checks: " + "; ".join(p["notes"]))
        if p.get("tool_calls"):
            steps = [
                f"{i}. `{c['name']}` {json.dumps(c['arguments'], ensure_ascii=False)[:160]}"
                for i, c in enumerate(p["tool_calls"], start=1)
            ]
            st.caption("Steps:\n\n" + "\n\n".join(steps))


progress = st.progress(0.0, text="Starting...")
live = st.container()
job: dict[str, Any] = {}
while True:
    try:
        job = get_json(API_URL, f"/comply/jobs/{job_id}")
    except httpx.HTTPError as exc:
        st.error(f"Lost the job ({exc}); start a new check.")
        st.session_state.pop("job_id", None)
        st.stop()
    total = max(1, job["clauses_total"])
    progress.progress(
        job["clauses_done"] / total,
        text=f"{job['state']}: {job['clauses_done']} of {job['clauses_total']} clauses "
        f"({job['system']}, {BACKEND_LABELS.get(job['backend'], job['backend'])})",
    )
    if job["state"] in {"succeeded", "failed"}:
        break
    time.sleep(2)

findings = (job.get("report") or {}).get("findings") or job.get("findings", [])
if job["state"] == "failed":
    st.error(
        f"The check stopped: {job.get('error')}. Results for the clauses done so far are below."
    )

with live:
    counts = verdict_counts(findings)
    for col, v in zip(st.columns(len(VERDICT_ORDER)), VERDICT_ORDER, strict=True):
        col.metric(VERDICT_STYLE[v][1].title(), counts.get(v, 0))
    report = job.get("report")
    if report:
        st.caption(
            f"Model: {report.get('model') or '-'} | LLM calls: {report['llm_calls']} | tokens: "
            f"{report['tokens']:,} | time: {report['latency_ms'] / 1000:.0f} s | "
            "cost: $0 (free tier)"
        )
    st.dataframe(findings_rows(findings), hide_index=True, use_container_width=True)
    for f in findings:
        render_finding(f)
    if report:
        md = httpx.get(f"{API_URL.rstrip('/')}/comply/jobs/{job_id}/report.md", timeout=30).text
        c1, c2 = st.columns(2)
        c1.download_button("Download report (Markdown)", md, file_name="compliance_report.md")
        c2.download_button(
            "Download report (JSON)",
            json.dumps(report, indent=2, ensure_ascii=False),
            file_name="compliance_report.json",
        )
