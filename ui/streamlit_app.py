"""Streamlit chat UI for rbi-circulars-rag.

Run: ``streamlit run ui/streamlit_app.py`` with the API at ``RBIRAG_API_URL``
(default http://localhost:8000).
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from rbirag.ui_client import get_json, linkify_citations, source_caption, stream_query

API_URL = os.environ.get("RBIRAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RBI circulars RAG", page_icon=":bank:", layout="wide")


def sidebar() -> dict[str, Any]:
    st.sidebar.header("Settings")
    try:
        health = get_json(API_URL, "/health")
    except Exception as exc:  # API down
        st.sidebar.error(f"API unreachable at {API_URL}: {exc}")
        health = {}
    if health:
        st.sidebar.caption(
            f"Index: {health.get('chunks', 0)} chunks from {health.get('documents', 0)} documents "
            f"({health.get('chunker')}) | status: {health.get('status')}"
        )
    backends = [b for b, ok in (health.get("backends") or {"groq": True}).items() if ok]
    default = health.get("default_backend", "groq")
    mode = st.sidebar.selectbox("Retrieval mode", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    top_k = st.sidebar.slider("Sources (top-k)", 1, 10, 5)
    backend = st.sidebar.selectbox(
        "LLM backend",
        backends or [default],
        index=(backends.index(default) if default in backends else 0),
    )
    gate = st.sidebar.checkbox("Confidence gate", value=True, disabled=mode != "hybrid_rerank")
    threshold = st.sidebar.slider("Gate threshold", 0.0, 1.0, 0.5, 0.01, disabled=not gate)
    return {"mode": mode, "top_k": top_k, "backend": backend, "gate": gate, "threshold": threshold}


def render_sources(sources: list[dict[str, Any]], key: str) -> None:
    for s in sources:
        n = s["n"]
        st.markdown(f'<a name="{key}-source-{n}"></a>', unsafe_allow_html=True)
        label = f"[{n}] {source_caption(s)}" + ("  (cited)" if s.get("cited") else "")
        with st.expander(label, expanded=False):
            st.write(s["text"])
            meta = {k: s.get(k) for k in ("doc_id", "section", "date", "circular_number", "score")}
            st.json(meta, expanded=False)
            if s.get("url"):
                st.markdown(f"[Open on rbi.org.in]({s['url']})")
            if s.get("pdf_url"):
                st.markdown(f"[PDF]({s['pdf_url']})")


def render_answer(resp: dict[str, Any], key: str) -> None:
    cited = {c["n"] for c in resp.get("citations", [])}
    st.markdown(linkify_citations(resp["answer"], cited, anchor=f"{key}-source"))
    if resp.get("refused"):
        st.info(f"Refused: {resp.get('refusal_reason')}")
    conf = resp.get("confidence")
    st.caption(
        f"{resp.get('backend') or '-'}:{resp.get('model') or '-'} | confidence "
        f"{conf if conf is None else round(conf, 3)} | tokens "
        f"{resp['usage']['input_tokens']}+{resp['usage']['output_tokens']} | cost "
        f"${resp['cost']['usd']:.5f} | "
        + ", ".join(f"{k} {v:.0f} ms" for k, v in resp["timings_ms"].items())
    )
    render_sources(resp.get("sources", []), key)


def main() -> None:
    st.title("Ask the RBI notifications")
    st.caption('Answers cite their sources; low-confidence questions get "I don\'t know".')
    opts = sidebar()
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    for i, turn in enumerate(history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["response"], f"h{i}")

    question = st.chat_input("e.g. How often must banks update KYC for high-risk customers?")
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        text = ""
        final: dict[str, Any] | None = None
        payload = {"question": question, **opts}
        for event, data in stream_query(API_URL, payload):
            if event == "token":
                text += data["text"]
                placeholder.markdown(text + " ...")
            elif event == "done":
                final = data
            elif event == "error":
                placeholder.error(data.get("detail", "error"))
        if final is not None:
            placeholder.empty()
            render_answer(final, f"h{len(history)}")
            history.append({"question": question, "response": final})


main()
