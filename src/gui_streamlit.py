"""Minimal Streamlit front-end for the LLM Collaborator."""
from __future__ import annotations

import streamlit as st

from llm_collaborator import run_collaboration


st.set_page_config(page_title="LLM Collaborator 😈", page_icon="😈")
st.title("LLM Collaborator 😈")

st.markdown(
    """
This demo wraps the core collaboration loop. Provide a task, select a sequence of
models (comma separated LiteLLM identifiers), and choose how many turns to run.
Outputs show each critique and the final draft.
"""
)

task = st.text_area(
    "Task",
    height=200,
    placeholder="Describe what you want...",
)
models_raw = st.text_input(
    "Models (comma-separated)",
    "anthropic/claude-3-5-sonnet-latest,gemini/gemini-1.5-pro-latest",
)
iters = st.slider("Iterations", 1, 10, 3)
temp = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)
verbose = st.checkbox("Verbose logging", value=False)

go = st.button("Collaborate")

if go:
    if not task.strip():
        st.error("Please enter a task before running the collaboration loop.")
    else:
        model_cycle = [m.strip() for m in models_raw.split(",") if m.strip()]
        with st.spinner("Collaborating..."):
            try:
                result = run_collaboration(
                    task=task,
                    model_cycle=model_cycle,
                    iterations=iters,
                    temperature=temp,
                    verbose=verbose,
                )
            except Exception as exc:  # pragma: no cover - interactive surface
                st.error(f"Failed to run collaboration: {exc}")
            else:
                with st.expander("Turn-by-turn critiques"):
                    for idx, (model_name, critique, draft) in enumerate(result.history, start=1):
                        st.markdown(f"**Turn {idx} — {model_name}**")
                        st.markdown(critique)
                st.subheader("Final Draft")
                st.code(result.final_draft)
                if result.convergence:
                    if result.convergence.converged:
                        st.success(
                            f"Converged with similarity {result.convergence.similarity:.3f}"
                        )
                    else:
                        st.info(
                            f"Similarity {result.convergence.similarity:.3f} (threshold {result.convergence.threshold})"
                        )
