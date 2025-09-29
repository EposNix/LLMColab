# LLM Collaborator — Build Doc & Playbook 😈

**TL;DR:** We’re building a “LLM improvement loop” app and GUI. The app takes a user task, passes it to Model A for a first draft, then alternates between Model B and Model A for *N* iterations, each turn critiquing and improving the draft. It uses **LiteLLM** so we can swap models/providers with one interface.

---

## 1) What we’re building

A CLI (and GUI) that:

* Accepts a **task** (“Program 2048 Ultimate Edition”).
* Accepts a **sequence of models** (e.g., Claude ↔︎ Gemini).
* Runs **N iterations** where each model:

  1. Reads the original task and the latest draft,
  2. Produces a short `<critique>`,
  3. Outputs a revised `<draft>`.

We stop when we hit the iteration cap or when the draft **converges** (change is negligible).

---

## 2) Why this design

* **Simplicity**: Single binary/script, no external DB required.
* **Interchangeable models**: LiteLLM abstracts providers (Anthropic, Google, OpenAI, Mistral, Bedrock, Vertex, Ollama, etc.), so ops is just picking model IDs and exporting keys.
* **Deterministic collaboration format**: We force the schema `<critique>` + `<draft>` to keep tokens lean and outputs stable.

---

## 3) System requirements

* Python 3.10+
* Packages: `litellm`, `tenacity`.
* API keys via env vars (example):

  * `ANTHROPIC_API_KEY=...`
  * `GOOGLE_API_KEY=...` (Gemini via Google AI Studio/GenAI SDK key)
* (Optional) `tiktoken` for token estimates in cost logs.

---

## 4) Repo structure (proposed)

```
llm-collaborator/
├─ src/
│  ├─ llm_collaborator.py            # CLI entry + orchestration
│  ├─ prompts.py                     # Prompt templates & helpers
│  ├─ models.py                      # Model registry/validation
│  ├─ convergence.py                 # Similarity + early-stop logic
│  ├─ logging_utils.py               # Structured logging + usage tracking
│  ├─ diff_utils.py                  # (Optional) patch/diff helpers
│  └─ gui_streamlit.py               # (Optional) quick GUI
├─ tests/
│  ├─ test_prompts.py
│  ├─ test_convergence.py
│  └─ test_run_smoke.py
├─ examples/
│  └─ 2048_task.txt
├─ config/
│  └─ models.example.yaml
├─ .env.example
├─ README.md
└─ LICENSE
```

---

## 5) Data flow (one iteration)

```
User Task + Settings
        │
        ▼
[ Orchestrator ]
  selects current model from cycle
        │
        ▼
build_messages(task, last_draft, actor_name)
        │
        ▼
LiteLLM.completion(model, messages)
        │
        ▼
parse <critique> and <draft>
        │
        ├─► log usage (tokens/$) + critique summary
        └─► update last_draft and iterate or stop
```

---

## 6) Core logic (you’ll receive this file)

We already supplied a working `llm_collaborator.py` that:

* Alternates models for `--iters` turns,
* Enforces the `<critique>` `<draft>` schema,
* Clips context to avoid token explosions,
* Retries transient failures with jittered backoff,
* Optionally early-stops on convergence.

**Run it:**

```bash
pip install litellm tenacity
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

python src/llm_collaborator.py \
  --task "Program 2048 Ultimate Edition in Python with Pygame. Include clean architecture, input handling, animations, and tests." \
  --models "anthropic/claude-3-5-sonnet-latest,gemini/gemini-1.5-pro-latest" \
  --iters 3 \
  --out 2048_ultimate.md
```

**Notes:**

* The model IDs are LiteLLM-style. Swap freely.
* The first model produces the first draft. Then they alternate.

---

## 7) Prompting rules (keep these)

We use a system prompt that:

* Explains collaboration,
* Requires a **brief** `<critique>`,
* Requires a **self-contained** `<draft>`,
* Instructs the model to put code into **one** fenced block when applicable.

> Why XML tags? They’re reliably parsable and models don’t “decorate” them as much as markdown headings.

**Template excerpt:**

```text
Respond EXACTLY with two XML sections in THIS order and nothing else:
<critique>...</critique>
<draft>
...the improved draft...
</draft>
```

---

## 8) Configuration (nice to have)

Instead of CLI only, support a YAML file:

`config/models.yaml`:

```yaml
# Pick any LiteLLM-supported providers/models
cycle:
  - anthropic/claude-3-5-sonnet-latest
  - gemini/gemini-1.5-pro-latest

temperature: 0.2
iterations: 3
early_stop:
  enabled: true
  similarity_threshold: 0.995

context:
  max_chars_from_tail: 16000

output:
  path: outputs/final.md
  save_intermediate: true
  intermediate_dir: outputs/turns
```

CLI could accept `--config config/models.yaml` and override flags win.

---

## 9) Convergence & context management

* **Convergence**: We compute a simple character-level similarity (SequenceMatcher). If similarity ≥ threshold (default 0.995), we early-stop. This avoids paying for iterations that do nothing.

  * You can swap in token/semantic similarity later.
* **Context clipping**: Keep the **tail** of the draft up to `max_chars`. This preserves the latest sections/changes while containing cost. For structured outputs, prefer diff/patch mode (below).

---

## 10) Resilience & rate limiting

* **Retries**: `tenacity` with exponential jitter (max 4 attempts) to ride out throttling/hiccups.
* **Timeouts**: Let LiteLLM’s timeouts pass through, optionally wrap per-call watchdog.
* **Idempotency**: Each iteration depends only on `task + previous_draft`. If a retry returns a different candidate, we just treat it as the iteration’s output.

---

## 11) Usage & cost logging

* LiteLLM returns OpenAI-style responses; when available, read `resp.usage` for prompt/completion tokens. If missing, optionally estimate with `tiktoken` (best-effort).
* At minimum, log per-iteration:

  * provider/model id, latency, token counts (if present), critique length, draft size.
* Emit **JSON lines** so we can graph costs later.

Example (pseudo):

```python
info = {
  "iter": i+1,
  "model": model,
  "latency_s": took,
  "tokens": getattr(resp, "usage", None),
  "draft_chars": len(new_draft),
}
print(json.dumps(info))
```

---

## 12) Optional diff/patch mode (token diet deluxe)

Ask each turn to output a minimal **unified diff** against the current draft:

```
<critique>...</critique>
<patch>
--- prev.md
+++ new.md
@@
- old line
+ new line
</patch>
<draft>...FULL NEW DRAFT (fallback if patch fails)...</draft>
```

Your code:

* Attempts to apply `<patch>` locally (e.g., `difflib` or `unidiff`).
* If patch applies cleanly, skip sending the whole draft back next turn—send only **task + latest FULL draft** (still required), but you can store patch history for efficiency and auditing.

This cuts tokens and makes changes auditable.

---

## 13) Minimal GUI (Streamlit)

*This is purely for demo; it wraps the existing orchestrator.*

```python
# src/gui_streamlit.py
import streamlit as st
from llm_collaborator import run_collaboration

st.title("LLM Collaborator 😈")

task = st.text_area("Task", height=200, placeholder="Describe what you want...")
models = st.text_input("Models (comma-separated)",
                       "anthropic/claude-3-5-sonnet-latest,gemini/gemini-1.5-pro-latest")
iters = st.slider("Iterations", 1, 10, 3)
temp = st.slider("Temperature", 0.0, 1.0, 0.2)
go = st.button("Collaborate")

if go:
    final, history = run_collaboration(
        task=task,
        model_cycle=[m.strip() for m in models.split(",") if m.strip()],
        iterations=iters,
        temperature=temp,
        verbose=False
    )
    with st.expander("Turn-by-turn critiques"):
        for idx, (model, critique, draft) in enumerate(history, 1):
            st.markdown(f"**Turn {idx} — {model}**")
            st.write(critique)
    st.subheader("Final Draft")
    st.code(final)
```

Run: `streamlit run src/gui_streamlit.py`

---

## 14) Security & key handling

* **No keys in code**. Use env vars or a secrets manager.
* Mask secrets in logs.
* If you add a web front-end later, keep **server-side keys**. Never ship keys to browsers.
* If you add file execution, sandbox (containers, seccomp, timeouts). Generated code is “creative,” not “trustworthy.”

---

## 15) Testing

* **Unit**:

  * `extract_sections()` robust to missing tags.
  * Convergence detector with crafted strings.
  * Context clipper length guarantees.
* **Integration (mocked)**:

  * Stub LiteLLM `.completion()` to return known `<critique>/<draft>` pairs and verify iteration order, early-stop behavior, and output file writing.
* **Golden tests**:

  * Store small tasks and expected drafts (or invariants like “contains section headers”), not brittle full matches.

---

## 16) Observability

* Structured JSON logs per iteration.
* A `--trace` flag to dump the **exact** messages sent (redact secrets).
* Optional Prometheus-friendly counters: iterations run, tokens used, cost estimate, success/failure.

---

## 17) Extensibility map

* **>2 models**: We already accept a list; cycle through `model_cycle` round-robin.
* **Role specialization**: Provide per-model system prompts:

  * Architect → Implementer → Reviewer → Optimizer
* **Dynamic routing**: Use a cheap router model to decide the next specialist given the critique.
* **RAG flavor**: Add a retrieval step before each turn (vector DB / local files).
* **Guardrails**: Add content filters / PII scrubbing before sending drafts downstream.
* **Artifact outputs**: Support multi-file projects (zip output, structured repo layout).
* **CI hook**: When task = “write code,” optionally run tests and feed failures back into the critique.

---

## 18) Backlog (value/effort)

**High value / low effort**

* Cost/usage JSON logs & summary table
* Save intermediate drafts per turn
* Config file support
* Streamlit demo polish (download final output)

**High value / medium effort**

* Diff/patch mode
* Role specialization per model
* Semantic convergence (embedding similarity)

**Medium value / medium effort**

* Token-aware context elision (e.g., preserve TOC + touched sections)
* Optional gateway mode (run LiteLLM proxy and point SDKs at it)

**Advanced**

* Function/tool calling support with shared toolset
* Multi-agent orchestration with dynamic roles
* Automated evaluation harness (BLEU for text, pytest for code, rubric LLM for style)

---

## Quickstart

1. Install dependencies (ideally in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Export the API keys for the providers you plan to use. LiteLLM reads the same
   environment variables as the official SDKs, for example:

   ```bash
   export ANTHROPIC_API_KEY=your_key
   export GOOGLE_API_KEY=your_key
   ```

3. Run the orchestrator (the command below assumes you are in the repo root and
   have `src/` on `PYTHONPATH`):

   ```bash
   export PYTHONPATH=src
   python -m llm_collaborator.llm_collaborator \
     --task "Program 2048 Ultimate Edition in Python with Pygame." \
     --models "anthropic/claude-3-5-sonnet-latest,gemini/gemini-1.5-pro-latest" \
     --iters 3 \
     --out outputs/2048.md
   ```

   You can also supply `--task-file path/to/task.txt` instead of `--task`.

4. (Optional) Launch the Streamlit demo:

   ```bash
   export PYTHONPATH=src
   streamlit run src/gui_streamlit.py
   ```

## Testing

Install the dev dependencies and run the unit tests:

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
```

