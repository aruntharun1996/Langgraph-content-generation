# LangGraph Content Generator — Gemini Edition

An agentic content-generation pipeline built with **LangGraph** and **Google Gemini**.
The graph generates content, evaluates it, and self-corrects until quality standards
are met — or the retry cap is reached.

---

## Graph Flow

```
START
  │
  ▼
generate_content  ◄──────────────────────────────────┐
  │                                                   │
  ▼                                                   │
evaluate_content                                      │
  │                                                   │
  ├── score >= 7.0 (approved) ──► deliver_result ──► END
  │
  └── score < 7.0 + retries_left ──────────────────────┘
```

- **generate_content** — First run: clean prompt. Subsequent runs: previous draft +
  evaluator feedback are injected so Gemini can improve the draft.
- **evaluate_content** — Scores 0–10 and returns structured JSON feedback.
- **deliver_result** — Promotes the approved (or best-effort) content to `final_content`.

---

## Project Structure

```
langgraph-content-gen/
├── main.py                          # CLI entry point
├── requirements.txt
├── .env.example                     # Copy to .env and add your API key
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── state.py                 # ContentState (Pydantic)
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── generate_content.py      # Generation node
│   │   ├── evaluate_content.py      # Evaluation node
│   │   └── deliver_result.py        # Terminal delivery node
│   ├── graph/
│   │   ├── __init__.py
│   │   └── content_graph.py         # Graph definition + routing logic
│   └── utils/
│       ├── __init__.py
│       └── llm_client.py            # Cached Gemini LLM instances
└── tests/
    ├── __init__.py
    └── test_nodes.py                # Unit tests (pytest + mocks)
```

---

## Quick Start

```bash
# 1. Clone / copy the project
cd langgraph-content-gen

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY

# 5. Run
python main.py "Write a LinkedIn post about why Java developers should learn LangGraph"
```

---

## Configuration (`.env`)

| Variable           | Default              | Description                                  |
|--------------------|----------------------|----------------------------------------------|
| `GOOGLE_API_KEY`   | —                    | **Required.** Your Gemini API key.           |
| `GENERATION_MODEL` | `gemini-2.0-flash`   | Gemini model used for content generation.    |
| `EVALUATION_MODEL` | `gemini-2.0-flash`   | Gemini model used for content evaluation.    |
| `MAX_RETRIES`      | `3`                  | Max regeneration attempts before delivery.   |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock all LLM calls — no API key needed.

---

## Using as a Library

```python
from src import content_graph, ContentState

state = ContentState(user_request="Write a tweet about Spring AI")
result = content_graph.invoke(state)

print(result.final_content)
print(f"Score: {result.evaluation_score}/10 | Iterations: {result.iteration}")
```

---

## Extending the Pipeline

| Goal                              | Where to change                             |
|-----------------------------------|---------------------------------------------|
| Change approval threshold         | `evaluate_content.py` → `APPROVAL_THRESHOLD`|
| Add a human-in-the-loop review    | Add a new node between evaluate & deliver   |
| Stream intermediate steps         | Use `content_graph.stream(state)` in main   |
| Persist state between runs        | Add a LangGraph checkpointer                |
| Add content type routing          | Extend `ContentState` + add router node     |
