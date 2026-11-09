# Homework: Evaluating a Recipe Assistant

Week 6 (Evaluation) of the AI Engineering Buildcamp — offline evaluation of a
recipe-answering agent: scenario design, human labeling, an LLM judge, and
measuring/fixing judge alignment.

The agent (`recipe_agent.py`) answers cooking questions from a fixed collection of
15 recipes (`recipes.json`) using two tools (`recipe_tools.py`: `search_recipes`,
`get_recipe`), all provided by the course starter (Alexey Grigorev's gist).

## The evaluation pipeline

| file | what it does |
|---|---|
| `scenarios.csv` | 20 evaluation scenarios (5 from the homework + 15 covering phrasings, substitutions, dietary traps, dataset gaps, out-of-scope, edge cases) |
| `run_scenarios.py` | runs the agent on every scenario, saves outputs + tokens + cost to `results.json` (`uv run python run_scenarios.py [input.csv] [output.json]`) |
| `human_labels.csv` | my manual good/bad labels with the reason for every scenario (Q3) |
| `judge.py` | LLM judge (structured output: reasoning before label); prompt improved after comparing against my labels (Q6) (`uv run python judge.py [results.json]`) |
| `results_judged_v1.json` | judge output with the **original** course prompt (used for Q4/Q5) |
| `results_judged.json` | judge output with the **improved** prompt (used for Q6) |
| `compute_metrics.py` | accuracy / precision / recall + disagreements vs my labels |
| `gen_synthetic.py` | Q7 bonus: structured-output generation of 75 synthetic questions (15 recipes × 5 types) → `synthetic_scenarios.csv` |
| `synthetic_results.json`, `synthetic_results_judged.json` | agent + judge output on the synthetic set |

## Results

| metric | original judge | improved judge |
|---|---|---|
| accuracy | 0.80 | 0.95 |
| precision (for "bad") | 0.667 | 0.889 |
| recall (for "bad") | 1.00 | 1.00 |
| disagreements with human labels | 4 | 1 |

- Human labels: **8 bad / 20**; original judge: **12 bad**; improved judge: **9 bad**.
- Almond-milk substitution scenario: human **bad**, judge **bad** (agreement) —
  the agent invented substitution advice with zero tool calls.
- All 4 original disagreements were judge false positives: it penalized honest
  declines that *offered* further help, a correct "no calorie data in the
  collection" answer, and recipe-grounded answers it could not verify. The
  improved prompt separates *offered* follow-ups from *asserted* content and
  stops treating unverifiability as evidence of fabrication.

## Setup

```bash
uv sync                 # minsearch, pydantic-ai-slim[openai], python-dotenv
# .env needs OPENAI_API_KEY
uv run python run_scenarios.py
uv run python compute_metrics.py results_judged_v1.json
uv run python judge.py results_judged.json
```

Two starter files needed `load_dotenv()` (they read the API key from `.env`);
otherwise the course code runs unmodified.
