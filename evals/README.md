# Judge-quality evals

The **keystone** of the roadmap (Wave 1). These turn each documented `INSIGHTS.md`
incident into a runnable case with a **known-correct gate verdict**, so judge
quality stops being a vibe and becomes a number you can compare across models and
guard against on prompt changes (the recurring lesson of INSIGHTS #7: a judge
tuned for one model hides a boundary that shifts on swap).

## Two layers (don't confuse them)

| | Where | LLM? | CI? | Measures |
|---|---|---|---|---|
| **Deterministic logic** | `tests/test_*` | no | yes | tiling, coverage, count, parsing — correct by construction |
| **Judge quality** (this dir) | `evals/` | **yes** | **no** | do the LLM gates reach the right verdict? |

The judge-quality evals call real models, so they need an API key and cost money —
that's why they live outside `pytest`/CI and are run on demand.

## Run

```bash
# from repo root, with the venv active and a key in .env
python evals/run_evals.py                     # all cases, JUDGE_MODEL, 1 sample each
python evals/run_evals.py --list              # list cases, run nothing (no key needed)
python evals/run_evals.py --model gpt-4o-mini # score a specific model
python evals/run_evals.py --repeats 5         # 5 samples/case — the judge is noisy
python evals/run_evals.py --only syrniki      # one case
python evals/run_evals.py --kind ingest       # only ingest-gate cases
```

`--model` overrides `JUDGE_MODEL` for the run, so this is how you answer
"how smart a model do we actually need?" (ROADMAP #9): run the same set against a
free model and a paid one and compare the scorecards.

### Comparing a free (OpenRouter) model

`--model` only changes the model string; the client's endpoint/key come from the
configured provider. The runner infers the provider from the slug (a namespaced or
`:free` slug → OpenRouter) or you can force it with `--provider`. Testing a free
model therefore needs an **`OPENROUTER_API_KEY` in `.env`** (free at
<https://openrouter.ai/keys>) — without it the runner tells you so and exits,
rather than sending an OpenRouter slug to OpenAI:

```bash
python evals/run_evals.py --model meta-llama/llama-3.3-70b-instruct:free --repeats 3
python evals/run_evals.py --model gpt-4o-mini --provider openai      # force a provider
```

A case that errors (bad model, network blip) is isolated: it's marked `ER`, the
other cases still score, and the run exits non-zero — one flaky call never loses
the whole sweep.

## Reading the scorecard

Positive class = **"the case contains a real problem"** (`expect_pass=False`).

- **recall** — of the real problems, how many the judge **caught**. A miss (`FN`)
  is the dangerous mode: a flaw ships gate-passed.
- **precision** — of what it flagged, how much was real. Over-flagging (`FP`) sends
  good output back and burns retries (e.g. the INSIGHTS #6 truncation false positive).

Per-case markers: `OK` all samples correct · `~~` some · `!!` all wrong (or a
forbidden-term tripwire fired). The runner exits non-zero if any `FP`/`FN`
occurred, so it can gate a model/prompt change.

## Cases (`cases.py`)

Each `Case` declares the gate, the expected verdict, and optional reason/precision
checks. Add a case whenever a new incident teaches the system something — that's
how the regression guard grows. To add one:

1. Write a `build()` returning the gate inputs (`users`/`curriculum`/`personalized`
   for personalize, `ingest` for ingest).
2. Append a `Case(...)` to `CASES` with `expect_pass` and, if useful,
   `expect_issue_substr` (reason match) / `forbid_issue_substr` (precision tripwire).
