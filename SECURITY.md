# Security Policy — VerdictAI

VerdictAI handles evaluation artifacts: golden datasets, judge prompts, human
labels, and model snapshots. Threat model and defenses:

| Threat | Defense |
|---|---|
| Golden-set leakage into training data | n-gram decontamination with evidence in `datasets/`; sha256 manifest per dataset version |
| Judge manipulation via crafted responses (e.g., "rate this 5/5" embedded in candidate output) | Rubric judges score against criteria with quoted evidence; pairwise double-order de-biasing; position-bias rate is a reported metric — a response that games one order is caught by the order flip |
| Human-label tampering | Label sets are append-only JSONL with per-item provenance; calibration reports hash their inputs |
| Regression-gate tampering | Snapshots are append-only with model-version metadata; the CI gate re-computes statistics from raw scores, never trusts cached verdicts |
| Secrets | No keys in code; `VERDICTAI_API_KEY` env-only; offline mode requires nothing |

Report issues marked `security`.
