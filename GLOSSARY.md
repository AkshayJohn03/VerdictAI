# VerdictAI — Glossary

Every keyword this repo uses, in plain English, grouped by theme. Each entry:
what it means in one to three sentences, then why it matters *in this repo*.
Written for the owner of the repo who is learning AI — no prior jargon assumed.
The companion whiteboard lecture (`brag-output/brag.mp4`) defines these same
terms on screen.

---

## 1. Judging — how answers get graded

**LLM (large language model).**
A neural network trained on enormous amounts of text that predicts and
generates text. Ask it a question and it writes an answer, one piece of text at
a time.
*Why it matters here:* the LLM plays two roles — it writes the answers being
graded, and (as the judge) it grades other answers. VerdictAI treats both roles
as fallible and measurable.

**judge (LLM-as-judge).**
Using one AI model to grade the outputs of another — "rate this answer 1–5 for
helpfulness", or "which of these two answers is better?". It's popular because
human grading doesn't scale.
*Why it matters here:* the whole repo exists because that judge is itself a
fallible machine. Who grades the grader? VerdictAI does.

**rubric.**
A written marking scheme: the dimensions an answer is scored on, the scale for
each, and what each score means. The opposite of "score it 1–10, vibes-based".
*Why it matters here:* `RubricJudge` decomposes grading into weighted criteria
(accuracy, evidence, tone…) scored 0–5 each, because grading one dimension at
a time is far less noisy than one holistic number.

**anchor.**
A concrete description of what a specific score on a criterion looks like —
"accuracy = 2 means: mostly right direction, but with a factual error". Anchors
pin the scale down so "a 4" means the same thing on every run.
*Why it matters here:* each rubric criterion carries 0–5 anchors, so the
judge's scores are comparable across items, models and weeks.

**evidence quote.**
A snippet copied from the answer that justifies the score, required alongside
the rationale. If the quote doesn't support the score, the verdict is wrong and
you can see it.
*Why it matters here:* every rubric verdict must carry one — it makes grades
auditable by eye, not just trustworthy by faith.

**rubric decomposition.**
Splitting one big subjective judgement ("how good is this answer?") into
several small weighted judgements on separate criteria, then combining them.
*Why it matters here:* it's the difference between "I give it a 7" and
"accuracy 4/5 × 0.5 + evidence 3/5 × 0.3 + tone 5/5 × 0.2 = 3.8" — the second
one can be checked, weighted, and argued with.

**pairwise judging.**
Instead of scoring answers separately, show the judge two answers (A and B) and
ask which is better. Comparison is something models do more reliably than
absolute scoring.
*Why it matters here:* `PairwiseJudge` runs the comparison **twice, in both
orders** — (A,B) then (B,A) — and averages. It doubles judge cost on purpose:
silent order effects are worse than double spend.

**position bias.**
The tendency to favour whichever answer was presented first (or last) — a
reading-order habit, not a quality judgement. If swapping the order flips the
verdict, the verdict was about position, not merit.
*Why it matters here:* measured directly as `position_bias_rate`; an
order-flipped item lands at 0.5 (tie) instead of wrongly 0 or 1.

**verbosity bias.**
The tendency of judges to prefer longer answers, regardless of quality —
length masquerading as substance.
*Why it matters here:* `calibration/bias.py` fits the slope of judge score
against response length and tests it with a seeded permutation, so a length
preference shows up as a number with a p-value, not a hunch.

**self-preference.**
A model rating its own writing style (or its family's) more highly — the
grader being kind to its own homework.
*Why it matters here:* bias diagnostics flag items where the judge's model
family matches a candidate's family *and* it preferred that candidate. The
production note in the README: keep the judge model from a different family
than the model under test.

**judge ensemble.**
Several judges (rubric, pairwise, reference…) voting on the same answer, with
weights. When they disagree sharply, the item is flagged for a human.
*Why it matters here:* `JudgeEnsemble` produces the weighted verdict plus a
disagreement flag — disagreement is the cheapest signal for "send this one to
a person".

**self-consistency.**
Ask the same judge the same question k times and look at how much the answers
vary. High variance means the judge itself is unsure.
*Why it matters here:* `SelfConsistency` uses the variance across k samples as
a confidence measure, deciding which verdicts need a second opinion.

**temperature.**
The sampling dial that controls randomness in an LLM: 0 means "always pick the
most likely text" (deterministic), higher means more variety.
*Why it matters here:* grading runs at temperature 0 by default — a grader
should give the same answer twice; self-consistency deliberately turns the
dial up to *measure* the judge's uncertainty.

**reference judge.**
A judge that compares an answer against a known-correct golden answer instead
of an opinion — no LLM needed.
*Why it matters here:* `ReferenceJudge` computes token-F1 against the golden
reference. It's deterministic, free, and offline, so it's the default judge in
tests and CI.

**token-F1.**
An overlap score between two texts: the harmonic mean of precision (how much
of the answer's words appear in the reference) and recall (how much of the
reference appears in the answer). 1.0 = perfect word-level match.
*Why it matters here:* it's the offline score the regression pipeline replays
against — a simulated degraded model shows up as a lower F1, with no API call.

**heuristic judge.**
A rule-based fallback grader using keyword, length and format signals instead
of a model.
*Why it matters here:* `HeuristicJudge` keeps the whole pipeline runnable with
no API key and no network — plumbing that makes 177 tests possible, and
honestly labelled as plumbing, not intelligence.

**EchoMockClient.**
A deterministic fake LLM client that echoes scripted responses instead of
calling the network.
*Why it matters here:* every test and offline demo runs through it — same
input, same output, forever. Determinism is what makes the numbers trustworthy.

---

## 2. Calibration — checking the grader against humans

**human label set.**
A file of items where a human and the judge both graded the same answers — the
ground truth you compare the machine against.
*Why it matters here:* `HumanLabelSet` loads it as JSONL with per-line
validation; every calibration number downstream is only as good as this file.

**observed agreement (p_o).**
The raw fraction of items two graders scored identically. Simple — and
misleading, because two coin-flipping graders still agree a lot by luck.
*Why it matters here:* kappa starts from p_o and then subtracts the luck; the
repo computes it inside `cohen_kappa` so you can see both parts.

**Cohen's kappa.**
Agreement between two graders *beyond luck*:
kappa = (p_o − p_e) / (1 − p_e), where p_e is the agreement chance would
produce from their scoring habits. 0 ≈ a coin flip, 1 = perfect.
*Why it matters here:* the repo's worked example pins it: judges scoring
[0,0,1,1] vs humans [0,1,1,1] → p_o = 0.75, p_e = 0.5, kappa = 0.5 — tested
against that exact hand-computed number.

**weighted kappa (quadratic weighted kappa, QWK).**
Kappa for ordered scales that gives partial credit: rating a 4 as a 5 is a
small miss, rating it as a 1 is a disaster. Quadratic weighting penalises
errors by the square of their distance.
*Why it matters here:* judge scores are 0–5 and ordinal, so QWK is the honest
agreement number; the fixture pins it at 2/7 on a hand-worked example.

**Spearman correlation.**
A rank correlation: do two graders put items in the same *order*, even if
their absolute scores differ? Ranges from −1 (reversed) through 0 (no
relationship) to 1 (same order).
*Why it matters here:* the implementation is tie-aware (many judged items tie,
especially in pairwise scores), and it's pinned by a test at 3/√10.

**MAE (mean absolute error).**
The average size of the gaps between two graders' scores, ignoring direction.
An error of 0.3 points means "on average, off by a third of a point".
*Why it matters here:* the calibration report prints MAE before and after
isotonic calibration — the drop is the whole point, and a rise would expose
overfitting.

**within-1 accuracy.**
The fraction of items where the judge landed within one point of the human —
a forgiving "close enough" measure alongside exact agreement.
*Why it matters here:* it survives in `agreement_summary` because a judge that
is consistently off by a hair is usable after calibration; one that scatters
is not.

**isotonic regression.**
Fitting the best **monotone** (never-decreasing) mapping from judge scores to
human scores — "curving the grades" with no assumed shape. Monotone makes
sense: a higher judge score should never map to a lower human score.
*Why it matters here:* `calibration/isotonic.py` implements it by hand to move
judge scores onto the human scale — saturation at 4–5 and dead zones near 0
are exactly the kinks a single sigmoid can't fix.

**PAV algorithm (pool-adjacent-violators).**
The classic O(n) way to compute that isotonic fit: walk the data in order;
whenever a block would be lower than the one before it, pool the violators and
replace both with their weighted average until the sequence stops decreasing.
*Why it matters here:* implemented from scratch with a stack (~20 lines), and
pinned by tests — including the interpolation behaviour (a query of 2.5 between
fitted blocks resolves to 3.75 in the fixture).

**monotone step function.**
The shape PAV produces: a staircase that only goes up. Between observed data
points there are gaps; VerdictAI interpolates linearly in the gaps and clamps
outside the range.
*Why it matters here:* it's the calibration curve's actual shape — honest about
only knowing what the labels showed.

**calibration curve.**
The plot/idea of "judge score in → human score out". A perfectly calibrated
judge lies on the diagonal; a judge that says 5 when humans say 3.5 bulges.
*Why it matters here:* `CalibrationLayer` is that curve as working code — every
judge score afterwards is re-expressed on the human scale.

**Platt scaling.**
The parametric alternative: fit one sigmoid curve to the judge-vs-human
relationship. Cheap and fine below ~50 labels — but it can only stretch and
squeeze globally, never kink.
*Why it matters here:* the README's design note explains why PAV was chosen
over it; the trade-off (isotonic needs more labels, overfits below ~30) is
stated and reported rather than hidden.

**active labeling loop.**
Human attention is scarce, so rank the *unlabeled* items by how informative
their labels would be — big |judge − human| gaps and ensemble disagreement
first — and emit the queue for the next labeling round.
*Why it matters here:* `ActiveLabelLoop` writes the prioritised CSV, so each
round of human grading buys the most calibration improvement per label.

---

## 3. Datasets — building an exam that can't be gamed

**golden dataset.**
A versioned, curated set of test questions with known-good expected answers —
the fixed exam every model version must sit.
*Why it matters here:* `EvalSetGenerator` builds one over a topic × persona ×
difficulty grid, and every regression run scores the *same* items so versions
are comparable.

**coverage matrix.**
A grid of every kind of question you promised to test (topics × personas ×
difficulty × edge cases) with a cell marked when an item fills it. Empty cells
are visible gaps.
*Why it matters here:* the generator fills the matrix on purpose and a test
asserts full coverage — "the exam covers everything we said it would" is a
checked claim, not a hope.

**topic × persona × difficulty grid.**
The generator's recipe for variety: cross real topics with user personas
(a terse expert, a confused beginner…) with difficulty levels, so the exam
isn't 40 near-identical questions.
*Why it matters here:* variety is the antidote to a benchmark that one good
prompt can game — the grid forces breadth.

**edge case (adversarial item).**
Deliberately nasty items: prompts with hidden instructions, unicode look-alike
tricks, or answers that try to manipulate the grader.
*Why it matters here:* the generator plants them on purpose — a judge that
can be talked out of its rubric by a sneaky answer fails here, safely offline.

**data leakage.**
Test material ending up in the training material — the exam leaking into the
textbook. The model looks brilliant because it memorised the answers.
*Why it matters here:* decontamination exists to prevent exactly this between
your corpus and your eval set.

**decontamination.**
Stripping any eval item too similar to the corpus the model may have seen —
refusing to ask questions the model could have memorised.
*Why it matters here:* `gen-dataset --corpus … --contamination-n 8` drops items
with an 8-gram overlap against the corpus, and a test plants an 8-gram overlap
and asserts it's caught.

**n-gram overlap.**
An n-gram is a run of n consecutive words; overlap means an n-word run appears
in both texts. It catches verbatim copying, not paraphrase.
*Why it matters here:* n=8 is the default because eight shared words in a row
is a quote, not a coincidence. The README says plainly: paraphrased leakage
gets past it — known limitation.

**JSONL.**
One JSON object per line — the format of the datasets, labels and snapshots.
*Why it matters here:* line-per-record means appending a run never rewrites
old data, and any two runs can be diffed with ordinary text tools.

**SHA-256 manifest.**
A sidecar file listing the SHA-256 hash (a 64-character fingerprint that
changes if even one byte changes) of each dataset file.
*Why it matters here:* every snapshot records the dataset hash, so a score
change is never ambiguous about *which exam version* produced it.

**dataset versioning (changelog).**
Datasets are written as `dataset-v1.jsonl` + `manifest-v1.json` with an entry
appended to `CHANGELOG.md` — versions like code.
*Why it matters here:* "the model got worse" is only a meaningful sentence if
the exam it sat didn't silently change underneath it.

---

## 4. Regression — catching a model getting worse

**eval run.**
One complete scoring pass: a candidate model version answers every item of one
dataset version, and every answer gets a judge score.
*Why it matters here:* the unit of evidence. Runs are what snapshots store,
what drift compares, and what the gate decides about.

**model adapter.**
The pluggable skin that turns "an item" into "a model's answer" — a live
endpoint, a replay, or a mock.
*Why it matters here:* `ModelAdapter` is a protocol, so the whole regression
machinery runs unchanged against a real model or a simulation.

**replay adapter (OfflineReplayAdapter).**
An adapter that doesn't call a model at all: it replays the golden references,
deterministically degraded to a chosen quality level — a simulation of a
model having a bad day.
*Why it matters here:* it lets the gate's behaviour be *planted and proven*:
degrade to 0.75 quality, watch the gate fire. It's a simulation, not a model —
and the README says so.

**offline mock test.**
A test where every network touch is replaced by a scripted fake, so the whole
behaviour is exercised with no API key, no DNS, no cost, no flakes.
*Why it matters here:* all 177 tests are offline; CI needs no secrets. The
statistics are still real — pinned to hand-computed fixtures.

**baseline.**
The reference run you compare against — usually the last version you trusted.
*Why it matters here:* the gate's question is always "worse than *baseline* by
how much?" — `--baseline v1 --candidate v2` on the CLI.

**snapshot.**
One append-only JSONL record of a complete eval run: model version, dataset
hash, timestamp, and every per-item score. Nothing is ever edited or deleted.
*Why it matters here:* `SnapshotStore` makes runs bit-comparable across time —
the same item scored in March and in June lines up by `item_id`.

**paired delta.**
For each item, the baseline score minus the candidate score. Pairing by item
cancels item difficulty — the comparison is "did *this item* get worse?",
averaged over the same items.
*Why it matters here:* pairing is where most of the variance reduction comes
from; the README's design note calls it out as the reason to prefer paired
tests over a plain t-test.

**bootstrap confidence interval.**
Resample the observed deltas thousands of times (with replacement, a fixed
seed) and see what range the mean falls in 95% of resamples. If that range
excludes zero, the drop is very unlikely to be luck.
*Why it matters here:* score deltas are bounded and bimodal, not
bell-shaped — `paired_bootstrap_ci` makes no normality assumption, 2,000
resamples, deterministic per seed.

**Wilcoxon signed-rank test.**
A rank-based paired test: do the deltas lean negative more than symmetry
allows? Robust to outliers; exact enumeration for small samples (≤ 16), normal
approximation with tie correction above that.
*Why it matters here:* it complements the bootstrap at small n, where
percentile CIs are known to undercover — the honest-limits section says so.

**p-value.**
The probability of seeing a result at least this extreme if nothing had
actually changed. Small p-value = the pattern is hard to blame on luck.
*Why it matters here:* the verbosity-bias permutation test and Wilcoxon both
report one; the gate prints `wilcoxon_p` in its alert payload as evidence.

**effect size (Cliff's delta).**
How *big* the change is, independent of sample size: the fraction of positive
minus negative pairwise comparisons among deltas. Ranges −1 to +1; 1.0 means
every single item got worse.
*Why it matters here:* the gate's noise guard — on 5,000 items a 0.01 shift
can exclude zero, but `min_effect` demands a *meaningful* delta before a
regression blocks a deploy.

**drift detection.**
The whole comparison step: take two snapshots, compute paired deltas, then
bootstrap CI + Wilcoxon + Cliff's delta, and decide: regressed / improved /
no change.
*Why it matters here:* `DriftDetector` is where "is the drop real or luck?"
gets a statistical answer instead of an opinion.

**regression gate.**
The decision wired to the build: if the CI lower bound clears the threshold
**and** the effect size clears the floor, the candidate has regressed → fail
the run, print a markdown summary, emit a webhook-ready JSON alert.
*Why it matters here:* `run_gate` returns `exit_code` 1 on regression and 0
otherwise — the exact contract a CI system consumes.

**CI (continuous integration).**
The automation that runs the test suite and checks on every proposed change,
blocking merges that fail.
*Why it matters here:* the gate is designed to live there — `verdict regress
--gate` in CI is the alarm that stops a degraded model from shipping.

**exit code.**
The number a program returns to the shell: 0 = success, non-zero = failure.
CI systems gate on it mechanically.
*Why it matters here:* the repo's contract is explicit — 1 = regression, 0 =
pass (or report-only mode), 2 = bad input. Tests assert all three, and the
end-to-end demo ends with a real exit 1.

---

## 5. Serving — the same harness over HTTP

**FastAPI.**
A Python framework for building HTTP APIs with typed request models and async
handlers.
*Why it matters here:* `service.py` is a thin async skin over the offline
harness — same runners, same gate, reachable from any CI system that can POST.

**async eval run (202 Accepted).**
Submitting work that takes a while: the API accepts the request, returns 202
with a `run_id` immediately, and executes the run in a background thread.
*Why it matters here:* `POST /v1/eval-runs` answers in milliseconds and the
client polls `GET /v1/eval-runs/{run_id}` for status
(queued/running/succeeded/failed) — a long eval never holds an HTTP
connection open.

**webhook.**
A "call me back" URL: when an event happens, the server POSTs the event to a
URL you registered, instead of you polling for it.
*Why it matters here:* register one on an eval run and the service POSTs the
completion event (including the gate's alert JSON) to your Slack/PagerDuty
glue.

**HMAC signature.**
A message authentication code: hash the request body with a shared secret
(SHA-256) and send the result in a header. A receiver with the secret can
recompute it and prove the body wasn't forged or altered.
*Why it matters here:* webhooks carry `X-VerdictAI-Signature: sha256=<hex>`,
GitHub-style — verify over the raw bytes, so a fake "all clear" can't be
injected into your pipeline.

**SHA-256 at rest.**
Storing the *hash* of a secret instead of the secret itself. Hashes can be
compared but not reversed into the original key.
*Why it matters here:* the service keeps only SHA-256 hashes of API keys — a
database peek never reveals a usable credential.

**API-key auth.**
Simple machine authentication: the client sends a pre-shared key in a header
(`X-VerdictAI-Key`); the server checks it against the stored hashes.
*Why it matters here:* unset `VERDICTAI_API_KEYS` and auth is off for local
dev; set it and every route except `/health` demands a valid key.

**constant-time comparison.**
Comparing secrets in a way that takes the same time regardless of *where* the
first difference is, so an attacker can't learn the key bit by bit from
timing.
*Why it matters here:* key comparison uses `secrets.compare_digest` — a
one-line habit that closes a classic side channel.

**correlation ID.**
An identifier a client attaches to a request (`X-Correlation-ID`) that the
server echoes on every response and log line, tying one journey together
across systems.
*Why it matters here:* when a CI job and the eval service exchange many
requests, the correlation ID is how you grep one run's story out of both logs.

**liveness probe.**
A cheap endpoint (`GET /health`, no auth) that answers "is the process up?"
— what load balancers and orchestrators poll.
*Why it matters here:* it's deliberately key-free, because the question is
"is it alive?", not "who's asking?".

**TTL (time-to-live).**
How long a piece of server-side state is kept before it expires.
*Why it matters here:* run state lives in memory for 1 hour with a 100-run
cap (oldest evicted) — the service is a thin skin, not a run database, and
the README is upfront about it.

**retry with exponential backoff.**
On a transient failure (rate limit, 5xx, network blip), wait a little, retry,
and double the wait each time instead of hammering.
*Why it matters here:* the OpenAI-compatible client retries 429/5xx with a
0.5 s backoff base, configurable via `VERDICTAI_MAX_RETRIES`.

**JSON mode.**
Requesting that the model return well-formed JSON, then validating and
clamping the payload before trusting it.
*Why it matters here:* rubric verdicts arrive as JSON — strict payload
validation is what keeps a chatty model from corrupting a score file.

---

*Every number in this glossary is measured in the repo's own test suite —
177 tests, offline, under two seconds. Nothing here is trusted without a
number.*
