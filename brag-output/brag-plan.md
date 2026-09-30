# Brag Plan: VerdictAI — "an exam grader that first learns from human teachers"

> TUTOR BRIEF OVERRIDE: this is a whiteboard explainer LECTURE (not a launch
> video, not 15–25s). `--voice` is ON. Target 5–6+ minutes. Success metric:
> the owner of the repo can explain the system to someone else afterwards.

## What is this app?

VerdictAI is a quality-control lab for AI graders: it judges model outputs
with rubric-decomposed LLM judges (plus pairwise both-orders de-biasing),
measures how well the judge agrees with humans (Cohen's kappa, QWK, Spearman),
diagnoses the judge's systematic biases (position, verbosity, self-preference),
re-curves judge scores onto the human scale with isotonic regression (PAV, by
hand), generates versioned golden datasets with n-gram decontamination, and
gates deploys on paired bootstrap CIs plus effect sizes — exit code 1 on a real
regression, silence on noise.

## The angle

A patient senior engineer at a whiteboard teaches ONE system to a smart junior
who knows almost nothing about AI. The scenario that carries the lecture:
companies now use an AI model to grade other AI models — but who grades the
grader? The grader might prefer the first answer it reads, favor long answers,
or grade its own writing kindly, and nobody checks its work. No hype, no launch
energy — calm, precise, friendly. Every technical keyword is defined on screen
in one plain sentence the moment it first appears.

## Hook (first ~35 seconds)

The awkward infinite regress, drawn on the board: answer → judge → "…and who
grades the judge?". The judge is itself a fallible machine with tics — first-read
favoritism, length love, self-kindness — and it is standing between your team
and every deploy decision. The promise: treat the grader as a first-class,
measured component — an exam grader that first learns from human teachers.

## Structure (lecture beats, per TUTOR_BRIEF section 2)

1. The real-world problem — who grades the grader? Three named tics.
2. The core idea — a quality-control lab with four stations: grade → check
   against humans → build a clean exam → gate the release. Analogy: training a
   new teaching assistant.
3. How the pieces work — one whiteboard sketch per concept:
   - the rubric (marking scheme, anchors, evidence quote)
   - pairwise judging in BOTH orders; position / verbosity / self-preference bias
   - Cohen's kappa ("agreement beyond luck"), with the kappa = 0.5 worked example
   - isotonic regression ("curving the grades") and PAV
   - golden datasets, coverage matrix, decontamination ("don't let the exam
     leak into the textbook")
   - paired bootstrap CIs and the CI gate that blocks a release (exit 1)
4. The measured numbers and what each means in plain words: 177 offline tests,
   kappa 0.5 worked example, QWK 2/7, planted ~19% degradation caught with
   exit code 1.
5. A 30-second recap the viewer could repeat to a colleague.

## Keywords defined on screen at first use

LLM · judge (LLM-as-judge) · calibration · golden dataset · regression gate ·
rubric · anchor · evidence quote · rubric decomposition · pairwise judging ·
position bias · verbosity bias · self-preference · Cohen's kappa · observed
agreement · chance agreement · weighted kappa (QWK) · Spearman correlation ·
isotonic regression · PAV algorithm · calibration curve · Platt scaling ·
coverage matrix · n-gram overlap · decontamination · baseline · snapshot ·
paired delta · bootstrap confidence interval · Wilcoxon signed-rank ·
effect size (Cliff's delta) · regression gate · exit code · replay adapter ·
FastAPI · API-key auth · SHA-256 at rest · HMAC signature · correlation ID

## Key moments (the middle)

- The both-orders sketch: answer cards A and B graded (A,B) then (B,A); the
  verdict flips → the flip is *measured* as position_bias_rate, and the item
  lands at 0.5 instead of a wrong 0 or 1.
- The kappa worked example, computed live on the board:
  judge [0,0,1,1] vs human [0,1,1,1] → p_o 0.75, p_e 0.5 → kappa = 0.5.
- The exam-leak sketch: an eval item sharing an 8-gram with the corpus gets
  crossed out — "the exam must not leak into the textbook".
- The gate board: paired deltas → bootstrap CI [0.65, 0.81] excluding zero →
  Cliff's delta 1.0 → big red EXIT 1, and the noise run beside it staying
  silently green.

## Outro / punchline

The recap, then the end card: "VerdictAI — nothing is trusted without a number."

## User flow worth showing

The repo is a harness + CLI + service, so the "flow" shown is the harness's
real flow, recreated as board sketches: `verdict pair` judging both orders →
`verdict calibrate` printing before/after agreement → `verdict regress --gate`
ending with exit 1. Real command names from the repo appear verbatim.

## Tone

- Preset: polished (mapped from the freeform tutor direction)
- Creative direction: patient senior engineer at a whiteboard; long-form
  lecture pacing; every keyword defined on screen at first use; calm, precise,
  friendly
- Interpretation: long holds, one idea per sketch, generous reading time;
  understated motion; the numbers do the impressing, not adjectives

## Format: landscape — 1920x1080
## Duration: ~390 seconds (TUTOR_BRIEF override of the 15–25s default; verify 4–7 min)

## Visual identity (series-consistent with the sibling HVAC-Copilot episode)

- Background: dark chalkboard #13211d (radial light washes)
- Text: chalk #f2f0e9; dimmed chalk #d5d2c6
- Accent: amber #f5b942 (rules, warnings, exit 1), teal #63d3c3 (verified,
  passing), blue #9ec5e8 (structure, humans), soft red #f09a8e (bias, failure)
- Display font: "Ink Free" (shipped locally at assets/fonts/Inkfree.ttf)
- Body font: system-ui sans
- Strongest visual element: the kcard definition rail — every keyword appears
  as a handwritten card with its one-sentence definition, series signature

## Voiceover script

See `voiceover-script.txt` (scene-delimited). Voice: Kokoro af_heart, one WAV
per scene; scene durations are derived from measured WAV lengths, never
hardcoded. Numbers written out for TTS; exact figures appear on screen.

## Share copy (draft)

"I recorded a whiteboard lecture on my own repo: VerdictAI — who grades the
grader? Rubric judges run pairwise in both orders, biases are measured not
assumed, kappa checks the grader against humans, isotonic regression curves the
grades, golden datasets are decontaminated so the exam can't leak, and a
bootstrap-CI gate blocks real regressions with exit 1. Six minutes, every
keyword defined on screen. 177 tests pass offline."

## Audio direction

- Role: warm, quiet bed under continuous narration — the voice is the lecture
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3 (steady, clean),
  looped back-to-back, volume 0.13 for the whole video (narration never stops
  long enough to duck up)
- Music treatment: constant low bed; no swells that fight the voice; fade under
  the end card
- Music cue guidance: not used — pacing is driven by measured narration WAV
  lengths, not the beat grid; continue without beat/cue sync
- Audio-reactive treatment: none — a lecture stays still; motion comes from the
  board, not the waveform
- SFX posture: very sparse (4-5 cues in ~6.5 minutes): soft drops for board
  elements landing, one impact on the EXIT 1 stamp, one soft drop on the end
  card; all low volume
- Audio-coupled moments: the EXIT 1 stamp (impact), the three bias chips
  landing one by one (soft drops), stat tiles counting up (no SFX — voice carries)
- Restraint rule: audio must never draw attention; if in doubt, omit the cue

## Storyboard

### Scene 1 — who grades the grader? — ~35s
The board: an answer sheet feeding an "AI JUDGE" box that stamps a verdict;
beneath it the question "…and who grades the JUDGE?". Three tic cards list
themselves: prefers the first answer read · favors longer answers · kind to its
own style. The promise card: "treat the grader as a measured component."
Keywords: LLM, judge. Sequential: tic cards arrive one by one.
Audio intent: quiet intrigue, no drama. SFX: soft drop per tic card.
Transition mood: clean wipe → Scene 2.

### Scene 2 — the core idea: a quality-control lab — ~34s
The analogy: training a new teaching assistant. Four stations drawn on the pipe:
GRADE (rubric) → CHECK against humans (calibration) → BUILD the exam (golden
datasets) → GATE the release (exit codes). Keywords: calibration, golden
dataset, regression gate. Sequential: stations + arrows appear left to right.
Audio intent: steady, orienting. Transition: clean → Scene 3.

### Scene 3 — station one: the rubric — ~33s
A marking-scheme card: accuracy 0–5, evidence 0–5, tone 0–5, weighted; anchors
line ("4 = right idea, one factual slip"); a required evidence quote chip:
"…because photosynthesis happens in chloroplasts". Keyword: rubric, anchor,
evidence quote, rubric decomposition. Sequential: criterion rows appear one by
one, the quote chip last. Audio intent: methodical. Transition: clean → Scene 4.

### Scene 4 — pairwise judging, both orders — ~44s
Answer cards A ("refunds post within 5 business days…") and B ("soon"); judge
says A wins; the cards SWAP and the judge is asked again — flip detected. The
score lands at 0.5 and position_bias_rate counts the flips. Two more bias chips:
verbosity bias (score-vs-length slope, permutation test), self-preference
(family match + preference flag). Keywords: pairwise judging, position bias,
verbosity bias, self-preference. Sequential: swap animation, then chips.
Audio intent: the "aha" of catching a tic. SFX: soft drops on chips.
Transition: clean → Scene 5.

### Scene 5 — kappa: agreement beyond luck — ~37s
Humans grade a sample too. Raw agreement misleads: coin-flippers agree by luck.
The formula kappa = (p_o − p_e) / (1 − p_e) with the worked example table:
judge [0,0,1,1] vs human [0,1,1,1] → p_o 0.75, p_e 0.50 → kappa 0.5. A 0→1
scale bar marks 0.5 as "moderate". Side cards: weighted kappa (partial credit
on 0–5), Spearman (ranks). Keywords: Cohen's kappa, observed agreement, chance
agreement, QWK, Spearman. Sequential: the worked-example rows compute live.
Audio intent: careful, arithmetic-out-loud pacing. Transition: clean → Scene 6.

### Scene 6 — curving the grades: isotonic regression + PAV — ~37s
Scatter: judge scores vs human scores; judge says 5, humans say 3.5 —
saturation kink. The fix: fit the best never-decreasing mapping — a teacher
curving marks with no assumed shape. PAV in one line: walk left to right; when
a block dips below its neighbor, pool and average until monotone. Report shows
before/after MAE; overfitting below ~30 labels is admitted. Side cards: PAV
fixture 2.5 → 3.75 interpolation; Platt scaling (one sigmoid) and why not.
Keywords: isotonic regression, PAV algorithm, calibration curve, Platt scaling.
Audio intent: the satisfying click of a correct curve. Transition: clean → Scene 7.

### Scene 7 — a better exam: golden datasets + decontamination — ~37s
The generator's grid: topics × personas × difficulty × edge cases; coverage
matrix cells fill as items land; dataset written as v1 JSONL + sha256 manifest
+ changelog. Then the leak sketch: an eval item sharing an 8-word run with the
corpus gets crossed out — decontamination, n-gram overlap n=8. Keywords: golden
dataset, coverage matrix, n-gram overlap, decontamination. Sequential: grid
cells, then the crossed-out item. Audio intent: quiet vigilance.
Transition: clean → Scene 8.

### Scene 8 — the alarm: bootstrap CI + the gate — ~46s
Two snapshot rows (append-only, dataset hash recorded); paired deltas per item;
resample 2,000 times → 95% CI of the mean delta; if it excludes zero AND
Cliff's delta clears the floor → REGRESSED → EXIT 1. Beside it, the noise run
(CI includes zero) stays silently green. Side cards: Wilcoxon signed-rank
(small-n backup), snapshot, baseline. Keywords: baseline, snapshot, paired
delta, bootstrap confidence interval, Wilcoxon signed-rank, Cliff's delta,
regression gate, exit code. Sequential: deltas, CI bar, then the EXIT 1 stamp.
Audio intent: the one dramatic beat of the lecture. SFX: impact on EXIT 1.
Transition: hard → Scene 9.

### Scene 9 — the numbers — ~40s
Stat grid, counting up: 177 tests, offline, under two seconds · kappa fixture
0.5 (hand-computed) · QWK fixture 2/7 · planted degradation: candidate replays
golden answers at 75% quality → mean 5.00 → 4.07, a ~19% drop → gate exit 1 ·
noise passes with exit 0. Keywords: replay adapter. Sequential: tiles arrive
one by one with count-ups. Audio intent: measured pride. Transition: clean → Scene 10.

### Scene 10 — the lab has a front door — ~26s
The same harness over HTTP: FastAPI; POST an eval run → 202 + run_id, poll
until succeeded; webhook with HMAC signature (verify the raw bytes); API keys
stored only as SHA-256 hashes; X-Correlation-ID echoed everywhere. Keywords:
FastAPI, API-key auth, SHA-256 at rest, HMAC signature, correlation ID.
Audio intent: practical, closing the loop. Transition: soft → Scene 11.

### Scene 11 — the 30-second recap — ~34s
Six lines, one per beat: judge with a rubric, not vibes · pairwise both orders,
biases measured · kappa checks the grader against humans, beyond luck ·
isotonic curves judge scores onto the human scale · golden datasets,
decontaminated and versioned · the gate separates real regressions from noise
— exit 1 only for the real ones. End card: "VerdictAI — nothing is trusted
without a number." Audio intent: calm summary. SFX: soft drop on end card.

**Music mood for this video:** steady, clean, quiet (vol-12)
**Audio summary:** one constant low music bed under eleven scene narrations;
SFX limited to a handful of soft drops and a single impact on EXIT 1.

## Reading-time floors

Every kcard definition is a full sentence: it must hold ≥ 0.3s per word,
settled. Scene narration lengths give each kcard 15–45s of settled time — far
above the floor. Sequential text reveals are spaced ≥ 0.8s apart and hold.
