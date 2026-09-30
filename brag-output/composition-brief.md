# Hyperframes Composition Brief: VerdictAI — whiteboard lecture

## Objective
Create a long-form whiteboard explainer LECTURE video for VerdictAI
(tutor series; NOT a launch video). Narration on (Kokoro af_heart). The owner
of the repo must finish the video able to explain the system to someone else.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: ~390 seconds (TUTOR_BRIEF override of the 15–25s default;
  must land 4–7 min). Scene durations derive from measured narration WAV
  lengths — never hardcode them.

## Source Material
- Project root: `D:\aria\Projects\VerdictAI`
- Primary files read: README.md, src/verdictai/** (metrics.py, isotonic.py,
  bias.py, drift.py, gate.py, runner.py, service.py, pairwise.py, rubric.py),
  tests/** (test_metrics.py, test_isotonic.py, test_gate.py, test_cli.py)
- Product name: VerdictAI
- Tagline: "an exam grader that first learns from human teachers"
- Key moments to recreate as board sketches: pairwise both-orders swap with
  flip detection; the kappa = 0.5 worked example; the PAV staircase; the
  coverage matrix filling; the leak item crossed out; the gate's EXIT 1 vs the
  silent green noise run
- Copy that must appear verbatim:
  - `verdict pair` / `verdict calibrate` / `verdict regress --gate`
  - kappa = (p_o − p_e) / (1 − p_e)
  - EXIT 1
  - "nothing is trusted without a number"

## Creative Direction
- Tone preset: polished (mapped from freeform tutor direction)
- Creative direction: patient senior engineer at a whiteboard teaching a smart
  junior; long-form lecture pacing; calm, precise, friendly; no hype
  adjectives; every keyword defined on screen the moment it first appears
- Angle: "who grades the grader?" — the judge is a fallible machine, so treat
  it as a first-class measured component: grade with rubrics, check against
  humans, build a clean exam, gate the release
- Hook: answer → judge → "…and who grades the judge?" with three named tics
- Outro: recap + end card "VerdictAI — nothing is trusted without a number."
- Avoid:
  - Generic SaaS language, launch energy, adjectives like "lightning-fast"
  - Abstract filler visuals
  - Waveform/equalizer graphics

## Visual Identity
Series-consistent with the sibling HVAC-Copilot episode (proven 83/83 WCAG AA):
- Background: #13211d chalkboard with subtle radial washes
- Text: #f2f0e9 (chalk), #d5d2c6 (dim chalk)
- Accents: amber #f5b942 (rules/warnings/EXIT 1), teal #63d3c3 (verified),
  blue #9ec5e8 (structure/humans), soft red #f09a8e (bias/failure)
- Display font: "Ink Free" shipped at `assets/fonts/Inkfree.ttf` with in-file
  @font-face; body: system-ui
- Layout contract: `.board` grid (main sketch area + 440px definition rail of
  kcards); `.scene-tag` bottom-left; each scene a `.clip` section with
  `data-start`/`data-duration`

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract
(11 scenes: problem → lab analogy → rubric → pairwise both-orders → kappa
worked example → isotonic/PAV → golden datasets/decontamination → bootstrap CI
gate → numbers → HTTP front door → recap).

## Audio
- Audio role: warm quiet bed under continuous narration
- Audio arc: constant low bed 0.13 for the full video; no swells; fade under
  end card
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
  (1:58), looped back-to-back with separate `<audio>` elements
- Music treatment: volume 0.13 throughout (narration never pauses)
- Music cue guidance: unavailable/not used — narration WAV lengths drive
  pacing; continue without beat/cue sync
- Audio-reactive treatment: none — a lecture stays still
- SFX selection guidance: `sfx-analysis.md` in the skill assets; prefer low
  HF-risk files; ≤ 5 cues total, volumes 0.4–0.6
- Audio-coupled moments:
  - Scene 1 tic cards — soft drop per card
  - Scene 4 bias chips — soft drops
  - Scene 8 EXIT 1 stamp — one impact (the lecture's single dramatic beat)
  - Scene 11 end card — soft drop
- Exact SFX choice: Hyperframes picks filenames/timestamps to match the
  implemented animation; `interface/drop_*` and `impact/impactSoft_medium_*`
  are the expected families
- Audio files: music + chosen SFX copied into `brag-output/composition/assets/`;
  narration WAVs at `assets/voiceover/voice_XX.wav` (Kokoro af_heart, one per
  scene, generated from `voiceover-script.txt`)

## Hyperframes Instructions
Load `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`,
`hyperframes-keyframes`, `hyperframes-cli`. /brag owns the angle, storyboard,
copy, audio policy; Hyperframes owns composition structure, exact timing,
linting, render. Requirements:
- Monolithic standalone `index.html` (no `<template>` wrapper on the root),
  one paused GSAP timeline registered at `window.__timelines["verdict-lecture"]`
  matching the root `data-composition-id`
- Root `data-duration` = exact sum of scene durations derived from measured
  narration WAV lengths
- Never tween `.clip` elements; animate inner wrappers; use fromTo, no CSS
  transform/GSAP conflicts; every `<audio>` has an id; no crossorigin on media
- All text readable: kcard definitions hold ≥ 0.3s/word settled; no `<br>` in
  body text
- Run `hyperframes check` before render and fix every error including WCAG
  contrast (83/83 text checks passing is the bar)
- Copy the shipped font (sibling provenance: Windows "Ink Free") into
  `assets/fonts/` — do not fetch network fonts
