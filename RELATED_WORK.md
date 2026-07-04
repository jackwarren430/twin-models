# Related Work — self-play problem-generation RL for LLMs

**One-shot literature overview, written 2026-07-03 (post-Sprint 7, pre-mini-04). Not a living
document** — it captures the field as of this date and how each piece of work relates to
twin-models. The project spec is DESIGN_V2.md; run history is EXPERIMENTS.md.

Confidence markers: entries tagged *(abstract-verified)* were checked against the paper/abstract
during writing; untagged claims for the 2024–2025 papers come from a deeper read. The 2026 papers
(PopuLoRA, SGS, Vocabulary Dropout) are post-hoc discoveries known only from their abstracts.

---

## 0. The map

Twin-models sits at the intersection of four literatures:

| Thread | Canonical works | What twin-models takes / contests |
|---|---|---|
| Proposer–solver self-play for LLM reasoning | Absolute Zero (AZR), R-Zero, SPICE, SQLM, PSV, PopuLoRA | The whole game loop; twin's deltas are two-weight separation + rotation + a *calibration* (multi-rank) reward instead of a single learnability band |
| Automatic curricula in RL (pre-LLM) | Asymmetric self-play (Alice/Bob), Goal GAN/GOID, POET, PAIRED, TSCL | The interior target band, the consistency requirement, regret-flavored difficulty |
| Policy-gradient mechanics for RLVR | DeepSeekMath GRPO, Dr. GRPO, SPIRAL's RAE | Mean-baseline advantages, group structure, the queued per-rank credit decomposition |
| Verification & checkability | Prover-Verifier Games, PSV, LLM-judge critiques | Certificate-first consistency; the judge-as-fallback design and its known weaknesses |

The single most important external fact for this project: **every published proposer–solver system
that measures long-horizon behaviour reports the same two failure modes twin-models has already
hit at miniature scale — difficulty ceiling compression (proposer self-calibrates to easy) and
curriculum diversity collapse.** The field's countermeasures are catalogued in §5.

---

## 1. The direct ancestor

### Absolute Zero: Reinforced Self-play Reasoning with Zero Data (Zhao et al., 2025 — arXiv:2505.03335) *(abstract-verified)*

**What it is.** A single model plays both **proposer** and **solver** over code-based reasoning
tasks in three modes — deduction (given program+input, predict output), abduction (given
program+output, find an input), induction (given input/output pairs, write the program). A **code
executor** is the unified verifier: it validates proposed tasks (runs the proposer's program to
get the ground-truth answer) and grades solver attempts. Trained with **Task-Relative REINFORCE++
(TRR++)**: separate advantage baselines for each of the six task×role configurations. AZR-7B
reached SOTA-for-zero-data on math+coding averages, beating models trained on tens of thousands
of curated examples, with strong cross-domain transfer (code self-play → math gains).

**The learnability reward.** The proposer's reward is `1 − p̄_solve` when the solver's average
solve rate over rollouts is strictly between 0 and 1, and **0 when p̄ is exactly 0 or 1** —
trivial and impossible tasks earn nothing. This is a *single band per task*.

**Relation to twin-models.**
- Twin's **difficulty-gradient reward is a strict generalization**: instead of "somewhere in
  (0,1)", the creator must place N problems at N *specific* solve rates (0.9→0.1). This is a
  harder, more informative calibration game — a real claimed contribution, but note AZR's zero-
  at-the-endpoints already encodes the same anti-saturation logic twin rediscovered mechanically
  (the zero-advantage skip fires exactly where AZR's reward is zero).
- AZR's **executor-validates-the-proposal** step is twin's consistency check — but with a crucial
  difference in trust. AZR *derives* the ground truth by running the program; the proposer cannot
  state a wrong answer. Twin's creator **states its own answer** and backs it with a
  self-written certificate. The executor is a ground-truth oracle; a certificate is only a
  self-consistency witness (see §4 and the statement/cert-mismatch gap in DESIGN_V2 §14). This is
  the deepest structural difference and the main thing math-domain twin gives up relative to
  code-domain AZR.
- **TRR++ says per-role, per-task baselines matter.** Twin has per-role updates (separate
  adapters, separate GRPO groups) but the creator group pools all ranks under broadcast credit —
  the Sprint-8 credit-decomposition ablation is exactly the TRR++ direction and the literature
  supports promoting it.
- AZR reports an **"uh-oh moment"** — a Llama-3.1-8B run produced an adversarial "outsmart
  humans" chain of thought — flagging that unsupervised self-play needs oversight even at small
  scale. Twin's transcript-always-on habit is the right hygiene; worth keeping a safety eye on
  transcripts, not just a debugging eye.

---

## 2. The closest relatives (two-weight / co-evolving designs)

### R-Zero: Self-Evolving Reasoning LLM from Zero Data (Huang et al., 2025 — arXiv:2508.05004, ICLR 2026) *(abstract-verified)*

**What it is.** The published system closest to twin's headline claim: **two separately-trained
models** from one base — a **Challenger** and a **Solver** — co-evolve, alternating freeze/train
phases. The Challenger is rewarded by an **uncertainty reward peaked at a 50% solver success
rate**, plus a BLEU-based **repetition penalty** for within-batch problem diversity. Ground truth
for training the Solver is a **majority-vote pseudo-label** over m=10 solver samples, and only
problems with *moderate* vote consistency (e.g. 3–7 of 10 agreeing) are kept — an explicit
informativeness filter. Gains: +6.49 math / +7.54 general on Qwen3-4B-Base.

**Measured failure modes:** pseudo-label accuracy decays 79% → 63% by co-evolution round 3 as
problems get harder (majority vote stops being reliable ground truth), and overall gains
plateau/fade across rounds — the authors call sustaining long-term improvement "a fundamental
problem for the self-evolving paradigm."

**Relation to twin-models.**
- Twin's independent-weights claim is **not unprecedented** (R-Zero also has two weight sets),
  but twin's specific mechanisms are distinct: (a) **role rotation** — R-Zero's Challenger never
  solves and its Solver never creates; twin's "solver heuristics injected into the creator by
  rotation" is a genuinely different (and still unvalidated — no fixed-roles control run yet)
  hypothesis; (b) **LoRA adapters on one frozen base** vs two full models — the 32GB-Mac trick,
  which PopuLoRA (below) independently validates as a research-worthy architecture.
- **Twin's certificates directly attack R-Zero's measured bottleneck.** R-Zero's ground truth
  (majority vote) *degrades precisely as the curriculum succeeds*. A CAS-checked certificate does
  not decay with difficulty. This is a strong argument for twin's verifiable-first commitment —
  and equally a warning: everywhere twin falls back off the certificate path (judge fallback,
  string-compare answer grading), it re-inherits R-Zero's problem.
- **R-Zero's band filter is the data-side fix for twin's solver starvation.** mini-02 measured
  88% of K-groups saturated at 0/1; twin's fixes were reward-side (brevity tie-break, interior
  band). R-Zero simply *drops* saturated problems from the solver's training batch. Twin
  effectively does this via the zero-advantage skip, but only after paying full generation cost —
  the observation that generation, not the update, is the cost driver makes R-Zero-style early
  filtering (or rank-adaptive K) attractive.
- **The repetition penalty has no twin analog** — see §5 on diversity.

### PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play (2026 — arXiv:2605.16727) *(abstract only)*

**What it is.** Built explicitly on AZR: **populations of LoRA adapters over a shared frozen
base**, split into teacher (proposer) and student (solver) sub-populations, with cross-evaluation
between sub-populations replacing single-agent self-grading, plus LoRA weight-space evolution
operators (mutation/crossover). Key empirical claim: **single-agent self-play self-calibrates —
the proposer converges to problems it can reliably solve and the training distribution collapses
onto a narrow easy band**; cross-evaluation across a population breaks this. At 7B scale, the
whole population (even its weakest member) beat the baseline across ten math+code benchmarks,
with an ongoing teacher-student arms race and expanding problem-space coverage.

**Relation to twin-models.**
- **Twin's architecture is this paper's architecture at population size 2.** Multiple LoRA trees
  over one frozen base as cheap population members is exactly the Adapters design. This both
  validates the choice and marks the scaling axis: if rotation-with-two proves insufficient
  anti-collapse pressure, the next move the literature suggests is not a bigger model but *more
  adapters* (a third/fourth tree is tens of MB).
- Their diagnosis — proposer self-calibrating to solvable-by-itself problems — **is mini-02/03b's
  ceiling compression**, observed independently. Twin's counter so far is prompt-side (dictated
  targets, personas, "hardest rank" language); PopuLoRA's counter is structural (evaluate against
  *someone else*). Twin already grades the creator against the *other* adapter's solve rate, which
  is the two-agent version of cross-evaluation — the mini-04 question is whether two agents that
  co-train stay "different enough" for that pressure to persist. If A and B converge to similar
  policies, cross-evaluation degenerates toward self-evaluation, and the population insight says
  the gradient signal quietly dies. Watch: A/B adapter drift *divergence* (are they moving apart
  or together?) — the instrumentation already logs both norms.

### SPIRAL: Self-Play on Zero-Sum Games… (Liu et al., 2025 — arXiv:2506.24119) *(abstract-verified)*

**What it is.** Self-play on multi-turn zero-sum *games* (e.g. Kuhn Poker) rather than
problem-posing; fully online multi-agent RL; transfer to reasoning (Kuhn-Poker-only training:
+8.6% math on Qwen3-4B). Two findings matter here: (1) **Role-conditioned Advantage Estimation
(RAE)** — normalize rewards relative to each *role's* expected performance — is what keeps
multi-role training stable; (2) without it they observe **"thinking collapse": the model
progressively abandons its reasoning traces**, and transfer dies.

**Relation to twin-models.**
- RAE is more support for per-role/per-rank baselines (with broadcast credit, twin's creator
  advantages compare rank-0 trajectories against rank-4 trajectories through one group mean —
  role-conditioning within the creator group is the decomposition ablation again).
- **Thinking collapse is the named failure mode for w_brevity.** Twin pays brevity bonus on
  *all* completion tokens including `<think>`, at 0.25 in mini-04 (up from validated 0.15). The
  intended effect is thinking compression; SPIRAL documents the overshoot: models that stop
  thinking stop transferring. The EXPERIMENTS watch-item ("solve rate must not drop") is right
  but insufficiently sensitive — track *think-token share* per solved attempt, not just solve
  rate, because collapse can precede the solve-rate drop.
- Twin's game is deliberately **not zero-sum** (calibration, both sides can win) — a real design
  difference. Zero-sum gives an automatic curriculum but invites degenerate stumping; calibration
  gives a controllable curriculum but leans entirely on the reward's honesty (§5's hacking
  results are about exactly this class).

---

## 3. Grounding the curriculum (the 2025–2026 wave)

### SPICE: Self-Play In Corpus Environments (Liu, Jin et al., Meta, 2025 — arXiv:2510.24684) *(abstract-verified)*

A Challenger mines a **document corpus** to pose tasks whose gold answers are grounded in
retrieved text the Reasoner never sees (**information asymmetry** as the difficulty source);
Challenger reward maximizes solver-success *variance*. Their motivating claim: **ungrounded
self-play hallucinates and collapses because models train on their own synthetic
fantasies** — grounding in an external corpus is what breaks the error-amplification loop.
Gains: +8.9% math / +9.8% general across families.

*Relation:* twin's external anchors are the CAS/certificates (math truth) and the frozen base
(KL + judge + unwired oracle). That is grounding for *correctness* but not for *content* — the
creator's problem distribution is anchored to nothing outside the two policies plus ten hardcoded
themes. SPICE names the risk of that. If/when a knowledge domain is added, corpus-grounding with
information asymmetry is the design to copy (it also solves "where does the reference answer come
from" — the exact problem that killed twin's coding domain). Note twin already has natural
information asymmetry available for free: the creator's `solution` is hidden from the solver.

### Scaling Self-Play with Self-Guidance (2026 — arXiv:2604.20209) *(abstract only)*

On Lean4 theorem proving: pure difficulty-targeting proposers **hack the difficulty reward,
collapsing to artificially complex problems that don't help the solver**; self-play stops scaling
with compute. Fix: a third **Guide** role scores proposed tasks for (a) *relevance to
still-unsolved target problems* and (b) quality/naturalness; grounding generation in the target
distribution restored scaling (7B beating a 671B baseline after 200 rounds).

*Relation:* this is the sharpest published warning about twin's core reward. "Artificially
complex but useless" is the failure the difficulty-gradient reward cannot see — a problem can sit
at exactly the 0.1 target by being *tediously* hard rather than *instructively* hard (twin's
prompts say "structurally hard, never bigger numbers", but that's prompt-side, unmeasured, and
Goodhartable). Twin has a ready-made target distribution: the **hard bench**. A cheap SGS-flavored
step: periodically condition creator themes on bench categories/items the adapters still fail —
it grounds the curriculum in exactly what the held-out metric measures without contaminating it
(condition on *failure topics*, never items).

### Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution (2026 — arXiv:2604.03472) *(abstract only)*

In proposer/solver co-evolution the proposer **converges to a narrow set of question templates**
(diversity collapse) even while difficulty targeting works; they inject diversity by constraint
(vocabulary dropout on the proposer).

*Relation:* see §5 — twin currently has *no* diversity mechanism beyond a 10-theme pool and a
"genuinely distinct" prompt line, and (unlike R-Zero) no repetition penalty. At 30 iterations
this hasn't bitten; every scaled system says it will.

### Propose, Solve, Verify: Self-Play Through Formal Verification (2025 — arXiv:2512.18160) *(abstract only)*

Self-play for verified code (Verus): proposer + solver + **formal verification as the reward
signal**, replacing brittle unit-test rewards; difficulty-aware proposal; expert iteration. Up to
9.6× pass@1 gains; ablations show **verification quality and difficulty-aware proposal are each
essential** — neither alone suffices.

*Relation:* independent convergence on twin's two core bets (machine-checkable verification +
difficulty-aware generation) in a different domain — good news for the design. Practical import
is for the queued **coding pipeline sprint**: mini-02 measured creator-written unit tests at
0/141 self-consistent; PSV says don't fix that by making the model write better ad-hoc tests,
fix it by making the *check* cheap to satisfy honestly and expensive to satisfy dishonestly
(property-style checks, executor-derived ground truth à la AZR — run the reference solution to
*generate* the expected outputs instead of asking the model to state them).

### Self-Questioning Language Models (2025 — arXiv:2508.03682)

Proposer/solver self-play from a one-line topic prompt (e.g. "algebra word problems"); no
external data; majority-vote self-grading for math. Confirms the minimal loop works at small
scale — and shares R-Zero's pseudo-label ceiling. *Relation:* twin's certificate path is again
the differentiator; SQLM is roughly "twin without certificates, without the gradient ramp,
without two weight sets," i.e. a useful mental baseline for what the extra machinery must beat.

### Towards Understanding Self-play for LLM Reasoning (2025 — arXiv:2510.27072) *(abstract only)*

Analysis paper on AZR-style training: studies parameter-update sparsity, token-entropy dynamics,
and alternative proposer rewards; positions self-play against RLVR/SFT and highlights inherent
limitations. *Relation:* worth a full read before designing the next reward change — it is the
closest thing to a theory of *why* the proposer reward shape matters, and twin is about to bet a
run (mini-04) on a particular shape.

---

## 4. Foundations twin-models is (knowingly or not) standing on

### Asymmetric self-play: Alice & Bob (Sukhbaatar et al., 2017 — arXiv:1703.05407)

The origin of the proposer/solver pattern in RL. Alice performs a task, Bob must repeat/undo it;
Alice's reward grows with Bob's failure **but Alice must be able to do the task herself** — the
self-consistency constraint that keeps proposals meaningful. *Relation:* twin's consistency check
is the LLM translation of "Alice must do it herself," and the void mechanism is its enforcement.
The classic Alice/Bob failure — Alice finds environment quirks Bob can't reproduce rather than
skills worth learning — translates directly: a creator exploiting verifier quirks (answer-format
voids in mini-02 were an *accidental* version; a trained version would be deliberate).

### Goal GAN / Goals of Intermediate Difficulty (Florensa et al., 2018 — arXiv:1705.06366)

Curriculum = generate goals whose current success rate lies in an **interior band
[R_min, R_max] ≈ [0.1, 0.9]**; endpoints carry no learning signal. *Relation:* twin's interior
target band (0.9→0.1) is GOID rediscovered, with the ramp as a refinement (a *spectrum* of GOID
bands, one per rank). The prior art both legitimizes the choice and suggests its known weakness:
success-rate bands are noisy at small K (twin's K=8 gives eighths — mini-04's exact quantization
concern).

### PAIRED / Unsupervised Environment Design (Dennis et al., 2020 — arXiv:2012.02096)

Curriculum via **regret** = (antagonist's performance − protagonist's performance): propose tasks
the *stronger* reference can solve but the learner can't — self-limiting difficulty with a
built-in solvability proof. *Relation:* twin's consistency check makes the creator its own
antagonist: "creator can solve it (cert passes), solver can't (low p̂)" *is* a regret signal.
Seeing it through the PAIRED lens suggests a principled alternative to the MSE-vs-ramp reward if
calibration ever proves too gameable: reward = verified-creator-solves × (1 − solver-solve-rate),
which needs no target curve at all.

### POET (Wang et al., 2019 — arXiv:1901.01753) & Teacher-Student Curriculum Learning (Matiisen et al., 2017 — arXiv:1707.00183)

POET: co-evolve environments and agents under a **minimal criterion** (not too easy, not too
hard) with transfer between niches — the population/arms-race lineage PopuLoRA imports to LLMs.
TSCL: the teacher should select tasks where the student's **learning progress** (slope, not
level) is highest. *Relation:* twin's reward targets a solve-rate *level*; TSCL argues the ideal
target is the *derivative* (fastest-improving problem types). Impractical to reward directly at
twin's scale, but the analysis lens is free: the analyzer could report solve-rate slope per
theme/difficulty bucket to show where learning actually happens.

### GRPO & Dr. GRPO (Shao et al., 2024 — arXiv:2402.03300; Liu et al., 2025 — arXiv:2503.20783)

GRPO: group-relative advantages replace the critic. Dr. GRPO: the ÷std term and per-response
length normalization each *bias* optimization (std-normalization over-weights near-tie groups;
length terms reward longer wrong answers); remove them. *Relation:* twin's mean-baseline switch
is Dr. GRPO-informed and was empirically forced by tiny groups (mini-02). One nuance the audit
surfaced: twin's loss divides summed token-losses by *total batch trained tokens*, so a
trajectory's gradient share scales with its length — a mild length bias of the family Dr. GRPO
warns about, which interacts non-obviously with the brevity bonus (long trajectories get more
gradient mass while the reward asks for short ones). Worth a deliberate decision, not an
accident.

### Prover-Verifier Games (Kirchner et al., OpenAI, 2024 — arXiv:2407.13692) and LLM-judge caveats

Training against a small verifier for **checkability** produces legible solutions; conversely,
LLM judges are known to be gameable (self-preference, verbosity bias, format anchoring).
*Relation:* twin's certificate contract is checkability-by-construction — the right instinct.
The judge fallback is the soft spot: it is the same base model family as the policy (family
self-preference), on the protocol already shown broken (mini-03b), ruling on exactly the
problems that *chose* not to certify — an adversely selected sample. The prompt already
threatens "cert-less problems are discarded" while the code actually falls back to the judge;
the literature says close that gap in the strict direction (make the threat real, or make the
judge audit *certified* problems too as a spot-check).

### Others in the family, briefly

- **SPIN** (Chen et al., 2024 — arXiv:2401.01335): self-play as generator-vs-discriminator on SFT
  data; the "self-play" name, different mechanism.
- **SPAG** (Cheng et al., 2024 — arXiv:2404.10642): adversarial Taboo self-play improving general
  reasoning — early evidence that game-shaped self-play transfers.
- **Self-Rewarding LMs** (Yuan et al., 2024 — arXiv:2401.10020): model judges its own outputs;
  documents the drift risk of self-judged reward, i.e. why twin keeps the judge frozen.
- **Minimo** (Poesia et al., 2024 — arXiv:2407.00695): conjecturer/prover self-play in a formal
  system with **hindsight relabeling** — when the prover fails the target but proves something
  else en route, that byproduct becomes training data. Twin's analog would be salvaging solver
  attempts that verify against a *different* rank's answer; cheap data at zero extra generation
  cost.

---

## 5. Synthesis — what the literature says about this project

### Where twin-models is ahead of or level with the field

1. **Verifiable-first consistency (certificates)** — the strongest single design choice. It
   dodges the measured Achilles heel of R-Zero/SQLM (pseudo-label decay) and matches the PSV
   finding that verification quality is an essential ingredient, in a domain (school math) where
   full formal verification is overkill.
2. **Calibration-to-a-ramp creator reward** — a genuine generalization of AZR's learnability band
   and GOID's interior band; nobody else grades a proposer on hitting *multiple* solve-rate
   targets simultaneously. It is also unproven, and §5's hacking results (SGS) apply to it with
   full force. mini-04 is, in effect, this idea's first real test.
3. **Two adapters on one frozen base** — independently validated as a research architecture by
   PopuLoRA; twin's rotation-as-injection variant is distinct from both R-Zero (fixed roles) and
   PopuLoRA (populations). It is the project's most original claim and currently has **no
   control run** (fixed-roles ablation) behind it.
4. **Held-out absolute benchmark discipline** — many papers in this space report only benchmark
   deltas without a creator-relative/absolute split; twin's core+hard two-tier design with a
   frozen-base bar is clean methodology. (Caveat: 20 items/category makes per-category deltas
   directional; and the bench runs thinking-OFF while training runs thinking-ON — a regime
   mismatch worth either fixing or explicitly owning.)
5. **Transcript-first debugging culture** — mini-02's cert-fail decomposition and mini-03b's
   fake-tool-call discovery are exactly the kind of failure analysis most self-play papers lack.

### What the literature says is missing (ranked by evidence strength)

1. **A diversity mechanism.** Unanimous across R-Zero (BLEU repetition penalty), Vocabulary
   Dropout (template collapse), PopuLoRA (coverage expansion as a headline metric): difficulty
   control without diversity control ends in a narrow curriculum. Twin has ten fixed themes and a
   prompt adjective. Cheap first steps: per-iteration n-gram/embedding overlap telemetry across
   problems (detect before fixing), an R-Zero-style within-suite repetition penalty, a grown
   theme pool (the creator itself can propose themes — logged, curated offline).
2. **Grounding the curriculum in a target distribution.** SGS and SPICE both find pure
   self-referential difficulty eventually stops helping (reward-hacked complexity, hallucinated
   content). Twin's hard bench is a ready-made anchor: condition creator prompts on
   failure *topics* from the last bench run. Zero new infrastructure, directly attacks ceiling
   compression with content rather than adjectives.
3. **Per-role/per-rank advantage baselines.** TRR++ (six baselines), SPIRAL's RAE (per-role
   normalization): the field treats this as load-bearing, not an ablation. Twin's queued Sprint-8
   credit decomposition should probably be promoted after mini-04 regardless of outcome.
4. **A plan for the plateau.** R-Zero's gains fade within ~3–5 co-evolution rounds; SGS frames
   sustaining improvement as *the* open problem. Twin's 30-iteration runs are far below the
   horizon where this bites, which cuts both ways: no immediate risk, but also no evidence yet
   that the loop *scales in iterations* — the next milestone after mini-04 succeeds should
   probably be a 100+ iteration run watching for the R-Zero fade, before any mechanism work.
5. **Thinking-collapse telemetry.** SPIRAL: brevity/efficiency pressure can silently kill the
   reasoning traces that make transfer work. w_brevity at 0.25 with think-tokens in scope needs a
   think-share-per-solved-attempt metric next to solve rate.
6. **Salvage/hindsight data.** Minimo's relabeling and AZR's task-triplet reuse both extract more
   signal per generated token; twin discards non-verified generation wholesale. At 16 tok/s,
   generation is the budget — salvage is the cheapest scaling lever after batching.

### Honest positioning

Twin-models is a *serious, competently engineered instance* of a research direction that got
crowded in 2025–2026. Its distinguishing bets — rotation-as-injection, the calibration ramp,
CAS certificates on natural-language math — are real and none has been published in this
combination. But two of the three (rotation, ramp) are currently *unvalidated bets*, and the
field's convergent findings (diversity collapse, difficulty-reward hacking, plateau) predict the
specific ways they will be stressed. The project's experimental discipline is its best asset:
the same run-and-decompose loop that caught mini-03b's fake tool calls is what will catch these.

---

## 6. Backlog answers (2026-07-03, web-verified)

Three questions from the project backlog, researched against the July-2026 literature.

### 6a. Can we use TinyLoRA? ("Learning to Reason in 13 Parameters", arXiv:2602.04118)

**What it is.** Weight tying + fixed random projections scale a LoRA-style update down to
arbitrarily few trainable parameters — as few as ONE. Headline: Qwen2.5-7B-Instruct goes
76.0% → 91.8% on GSM8K trained with **GRPO on 13 parameters** (26 bytes in bf16), ≈ full-FT's
91.7%; ~90% of the gains at 1000× fewer params across AIME/AMC/MATH500. Two findings matter more
than the stunt number: (1) **RL specifically** works at tiny capacity — SFT needs 100–1000×
larger updates for the same gains; (2) "tiling" (sharing params across depth) beats sharing by
module type.

**Verdict for twin-models: yes, trivially implementable (fixed seeded projections + a tiny
trainable vector on the frozen base — same `model.update(tree)` machinery), but it does NOT
attack our bottleneck, and it cuts against the headline hypothesis.**
- Our constraint is generation throughput (~16 tok/s) and backward *activation* memory (the
  [T,V] logits, 55GB worst-case at 4096 — see EXPERIMENTS.md mini-02 probe), neither of which shrinks with
  adapter size. 2×77M adapter params are already free.
- The scientific reading is the real payoff: TinyLoRA is strong evidence that RLVR mostly
  **elicits** capabilities already in the base rather than adding knowledge. Twin-models' story
  ("two adapters diverge into distinct skills/knowledge and pull each other up") needs adapter
  capacity to be *load-bearing*. A **capacity-downward ablation** (rank 64 → 8 → 1 → TinyLoRA-
  style tied) is therefore a *sharper version of an experiment we already planned*: if the loop's
  gains survive at rank-1, rotation-injection is eliciting, not teaching — that's a publishable
  negative/positive either way. Queue it with the ablation series, not before.

### 6b. What problems did AZR use? What did comparable setups use?

Already detailed in §1/§2; the compact answer the backlog wants:
- **AZR**: no natural-language problems at all — **code triplets** (program, input, output) in
  three modes: *deduction* (program+input → output), *abduction* (program+output → find input),
  *induction* (I/O pairs → write program). The Python executor derives ground truth by running
  the proposer's program — **the proposer cannot state a wrong answer**, which is the single
  deepest difference from our stated-answer+certificate design.
- **R-Zero**: open-ended *math word problems*, ground truth = majority vote over 10 solver
  samples (decays 79%→63% as problems harden — the failure our certificates exist to avoid).
- **SPIRAL**: zero-sum *text games* (Kuhn poker etc.) — skills transfer to math anyway.
- **SPICE**: questions grounded in a *retrieved corpus* — ground truth from documents.
- **Minimo**: formal *Lean conjectures* — proof-checker ground truth.
- Lesson mined for us: AZR's three task *modes* are a built-in **structural diversity mechanism**
  (same theme, three inverse problems). Our creator has one mode: "pose a problem". A cheap
  math analog — "given this solution/derivation, write the problem" (abduction) — is a
  Sprint-9-adjacent diversity lever no one in our queue has yet.

### 6c. "Use the new agentworld model — why would this be better than an actual environment?"

Two distinct 2026 artifacts, neither actually DeepSeek:
- **Qwen-AgentWorld** (Qwen, arXiv:2606.24597, June 2026): a *language world model* (LWM) —
  35B-A3B and 397B-A17B MoE — trained on 10M+ interaction trajectories to *predict environment
  transitions* (terminal, web, OS, Android, SWE, MCP, search). Agents trained inside the
  simulation beat real-environment-only training on 7 agentic benchmarks.
- **Agent World Model** (Snowflake, arXiv:2602.10090, ICML 2026): NOT an LLM simulator — a
  pipeline that *synthesizes 1,000 executable, database-backed environments* with reward
  functions over real system state. (DeepSeek-V3.2's env-synthesis pipeline is the same family.)

**Answer to the backlog's own question.** Simulated/synthesized environments beat "an actual
environment" on *scale* (thousands in parallel, no infra), *control* (deterministic resets,
reproducible curricula), *coverage* (tasks nobody hosted), and *safety* — at the price of
**ground-truth fidelity** (an LWM hallucinates transitions; a policy learns to exploit the
simulator). That tradeoff is exactly the axis twin-models already fights on (certificate vs
LLM-judge), so the project's answer writes itself:
- **For math/coding (now): an LWM is strictly worse than our "actual environment".** SymPy and
  the sandbox ARE the environment — exact, free, un-hackable by construction. Swapping them for
  a 35B simulator re-inherits R-Zero's decay problem at GPU prices, on hardware (32GB M5) that
  can't host it anyway.
- **For the eventual agentic broadening (later): the Snowflake/DeepSeek *synthesis* route fits
  twin better than the Qwen *simulation* route** — executable envs with state-based rewards
  preserve our verifier-first commitment; the creator's job generalizes from "write a problem +
  certificate" to "write a task + environment + reward check", which is the same contract.
- One genuinely new idea worth keeping: Qwen-AgentWorld shows *world-model training as warm-up*
  improves downstream agents. The twin analog — creator pre-trained to *predict the solver's
  solve rate* before the RL loop starts — is a cheap calibration warm-up that directly serves
  our gradient reward. Filed for Sprint 9+.

---

## References (compact)

| Work | Ref |
|---|---|
| Absolute Zero (AZR) | Zhao et al., 2025, arXiv:2505.03335 |
| R-Zero | Huang et al., 2025, arXiv:2508.05004 |
| SPIRAL | Liu et al., 2025, arXiv:2506.24119 |
| SPICE | Liu, Jin et al., 2025, arXiv:2510.24684 |
| Towards Understanding Self-play for LLM Reasoning | 2025, arXiv:2510.27072 |
| Propose, Solve, Verify (PSV) | 2025, arXiv:2512.18160 |
| PopuLoRA | 2026, arXiv:2605.16727 |
| Vocabulary Dropout | 2026, arXiv:2604.03472 |
| Self-Guided Self-Play (SGS) | 2026, arXiv:2604.20209 |
| Self-Questioning LMs (SQLM) | 2025, arXiv:2508.03682 |
| Asymmetric self-play (Alice/Bob) | Sukhbaatar et al., 2017, arXiv:1703.05407 |
| Goal GAN / GOID | Florensa et al., 2018, arXiv:1705.06366 |
| POET | Wang et al., 2019, arXiv:1901.01753 |
| PAIRED / UED | Dennis et al., 2020, arXiv:2012.02096 |
| Teacher-Student Curriculum Learning | Matiisen et al., 2017, arXiv:1707.00183 |
| GRPO (DeepSeekMath) | Shao et al., 2024, arXiv:2402.03300 |
| Dr. GRPO | Liu et al., 2025, arXiv:2503.20783 |
| Prover-Verifier Games | Kirchner et al., 2024, arXiv:2407.13692 |
| SPIN | Chen et al., 2024, arXiv:2401.01335 |
| SPAG | Cheng et al., 2024, arXiv:2404.10642 |
| Self-Rewarding LMs | Yuan et al., 2024, arXiv:2401.10020 |
| Minimo | Poesia et al., 2024, arXiv:2407.00695 |
| TinyLoRA ("13 Parameters") | 2026, arXiv:2602.04118 |
| Qwen-AgentWorld | Qwen Team, 2026, arXiv:2606.24597 |
| Agent World Model (synthetic envs) | Snowflake, 2026, arXiv:2602.10090 |
