# FlyCommander Current Architecture (code-verified, 2026-10-01)

Verified by reading every module in the repo on 2026-10-01. Scale: ~7,250
lines across 33 Python modules, 7 Java classes, 1 HTML UI, 6 test files.
Test status: **106 passed / 1 skipped** (`.venv`, OpenCV present), 104/3
(system Python); Java patch compiles against Forge 2.0.15.

## Components (as implemented)

```
┌────────────────────── Forge 2.0.15 (Java 25, headless) ────────────────────┐
│ AgentMain.main         boots FModel.initialize(null,null) (sim-CLI sequence)│
│ DeckResolver           deck specs: random|name|.dck via DeckProxy pool +    │
│                        MyRandom (seedable via -Dfly.agent.seed)             │
│ FlyLobbyPlayer         LobbyPlayerAi subclass; createIngamePlayer override  │
│ FlyPlayerController    PlayerControllerAi subclass; gates brain queries to  │
│                        own main phases (≤8/phase), disables AI simulation  │
│ GameObserver           JSON observation: life/zones/board(identity+tap)/    │
│                        stack/combat counts/canPlay{land,spell,ability}     │
│ AgentServer :8791      GET /health /observation /result                    │
│ BrainClient            POST /decide → {action:0..3}; fallback on failure   │
└──────────────┬─────────────────────────────────────────────────────────────┘
               │ observation JSON (per macro decision point)
               ▼
┌────────────────────── Python (FlyBrainServer :8792 /decide) ───────────────┐
│ sensory_encoder        obs → 64ch dense vector (groups of 4 seats × 10     │
│                        zone features + global + 12 name-hash channels)     │
│ mushroom_body          tanh → PN→KC fixed random projection (7 PN/KC) →    │
│                        winner-take-all top-10% (406 KCs) → LIF MBON rates  │
│                        (steps=8, dt=0.5ms) → w_mbon_action → masked        │
│                        softmax(τ) → macro action {play,attack,hold,interact}│
│ dopamine_plasticity    episode-end replay: RPE=reward−baseline; DAN gate = │
│                        dan_to_mbon^T·act; ΔW only on active KC rows;      │
│                        STM→LTM blend (decay .9995, mix .5); baseline creep │
│ reward_shaping         potential-ish deltas: life/board/hand/removal/      │
│                        cmdr damage, clip ±0.5; terminal ±1 (win ×len dec.)│
│ forge_agent_client     /decide server + episode journaling (JSONL logs/)   │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼
        scripts/run_match.py (1 game) / scripts/train.py (JVM-amortized
        sessions, checkpoints checkpoints/*.npz, per-episode JSONL)
```

Physical-table mode (`physical/`): `CameraWatcher` holds `/dev/video0`
(1920×1080 MJPEG) continuously feeding `CardTracker` (IoU+centroid matching,
occlusion grace 10f, tap hysteresis: ±30° of 90°, settle 2 frames) →
`PhysicalObserver` → candidate events → `Engine` (vision events needing
confirmation: zone_change/left_battlefield/combat/attack/block) →
`PhysicalGameState` → `to_fly_observation()` in Forge observation shape →
same encoder/brain. Registration (`card_scan.py` + `identifier.py`):
burst → sharpest frame → quad rectify → independent name+collector OCR
(name = primary; tesseract whitelists cannot contain spaces) → fuzzy
Scryfall named/prints lookup → SQLite cache.

## The control boundary (verified in bytecode)

`FlyPlayerController` overrides ONLY `chooseSpellAbilityToPlay()` and
`declareAttackers()`. Everything else inherits `PlayerControllerAi`,
which delegates to `AiController`:

| Decision | Decider | Fly-influence |
| --- | --- | --- |
| Main-phase play (which spell/land) | **Fly** (picks from AiController.chooseSpellAbilityToPlay list by AiPlayDecision score) | direct |
| Attack yes/no | **Fly** (last decision == ACT_ATTACK gates super call) | direct |
| Which attackers | Forge AI (`AiController.declareAttackers`) | none |
| Blocks | Forge AI | none |
| Targets | Forge AI (`chooseTargetsFor` → AiController) | none |
| Instants on others' turns | Forge AI (gate returns empty list outside own mains) | none |
| Mana, damage assignment, SAC choices, mulligan-ish confirms | Forge AI | none |
| Triggers ordering | Forge AI (`orderAndPlaySimultaneousSa`) | none |

**Consequence (important for training validity):** the fly only ever chooses
among options that stock Forge AI generated and scored. The credit for
"play" therefore mixes fly policy with Forge's candidate generation. There
is no path today for the fly to cast a spell Forge AI wouldn't have
proposed. Training signal thus teaches preferences *over Forge's proposal
distribution* — a documented architectural constraint, not a bug.

## State transformation chain & information loss

1. Forge `Game` → `GameObserver` JSON: **loses** card types except creature/
   land counts, all counters, all abilities, mana pool, graveyards' contents,
   hands (counts only — correct), libraries (count), exact permanents beyond
   24 cards, attacker→target mapping (counts only), phase (sent but encoder
   ignores it), player names beyond first 4 seats.
2. JSON → 64ch encoder: **collapses** identity (12 hash buckets, additive
   0.25/counters), phase, stack contents, attacker targets; life normalized
   /40 (Commander-start assumption); **clips** at 2.0.
3. 64ch → 4064-KC WTA: measured overlap ~0.61–0.68 for strategically
   opposite states (see sensory_encoding/encoder_audit.md) — the random
   projection preserves coarse similarity but poorly separates key states.
4. KCs → MBON via LIF (8 steps): MBON activity is a *rate code over a fixed
   window*, not a temporal code; the LIF parameters (τ=20ms sim, 4ms window)
   mean MBON output ≈ saturating linear readout of KC input.
5. Action → Java: only the integer 0..3 crosses back; the specific SA chosen
   Python-side is `options[best]` from Forge's scored list.

## Checkpoints & logs

- `checkpoints/*.npz`: kc_to_mbon (4064×97 f32), w_mbon_action (97×4),
  value_baseline, temperature. **pn_to_kc / dan_to_mbon are NOT saved** —
  loading requires reconstructing the same seed (2049) connectome. Any seed
  change silently invalidates old checkpoints.
- `logs/*.jsonl`: per-episode summary (actions sequence, rewards) + java.log
  capture. No per-decision logging (valences are sent in episode steps but
  trainReward only per step; no state snapshot stored) — insufficient for
  offline post-hoc analysis or counterfactual evaluation.

## Known implementation quirks (verified)

- `GameObserver` includes the fly's own hand **size** and its `canPlay`
  flags — fine; but `board` entries include controller names (leaks nothing
  private; verbose).
- `MushroomBody.decide` uses `self.kc.rng` (the KC population's RNG) for
  action sampling — policy stochasticity is tied to the LIF RNG stream, so
  LIF simulations and action sampling share one generator (determinism
  coupling, harmless but fragile).
- `DopamineSystem.update` computes `dan_gate` from *all* DANs ×
  `dan_to_mbon`, but `dan_activation` sets at most one half of DANs to a
  single scalar — i.e., the "344 DANs" carry exactly one bit of sign and one
  magnitude. The per-DAN structure is currently decorative.
- `value_estimate` (used as critic) is `mean(MBON rates)*2−1` over 8 LIF
  steps — mostly a function of total KC input, weakly action-related.
- The `interact` action reuses `pickBestSa(interactionOnly=True)` which
  filters to instants/Destroy/Counter/DamageAll/Sacrifice from the same
  Forge-proposed list; when empty the fly silently holds (no log event).
- Physical mode: `fly` player's board is digital-only; `to_fly_observation`
  reuses Forge's shape, so reward shaping works unchanged — but
  `canPlay` is hardcoded `{land:False,spell:False,ability:False}`, forcing
  `play`/`interact` masks off in physical mode.
- `AgentMain` ends with `Thread.sleep(15000)` then `System.exit(0)` —
  process lifetime couples to polling cadence.
