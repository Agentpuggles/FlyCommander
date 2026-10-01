# Forge Internals (as used by FlyCommander)

Status: RESEARCH DOCUMENT — no code changes proposed here.
Last verified: 2026-10-01 (bytecode inspection of the exact jar in use).
Confidence labels follow [research/README.md](../README.md) conventions.

---

## 1. Version pinning

| Item | Value | Evidence |
|---|---|---|
| Forge version | **2.0.15** | jar filename, manifest read via `unzip -p` |
| Jar | `/home/flynn/Downloads/mtg forge/forge-gui-desktop-2.0.15-jar-with-dependencies.jar` | filesystem |
| Patch classes | `/home/flynn/Downloads/mtg forge/forge-agent-patch/classes` | build output of `forge_patch/` |
| Decks | `~/.forge/decks/commander/` | DeckResolver behavior |
| Untouchable | `~/.forge` state dir and Forge `res/` resources | project convention |

**Version-sensitivity warning (HIGH):** every statement below is true for
2.0.15. Forge is a fast-moving project; class names, method signatures, and AI
behavior can change between releases. Re-verify before any jar upgrade.

---

## 2. Boot path (bytecode-verified)

Traced by disassembling `fly.agent.AgentMain` with `javap -c` against the jar:

```text
AgentMain.main
  → FModel.initialize            (loads card db, static data, preferences)
  → DeckResolver                 (resolves commander decks from ~/.forge/decks/commander)
  → GameObserver                 (wraps game lifecycle, serves observations)
  → AgentServer                  (HTTP server bridging to Python BrainClient)
  → GameEngine / Match           (Forge match construction)
```

Key facts:

1. **`FModel.initialize` is mandatory and expensive.** It parses the full card
   database and static data. This is the dominant JVM startup cost (measured
   order: tens of seconds on this machine). FlyCommander amortizes it by
   running many games per JVM process.
2. **`DeckProxy`** is how decks are surfaced to the lobby/match layer; our
   `DeckResolver` selects commander decks from `~/.forge/decks/commander/` and
   hands them to the match. Deck file format is Forge's `.dck` plaintext
   format.
3. **Randomness** flows through Forge's `forge.util.MyRandom`, a seeded
   `Random`-like shim. Consequences for reproducibility:
   - Shuffling/coin-flips are deterministic *given the same seed sequence*,
     but our patched path does not currently pin the per-game seed — seeds are
     process-lifetime dependent.
   - Python-side RNGs (brain init, encoder probe) *are* explicitly seeded
     (seed 2049). Java-side is the reproducibility gap.
   - **Experiment implication:** any A/B experiment must either pin Forge RNG
     per game (requires patch) or run enough games that Forge-side variance is
     averaged out (see experiments/roadmap.md, E-TRAIN-02).

---

## 3. Forge AI architecture (from Forge `docs/AI.md` + bytecode)

Forge's own documentation (`docs/AI.md` in the source tree) and the jar's
class list confirm this structure:

- `forge.ai.PlayerControllerAi` — the AI implementation of the controller
  interface. All decision entry points live here and in helper classes.
- **Spell decisions:** `forge.ai.SpellAbilityChooser` scores candidate
  `SpellAbility` objects each priority window and picks one.
- **Card evaluation:** per-card heuristic scorers (`forge.ai.CardEvaluator`,
  card-specific overrides in `forge.ai.SpecialCardAi/*`).
- **Combat:** dedicated classes for choosing attackers/blockers and damage
  assignment; heavy use of simulations and heuristics rather than search.
- **Targeting:** candidates are ranked by threat/heuristic scores; the chooser
  picks among them.
- There is **no Monte-Carlo search or learned model** in stock Forge AI — it
  is entirely hand-written heuristics. Its strength varies widely by deck and
  by situation (e.g., combat math is decent; subtle stack interaction is weak).

**What this means for FlyCommander:** the "candidates" our fly picks among are
the products of a fixed heuristic pipeline. If Forge's proposals are weak, the
fly inherits that ceiling. The fly can never propose an option Forge didn't
offer. This is the single biggest architectural confound and is documented in
[architecture/flycommander_current.md](../architecture/flycommander_current.md).

---

## 4. The patch surface (8 Java files)

Our classes under `forge_patch/src/fly/agent/`:

| Class | Role |
|---|---|
| `AgentMain` | entrypoint: FModel boot, match construction |
| `AgentServer` | HTTP server; episode/decision loop; JSON plumbing |
| `FlyLobbyPlayer` | lobby-level player representation |
| `FlyPlayerController` | overrides **only** `chooseSpellAbilityToPlay` (own main phases, ≤8 activations/phase) and `declareAttackers` (gated on last macro decision = ACT_ATTACK); everything else delegates to Forge AI |
| `AgentGameState` | extracts the observation dict (life/zones/board names) |
| `GameObserver` | lifecycle + observation snapshots |
| `BrainClient` | HTTP client → Python; sends observation, receives action |
| `DeckResolver` | commander deck selection |

Bytecode verification confirmed FlyPlayerController overrides exactly the two
methods above and inherits the rest from `PlayerControllerAi` (see
[architecture/flycommander_current.md](../architecture/flycommander_current.md)
control-boundary table for the full who-decides-what list).

---

## 5. Forge ≠ Comprehensive Rules (known divergences to re-check)

Forge is *extremely* rules-faithful for most interactions but is not a
certified implementation. Known classes of divergence relevant to us:

1. **Card-specific hack classes** (`SpecialCardAi`, `CardScript*`, spell
   abilities marked `Choice[0]`-style special handling) — some cards behave
   via bespoke code paths that can differ in edge cases.
2. **Unimplemented/buggy cards** exist in any release (Forge tracks known
   issues in its changelog); a deck built around such a card would train the
   fly on false rules dynamics.
3. **AI heuristics are not rules.** E.g., Forge's proposal to not block can be
   strategically wrong; that is a *proposal-quality* issue, not a rules issue.
4. Forge's commander implementation follows the rules for commander tax and
   21-commander-damage loss (Comp Rules 903.10a); Duel-Commander-style rule
   differences (see [magic/comprehensive_rules_findings.md](../magic/comprehensive_rules_findings.md))
   are **not** in Forge's Commander format.

**Discrepancy protocol:** if the fly's behavior seems rules-based but odd,
first check Forge's card-script for the specific card (grep `res/` is banned at
runtime but reading it for research is fine), then the Comp Rules, and record
which authority disagreed.

---

## 6. State transformations on the bridge path

```text
Forge GameState (full: zones, cards, stack, effects, timers)
  → AgentGameState.buildObservation()      [LOSSY: counts + names only]
  → JSON over HTTP (BrainClient → AgentServer)
  → observation_to_state()                 [LOSSY: 64 dense channels]
  → PN activity → sparse KC code           [LOSSY: WTA keeps ~10%]
  → MBON rates → softmax over 4 actions    [irreversible]
```

Loss points documented in detail in
[sensory_encoding/encoder_audit.md](../sensory_encoding/encoder_audit.md). The
bridge-level loss (names→counts, no stack contents, no mana, no board
identities beyond the fly's own 12 hash buckets) happens **before** the encoder,
so no encoder redesign can recover it.

---

## 7. Verification method

- `javap -c -p` on classes inside the 2.0.15 jar (boot path, controller
  overrides, MyRandom usage).
- Forge `docs/AI.md` (bundled with the jar's source distribution).
- Class listing via `unzip -l` filtered on `forge/ai/`, `forge/game/`,
  `forge/deck/`, `forge/util/`.
- Cross-check against [FlyPlayerController.java](../../forge_patch/src/fly/agent/FlyPlayerController.java)
  source in this repo.

Any future jar change invalidates this document; re-run the javap checks above
and update the version table.
