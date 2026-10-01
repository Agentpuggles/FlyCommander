# 🪰 FlyCommander

A fruit-fly-brained Magic: The Gathering **Commander** agent. The decision
circuit is a spiking model of the *Drosophila* mushroom body — 4,064 Kenyon
cells, 97 MBONs, 344 dopaminergic neurons with the real circuit's statistics —
wired into a patched [Forge](https://github.com/Card-Forge/forge) rules engine
over HTTP, so the fly literally plays headless Commander games, loses with
dignity, and gets better through a dopaminergic plasticity rule.

It also plays on a **physical table** against you: real cards, a camera, and a
hybrid vision + manual-correction state engine feeding the same brain
(see [Physical-table mode](#physical-table-mode)).

**Verified working:** a full pipeline where the untrained fly played complete
multiplayer Commander games against Forge's AI (21–31 turns, 100–176 spiking
decisions per game), with decks drawn from Forge's own Commander deck pool
(`random` specs resolved inside Forge), per-episode DPR learning, JSONL
journals, and checkpointing — all in a ~3-minute JVM session.

```
                    ┌──────────────────────────────────────┐
 webcam → vision/   │  Java patch (forge-agent-patch/)     │
 (planned)          │  fly seat = PlayerControllerAi subclass
                    │  AgentServer HTTP :8791              │
                    └──────────────┬───────────────────────┘
                                   │ observation JSON
                                   ▼
        sensory_encoder.py  →  64-ch PN vector
                                   ▼
        brain/mushroom_body.py  →  sparse KC code (~10% of 4,064)
                                   ▼  KC→MBON plastic synapses (LIF)
        macro action 0..3  ◄────  97 MBONs, masked softmax
        (play / attack / hold / interact)
                                   │
        Forge AI handles micro decisions: targets, blockers, mana
                                   ▼
        reward_shaping.py (Python-side)  →  RPE
                                   ▼
        brain/dopamine_plasticity.py  →  DPR update, STM→LTM
```

## Layout

| Path | What it is |
| --- | --- |
| [brain/](brain/) | LIF engine, synthetic MB connectome, decision circuit, dopamine plasticity |
| [tests/](tests/) | pytest suites: brain dynamics, encoder, rewards, physical vision, camera config, card-detection pipeline |
| [forge_patch/](forge_patch/) | Java agent sources (compiled into `forge-agent-patch/classes`) |
| [flycommander/](flycommander/) | Forge HTTP client, `/decide` brain server, state encoder, reward computer |
| [scripts/](scripts/) | `run_match.py` demo driver, `train.py` training loop |
| [vision/](vision/) | Physical-table perception: MTG card-detection pipeline ([card_analysis.py](vision/card_analysis.py)) |
| [live/](live/) | Camera→suggestion pipeline stub |
| [connectome/](connectome/) | Real MaleCNS extraction plan/stub |

## Quickstart

```bash
make test      # full pytest suite, no Java needed
make run       # one full Commander match: fly (random deck) vs random AI decks
make train     # 5-game training session with checkpoints
```

### Deck selection (Forge-owned)

Deck discovery and randomization happen **inside Forge**, using the same pool
as the GUI's *Commander → Commander Decks* screen
(`DeckProxy.getAllCommanderDecks()` over `FModel.getDecks().getCommander()`,
randomized with Forge's own `MyRandom`). Every `--fly-deck` / `--ai-deck`
spec accepts:

| Spec | Meaning |
| --- | --- |
| `random` | Forge picks from its normal Commander Decks pool (repeat `--ai-deck random` for independent picks per seat) |
| `<deck name>` | a deck in Forge's Commander deck storage (e.g. `"Atraxa AI Deck"`) |
| `<path>.dck` | an explicit deck file (back-compat) |

```bash
python3 scripts/run_match.py --games 1 --fly-deck "Atraxa AI Deck" \
    --ai-deck random --ai-deck random --seed 42
```

Forge logs its resolution (`[FlyAgent] fly deck: …`, `[FlyAgent] ai deck 2: …`,
with pool size and random pick index) to `logs/java.log`, and the resolved
names — straight from the `Deck` objects — come back through `/health` and are
printed by the Python driver.

Requirements: Python 3.10+ with NumPy, Java 17+ (OpenJDK 25 verified), and the
Forge 2.0.15 desktop jar at `/home/flynn/Downloads/mtg forge`.

## How the fly thinks

- **Sensory encoding** — the Java agent publishes a structured observation
  (life, zones, board identities, stack, combat). [sensory_encoder.py](flycommander/sensory_encoder.py)
  compresses it to 64 projection-neuron channels; the MB's fixed random PN→KC
  projection and APL-style winner-take-all produce a ~10%-sparse Kenyon-cell
  code (in the fly, ~7 PNs converge per KC — replicated here).
- **Decision** — KC spikes drive MBON populations through LIF dynamics;
  MBON activity scores four macro actions. Illegal actions are masked
  (e.g. no "play" if nothing is castable). Temperature-annealed softmax
  exploration. If the Python brain is unreachable, the seat falls back to
  stock Forge AI mid-game — the show always goes on.
- **Learning** — rewards are computed Python-side from consecutive
  observations (board, life, commander damage, removal, card advantage) plus
  the terminal ±1. The RPE (reward − running baseline) activates PAM-like
  appetitive or PPL1-like aversive DANs; the DPR updates only the KC→MBON
  synapses that were active at decision time, with STM→LTM consolidation.
- **Java patch** — `AgentMain` boots Forge headless exactly like the
  `sim` CLI mode, builds a Commander match from `.dck` files, and seats
  `FlyBrain` via `FlyLobbyPlayer` → `FlyPlayerController`. The fly decides on
  its own main phases; everything else inherits Forge AI for speed and
  legality.

## Honest limitations (v1)

- Macro decisions only, on the fly's own main phases; instant-speed
  interaction passes through to Forge AI.
- Synthetic connectome with real statistics; the real-MaleCNS loader is
  stubbed in [connectome/download_malecns.py](connectome/download_malecns.py).
- Reward shaping and the 64-channel encoder are hand-designed; the fly's
  value baseline is a running mean, not a learned critic.
- Untrained, the fly loses to Forge AI (it hold-heavy-plays at first);
  expect slow improvement over hundreds of games, not AlphaZero.
- Recognition accuracy is trained on *synthetic* cards (`vision/synthetic.py`),
  because a real photos-of-real-cards dataset does not exist in this repo. The
  pipeline is real (detector → homography → embedding → Scryfall); the training
  data is a stand-in. `--sync-scryfall` builds the index from real card images,
  and these weights can be fine-tuned on real captures as they accumulate
  (`high-confidence confirmed` scans are exactly the labels to use).
- The Forge-side table endpoints (`forge_patch/src/fly/agent`, reviewed against
  Forge 2.0.15) are not compiled in CI here — `make compile` against a Forge jar
  is the verification step; the Python bridge is covered by tests.

## Training curriculum (implemented in `train.py`)

Start 1v1, expand later: `--ai-deck` is repeatable for 3–4 player pods.
Each JVM session amortizes Forge's ~60–90 s boot across `--games-per-vm`
games; checkpoints land in `checkpoints/`, journals in `logs/`.

## Physical-table mode

Play the fly with real cards. Forge remains the training environment; the
physical table is a second environment feeding the **same** brain.

```bash
make physical          # → http://127.0.0.1:8795
# or: python3 scripts/physical_table.py --port 8795 \
#        --checkpoint checkpoints/fly_ep50.npz --offline
```

### Neural recognition (primary path)

Cards are recognised from pixels — no per-card registration, no OCR-first
pipeline, and no requirement to flatten the card:

```
camera frame
  → detector            card quads: edge/structure/contrast evidence, aspect +
                        size priors, cardness model (vision/detector.py)
  → rectify             four-corner homography → canonical 492×688 portrait,
                        glare-masked, tilt (0°/90°) reported (vision/rectify.py)
  → embeddings          multi-scale card descriptor + optional CNN (dim 256,
                        ONNX → torch → dense-head → dense fallback chain,
                        never raises) (vision/embeddings.py)
  → matcher             fused rank over the whole index: embedding + artwork +
                        layout + title + collector + colour, all four
                        rotations of the card index, calibrated confidence,
                        same-name printings flagged ambiguous (vision/matcher.py)
  → tracking            per-frame association (quad IoU / containment /
                        centroid) + confidence-weighted vote window: one card
                        in view is one track, an identity is only published
                        after `stable_votes` agreeing frames
  → Scryfall            set+collector lookup, local Forge image cache reused
                        before any download (cards/scryfall.py)
  → Forge state update  tap / zone / counters through the event engine
```

Play flow:

- **Identify all** (`POST /api/vision/identify`) reads the current camera
  frame and returns every card it can see, with the rectified crop, the
  ranked candidates and the geometry (tilt, glare, coverage).
- **Scan card** (`POST /api/register/frame`) keeps the one-card workflow: it
  picks the sharpest frame of a burst, identifies it, and auto-registers a
  confident, stable match onto the battlefield (`auto_accept_confidence`,
  default 0.66). Uncertain matches are offered as tappable candidates.
- **Not in library?** A recognised-but-unknown card can be confirmed with one
  tap; confirming adds it to the recognition index (`addToIndex`) so it is
  found instantly next time — the library grows from play, never from manual
  data entry.
- The vision bar under the camera (`GET /api/vision/status`, also embedded in
  `/api/state`) shows the live index size, detector and embedder tier, and
  offers **Build index** when none exists.

Measured accuracy (synthetic benchmark, `scripts/eval_vision.py`, 300 cards
held out from training, mixed conditions — glare, foil, sleeve, blur,
occlusion, rotation):

| Stage | Result |
| --- | --- |
| Recognition (rectified captures, no detector) | top-1 **61.8 %**, top-3 72.5 %, top-5 78.2 % |
| — clean / occlusion / sleeve | 70 % / 78 % / 65 % top-1 |
| — hardest conditions (foil, glare, blur) | 55 % / 58 % / 53 % top-1 |
| End-to-end (detector + rectification + matching, 39 cards) | top-1 **74.4 %**, and **"unknown" instead of a wrong card** when the fused score is low |
| Detection — *does the crop contain the card?* | recall **0.98** (47/48 cards over 24 mixed scenes) |
| Detection — *are the corners tight?* (IoU > 0.5) | recall 67 %, corner error ≈ 31 px |
| Latency | detect ≈ 177 ms, match ≈ 49 ms per card (CPU, 1 thread) |
| Zero-training tier (no weights, `dense` descriptor) | top-1 ~15 % — training is what buys the accuracy |

The two detection rows measure different things on purpose: the pipeline only
needs a crop that *contains* the card (the rectifier snaps edges to the true
border before matching), while tight corners are still the main source of
head-room — a large over-crop lowers confidence and costs the strict metric.

These are synthetic-card numbers: a real camera adds glare, foil shimmer and
depth of field that the generator only approximates, and a real library is
thousands of cards, not 300. Treat the table as a **co-pilot that is right most
of the time and admits when it is not**, not as a barcode scanner — the UI
always shows the ranked candidates and offers one-tap confirmation.

Building the index:

```bash
# real cards: Scryfall bulk data + on-demand images
.venv/bin/python scripts/physical_table.py --sync-scryfall --images 4000

# offline demo library (synthetic cards, good for a first run)
.venv/bin/python scripts/physical_table.py --build-index 64
```

Training the recogniser (optional but recommended — it lifts recognition well
above the zero-training descriptor):

```bash
.venv/bin/python scripts/train_embedder.py --cards 400 --steps 1500
# writes vision/weights/{embedder.pt,embedder.onnx,embedder.json,
#                       dense_head.npz,training_report.json}
.venv/bin/python scripts/eval_vision.py --mode all --cards 300 --calibrate
# writes logs/eval_vision.json + vision/weights/calibration.json
```

The camera watcher runs the same pipeline continuously (analysis decoupled
from capture, `max_analysis_fps` default 6 Hz) so taps, zone changes and new
cards update the game state without pressing anything.

### Forge owns the rules (physical → Forge mirror)

The camera side observes; it never judges. Every physical fact that survives
reconciliation is forwarded to the running Forge game, which remains the only
rules engine in the system:

```
camera → detection → rectification → recognition/tracking
       → physical event (card put down / turned sideways / moved zone /
         +1/+1 counter / life changed / attacked with)
       → flycommander/forge_table_bridge.py   (translate, spool, retry)
       → POST /table/events  →  forge_patch/src/fly/agent  →  Forge
       → Forge validates it, updates the real game state
       → the fly seats and the UI read that state back (GET /observation)
```

| Piece | Where |
| --- | --- |
| Event → action mapping | `flycommander/forge_table_bridge.py` (`EVENT_TO_ACTION`) |
| Wire contract | `POST /table/events`, `POST /table/state`, `GET /table/queue` |
| Forge side | `forge_patch/src/fly/agent/{AgentServer,TableActionQueue,TableActionApplier,PhysicalTableController,TableLobbyPlayer,ForgeApi}.java` |
| UI panel | the **forge** line under the camera (`GET /api/forge/status`, *Sync to Forge*) |

Properties of the link, all covered by `tests/test_forge_bridge.py`:

- **No event can fall on the floor.** Every event type in `physical/events.py`
  is either mapped to a Forge action or explicitly listed as tracking-only,
  and the test suite enforces that partition.
- **Never raises, never stalls the table.** Forge may simply not be running:
  actions spool in order (bounded, with a drop counter) and flush on the next
  successful contact. `FLYCOMMANDER_FORGE_SYNC=0` disables the mirror;
  `FLYCOMMANDER_FORGE_AGENT` points at a non-default agent.
- **Idempotent.** Actions carry `actionId = <tableId>:<seq>`; Forge ignores an
  id it has already seen, so a retried batch cannot double-apply.
- **Honest.** Forge's verdict per action (`applied` / `rejected` + reason) is
  reported back and shown in the UI; the bridge never claims a game action
  happened because the camera saw something.
- **Forge stays the judge.** Applying an action means asking Forge to perform
  it (`moveTo`, `setTapped`, `addCounter`, `setLife`); an illegal physical move
  is refused by Forge with a reason, not silently accepted.

Run the table with a physical seat in Forge:

```bash
make compile                       # javac the patch against your Forge jar
java -Dfly.agent.table=1 -cp "$PATCH_CLASSES:$FORGE_JAR" fly.agent.AgentMain 1 random random random random
.venv/bin/python scripts/physical_table.py --port 8795   # same machine
```

> **Compile status:** the table-action endpoints and the applier are written
> against the Forge 2.0.15 API surface, but this repository's CI has no Java or
> Forge jar, so only the Python half is test-verified here. Run `make compile`;
> all Forge API assumptions are gathered in
> `forge_patch/src/fly/agent/ForgeApi.java` so a version bump is a one-file fix.

Architecture (all under [physical/](physical/)):

```
server-owned camera (V4L2, MJPG 1920x1080@30, verified negotiation)
   → single capture loop (one V4L2 owner): native-rate read (~24 FPS),
     analysis decoupled at 5 Hz, preview encode at 10 Hz
   → every frame, no OCR: analyze_frame()
       presence → geometry → quality → perspective correction → (OCR on Scan)
   → CardTracker (stable IDs, IoU matching, occlusion grace, tap hysteresis)
   → PhysicalObserver → candidate events
   → Engine (reconciliation: vision proposes, rules + player dispose)
   → PhysicalGameState (authoritative, public-only)
   → to_fly_observation() → existing sensory encoder → mushroom body
   → fly decision → spectator UI (fly board, brain panel, event log)
```

- **Registration (success-first, multi-signal)** — hit *Scan card*: the
  server samples a burst from its own camera, picks the sharpest frame,
  rectifies the full card, and runs **two independent OCR regions** — the
  card **name** (primary) and the collector/set line (secondary). Either can
  succeed alone; fuzzy matching handles OCR noise ("Doublinq Season" →
  Doubling Season); exact set+number Scryfall lookup disambiguates
  reprintings. Manual set+number entry remains as fallback.
- **Honest scan states** — the pipeline answers *what it actually sees* and
  never claims "OCR failed" when there is no card. The order is strictly
  presence → geometry → quality → OCR; OCR only runs after a card has been
  found and framed. Every scan returns a `state` with a human message,
  `card_detected`, `confidence` and a `reason`:

  | State | Message | UI color |
  | --- | --- | --- |
  | `no_card` | No Magic card detected. Place one card in view. | red |
  | `multiple_cards` | Multiple cards detected. Scan one card at a time. | red |
  | `too_small` | Move the card closer. | red |
  | `bad_quality` | Image quality low: improve lighting / reduce glare / hold still (specific hints) | yellow |
  | `ocr_failed` | Card detected, but text cannot be read. | yellow |
  | `card_detected` | Card detected (any tilt — perspective corrected). | green |

  **There is no "flatten the card" failure.** This is a tabletop observer,
  not a document scanner: cards at any human angle are perspective-corrected
  (four-corner homography → canonical 63×88 image) and read like flat cards;
  `analysis.perspective` reports `corrected`/`flat` and angled cards simply
  carry slightly lower confidence.

  Geometry uses MTG card properties (63×88 mm, aspect 0.716, ±14% match
  band) so sleeves, rotation and perspective don't break detection.
  Lighting is checked before blur — darkness depresses the Laplacian and
  would otherwise masquerade as blur. Identification fuses multiple signals:
  card-name OCR (primary), collector/set OCR, and **artwork similarity**
  (average-hash of the rectified art zone vs the cached Scryfall art — see
  [vision/artmatch.py](vision/artmatch.py)); the debug panel shows the full
  per-candidate confidence breakdown. A card that OCRs weakly but is
  well-formed still reports `card_detected`; only a genuinely unreadable
  card reports `ocr_failed`.
- **Camera control** — the watcher opens `/dev/video0` via the **V4L2**
  backend, forces **MJPG**, **1920×1080 @ 30 FPS**, and verifies the
  negotiated settings by reading them back from the driver. The C922's
  2304×1536@2fps YUYV stall mode is detected and rejected — never silently
  accepted (any mode under 5 FPS measured is a stall). If the preferred mode
  fails, it walks the fallback chain **1280×720@60 → 1280×720@30 →
  960×720@30**. On startup it prints exactly what it got:

  ```
  Camera initialized:
   Device: /dev/video0
   Format: MJPG
   Resolution: 1920x1080
   FPS: 30
  ```

  A runtime watchdog watches measured FPS and reopens the camera with
  backoff after stalls or grab failures.
- **Camera configuration** — settings load with increasing precedence:
  built-in C922 defaults → `physical/camera_config.json` → repo-root
  `camera_config.json` → `FLYCOMMANDER_CAMERA_*` environment variables.
  Keys (see [physical/camera_config.example.json](physical/camera_config.example.json)):

  | Key | Default | Meaning |
  | --- | --- | --- |
  | `camera_device` | `/dev/video0` | V4L2 device node (a missing node is a startup warning, not a config error) |
  | `camera_width` | `1920` | requested width (first of the fallback chain) |
  | `camera_height` | `1080` | requested height |
  | `camera_fps` | `30` | requested frame rate |
  | `camera_format` | `MJPG` | FOURCC; `MJPEG`/`JPG`/`MPEG` are normalized to `MJPG` |

  Env equivalents: `FLYCOMMANDER_CAMERA_DEVICE`, `FLYCOMMANDER_CAMERA_WIDTH`,
  `FLYCOMMANDER_CAMERA_HEIGHT`, `FLYCOMMANDER_CAMERA_FPS`,
  `FLYCOMMANDER_CAMERA_FORMAT`. Inspect the effective settings without
  starting the server:

  ```bash
  .venv/bin/python scripts/physical_table.py --show-camera-config
  # or with an explicit file:
  .venv/bin/python scripts/physical_table.py --show-camera-config \
      --camera-config physical/camera_config.example.json
  ```
- **Debug view** — the 🐞 panel shows the live **annotated** camera frame
  (`/api/debug/frame`): candidate outlines with corner points in green
  (valid card), yellow (uncertain) or red (no valid card), the current scan
  state and the camera settings burned into the frame. Per scan it also
  shows the perspective-corrected card, both OCR crops with confidences, and
  per-candidate confidence breakdowns (name OCR, artwork similarity,
  combined). `GET /api/debug/info` returns the same data as JSON, including
  per-stage timings (capture/analysis/tracker/preview ms, OCR/identify ms).
  The raw MJPEG stream is frame-deduplicated (no stale replays).
- **Tracking** — cards keep stable tracking IDs through movement and brief
  occlusion (configurable grace period). Tap detection: smoothed orientation
  within a configurable tolerance of 90°, settled over consecutive frames.
- **Hybrid state** — vision taps/untaps automatically; zone changes, combat
  and damage require player confirmation (✓/✗ in the UI). Counters, buffs,
  abilities, life and tokens are one-click annotations on any card.
- **Summoning sickness** — rules-derived from `turns_started` per controller
  (never inferred from pixels); `haste` exempts; correctable.
- **Hidden information** — hands/libraries are counts only; the fly receives
  state exclusively through `to_fly_observation()`; the UI never shows the
  fly's hand.

Verified working: 173-test suite green under the project `.venv`
(OpenCV 5.0 + Tesseract 5.5) — including camera configuration, the
perspective-tolerant scan pipeline (synthetic keystone scenes: 30°/45°
cards detected and rectified, never rejected), artwork-similarity matching,
and crash-proof watcher/detection handoffs; the full suite also passes
under a cv2-free Python (vision tests skip, nothing crashes). Camera
negotiation, single-owner capture and the state pipeline are unit-tested
with synthetic frames; real-card behavior under your exact lighting is
tuned via the 🐞 debug view. Without OpenCV/Tesseract, manual registration
still works.
