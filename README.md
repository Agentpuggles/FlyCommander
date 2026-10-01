# 🪰 FlyCommander

A fruit-fly-brained Magic: The Gathering **Commander** agent. The decision
circuit is a spiking model of the *Drosophila* mushroom body — 4,064 Kenyon
cells, 97 MBONs, 344 dopaminergic neurons with the real circuit's statistics —
wired into a patched [Forge](https://github.com/Card-Forge/forge) rules engine
over HTTP, so the fly literally plays headless Commander games, loses with
dignity, and gets better through a dopaminergic plasticity rule.

The project also includes a physical-table card scanner and an in-progress
browser-controlled Forge Commander prototype (see [Commander pod prototype](#commander-pod-prototype)).
The browser pod uses Forge's virtual deck and hand. Physical-card identity,
physical library synchronization and camera-observed actions are not integrated
into rules play.

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

Physical-table mode contains two separate tools: the card-scanner/state-mirror
prototype below, and `/play`, the assisted Forge Commander pod described later.
Scanner observations do not directly cast cards or change the pod's game state.

```bash
make physical          # → http://127.0.0.1:8795
# or: python3 scripts/physical_table.py --port 8795 \
#        --checkpoint checkpoints/fly_ep50.npz --offline
```

### Recommended: real-artwork deck scanner

The deck scanner uses **SIFT artwork features + RANSAC perspective verification**,
not the experimental untrained neural descriptor or synthetic demo cards. It
locates matching artwork directly in a frame, including rotated cards, without
requiring readable name text or a successfully detected outer border.

```bash
make physical-setup
make physical-check
make physical
# Remote server / browser camera only:
.venv/bin/python scripts/physical_table.py --no-camera
```

1. Open the table UI and paste your deck under **Deck scanner setup**. One name
   per line works; an Arena export with the exact printing is better:
   `1 The Gitrog Monster (SOI) 245`.
2. Click **Import real card images**. Progress and individual failures are shown.
   Name-only imports fetch up to four recent distinct illustrations per name;
   they do **not** cover every printing. Specify your set and collector number
   when your artwork differs. Both image-bearing faces of double-faced cards
   are imported when their artwork layout is supported. Sagas, Classes, Cases,
   Rooms, split/battle/flip and other unusual layouts are conservatively refused
   by artwork import (use Add by name). Adventure recognition and transforming existing
   battlefield objects are not specifically implemented by this scanner.
3. Choose **Use this device’s camera**, the server camera, or upload a photo.
   Browser capture needs HTTPS or localhost and camera permission. The browser
   sends frames to the same server only on Scan/Identify or when you explicitly
   enable live identification. Nothing is sent to a third-party vision service.
4. **Scan card** offers the largest verified card. **Identify all** / optional
   **Live identification** show verified cards and reference pictures. Check
   the name, then confirm to add to the battlefield. Separate copies can be
   added; repeated confirmation of the same bound track does not add duplicates.

All artwork matches require confirmation. Match strengths are heuristics, **not
calibrated accuracy percentages**. Shared art cannot prove an exact printing.
Ambiguous identities are offered for checking, not silently registered. An
unverified frame produces no reference identity; Scan may try OCR fallback, but
also requires confirmation when a reference library is active. The older
no-reference mode retains its legacy automatic-registration policy.

Reference pictures and their manifest live under
`data/physical/cards/references/` (ignored by Git). Imports are explicit,
background, throttled, incremental and bounded to 150 unique input rows / 600
reference faces. The library loads on restart and visual scans work offline.
Use a separate `--data-dir` for another library. Importing requires Scryfall
access. Network errors are reported, not replaced with demo cards. OCR rescue
requires the separate OS `tesseract-ocr` package; artwork matching does not.

**Limitations:** this is a deck-sized local reference matcher, not a whole-Magic
recognition service or demonstrated SpellTable/Convoke equivalent. Different art,
heavy sleeves/glare, motion blur, very small cards, extensive occlusion and
unusual layouts can fail. Automated tests use procedural art with perspective,
rotation, exposure, blur, negative examples and multiple copies; they do not
establish accuracy on physical-camera footage. A separate smoke check using a
real Gitrog reference with digital perspective/rotation/exposure/blur also passed
at four rotations; this is still not a webcam benchmark. Browser live mode currently
identifies cards; the server-owned camera watcher remains the continuous
movement/tap-observation path. Forge must still be connected for its rules/AI;
recognition alone does not implement a complete paper Magic rules engine.

Validation commands (network-free; Node is optional for the UI smoke test):

```bash
.venv/bin/python -m pytest tests/ -q
node tests/ui_scanner_smoke.cjs
```

### Experimental neural recognition (legacy path)

The following describes the older optional index, not the recommended deck scanner.
It requires suitable real images and a trained/calibrated model for useful accuracy.
The default UI no longer seeds fictional demo cards.


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

### How a card actually gets identified (three tiers, honest failures)

The visual index is the fast path, but it only works for cards it has seen.
A scan therefore falls through three tiers, and each one is allowed to say
"I don't know" — a 5 % neighbour from a demo library is noise, not a
suggestion, and it is never offered as a tappable choice
(`min_offer_confidence`, default 0.20):

1. **Visual index** — detector → rectify → embedding → rank. Used when the
   match clears `suggest_confidence` (0.35) and the matcher does not flag it
   `unknown`.
2. **OCR + Scryfall rescue** — when the visual match is weak or unknown, the
   scan reads the card's *own text* (name and collector line, on the
   perspective-corrected crop) and asks Scryfall by name. This needs no image
   library at all, which is what makes a real card playable on a fresh table
   with only the synthetic demo library installed. Confident answers
   (≥ `ocr_auto_accept_confidence`, 0.60) register directly; weaker ones are
   offered as candidates. Runs on an explicit Scan only — never per frame.
3. **Add by name** — the player types `gitrog monster`; Scryfall's fuzzy
   endpoint tolerates typos and partial names, and every answer is cached so
   the same card resolves offline next time (`POST /api/register/name`).

The UI always says where a name came from: *closest in library (visual)*,
*name (OCR)*, or the player's own typing — so a low-confidence library guess
can never be mistaken for what the camera read.

At startup the table prints which of those paths can actually answer:

```
[vision] library: 48 cards (0 real / 48 synthetic demo), 48 images
[vision] OCR + Scryfall rescue: available (tesseract 5.3.4)
[vision]   → the demo library cannot name real cards from pixels; Scan reads the card's text instead
[vision]   → embedder is untrained, so visual matching is weak: scripts/train_embedder.py is what buys accuracy
```

Only tier 1 runs continuously on the live camera feed; tiers 2 and 3 are
Scan-time or typed, because OCR is far too expensive to run per frame. Automatic
hands-free recognition of a real card therefore needs both a real image library
(`--sync-scryfall --images N`) **and** a trained embedder
(`scripts/train_embedder.py`) — with the demo library and an untrained embedder,
a real card is correctly reported as *not in library*, and Scan or "Add by name"
is what puts it on the table.
- The vision bar under the camera (`GET /api/vision/status`, also embedded in
  `/api/state`) shows the live index size, detector and embedder tier, and
  offers **Build index** when none exists.

Measured accuracy (synthetic benchmark, `scripts/eval_vision.py`, 300 cards
held out from training, mixed conditions — glare, foil, sleeve, blur,
occlusion, rotation):

| Stage | Result |
| --- | --- |
| Recognition (rectified captures, no detector) | top-1 **54.3 %**, top-3 70.0 %, top-5 75.7 % |
| — clean / occlusion / blur / sleeve | 60 % / 65 % / 55 % / 55 % top-1 |
| — worst conditions (foil, glare) | 45 % top-1 |
| Detection | recall ~0.95, corner error ≈ 20 px, ≈ 155 ms/frame |
| End-to-end | top-1 60 % on a small scene sample, and **"unknown" instead of a wrong card** when the fused score is low |
| Zero-training tier (no weights, `dense` descriptor) | top-1 ~15 % — training is what buys the accuracy |

These are synthetic-card numbers: a real camera adds glare, foil shimmer and
depth of field that the generator only approximates, and a real library is
thousands of cards, not 300. Treat the table as a **co-pilot that is right most
of the time and admits when it is not**, not as a barcode scanner — the UI
always shows the ranked candidates and offers one-tap confirmation.

> **Reproducing them (measured 2026-10-01).** The table above is the
> *trained*-embedder result, and `scripts/train_embedder.py` needs `torch`
> (not installed by default). Out of the box — no weights, `dense` descriptor,
> exactly the command above (`--cards 300 --scenes 40 --per-condition 40
> --calibrate`, seed 5) — this repo currently measures:
>
> | Stage | Claimed (trained) | Measured (untrained, fresh clone) |
> | --- | --- | --- |
> | Recognition top-1 | 54.3 % | **30.0 %** (clean 42.5 %, foil 17.5 %, glare 20.0 %) |
> | End-to-end top-1 | 60 % | **50.0 %** |
> | End-to-end wrong-card rate | “unknown instead of a wrong card” | **29.2 %** ⚠️ |
> | Detection recall / corner error | ~0.95 / ≈20 px | **0.61 / 27.3 px** |
> | Detection latency | ≈155 ms/frame | **≈208 ms/frame** |
>
> Two things follow. First, **training is not optional** if you want the
> published numbers. Second, with an untrained embedder the pipeline answers
> the *wrong card* on ~29 % of scenes instead of saying “unknown”, which
> contradicts the design rule above: until `train_embedder.py` has run, treat
> recognition as “OCR + Scryfall, confirmed by you” and raise
> `auto_accept_confidence` if you want it to stop auto-registering. This is
> the gap tracked as open question #13 in
> [research/open_questions](research/open_questions/open_questions.md).

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

The camera watcher runs the same pipeline continuously so taps, zone changes
and new cards update the game state without pressing anything. It uses three
threads — **capture** (reads at the device rate and nothing else),
**analysis** (detection + matching, 5 Hz by default) and **preview** (MJPEG
encode, 15 Hz on a downscaled copy) — plus a depth-1 V4L2 queue, so an
expensive recognition pass can never make the preview choppy or stale. The 🐞
debug panel shows all three rates; if the preview is choppy while capture is
at the device rate, the bottleneck is the encode, not the camera.

### Commander pod prototype

Open **`/play`** on the local table server. The default pod is a real Forge
`WebHumanController` at seat 0 and three stock Forge AI opponents. It uses the
submitted deck list as Forge's virtual deck. Fly-brain opponents are an optional
launcher setting, not a default runtime dependency. The browser mediates Forge
prompts; unsupported GUI/input paths fail closed.

This is the required **digital-controller development stage**, not the finished
physical-table product. Forge supplies the shuffle, draws, library, hand and
zones. No physical card instance is mapped to a Forge card ID, no physical
library order is synchronized, and camera/scanner observations never submit a
game action. **SCAN != PLAY.** The webcam is only a preview. The page states
that the displayed Forge hand is digital; this is not a synchronized paper-hand
workflow, keyboard-free physical game or Spelltable/Convoke substitute.

The Commander-list parser checks structure and size; Forge resolves card names
and enforces all game rules. The parser is not a rules engine. The eventual
physical layer must request actions from Forge and never edit Forge's zones,
stack, mana, life, combat or outcomes itself.

The new real-Forge integration harness,
`tests/test_forge_web_human_integration.py`, is intended to exercise land play,
spell cast/resolution and combat through the real WebHumanController and assert
Forge state changes. It has **not** been compiled or run in this environment.
The browser pod and full Commander completion are likewise not runtime
verified.

```bash
# If the pinned Forge checkout is sparse, materialize source/resources first.
git -C data/forge-source-checkout sparse-checkout disable
python3 scripts/build_forge_source.py
python3 -m pytest -q -s tests/test_forge_web_human_integration.py
.venv/bin/python scripts/physical_table.py --no-camera --port 8795
# Open http://localhost:8795/play
```

Use JDK 17+, Maven and the pinned Forge 2.0.15 resources. The pytest harness
skips without a real Forge JAR and `res/`; a skip is not a pass. This sandbox
has no Java/Javac/Maven and the source checkout is sparse, so no Java compile or
Forge game has been run here. See
[docs/paper-game-integration.md](docs/paper-game-integration.md) for the
required milestones and physical-synchronization constraints.

**Correction to the old mirror documentation:** direct `moveTo`, `setTapped`,
`addCounter`, and `setLife` calls are state edits, not legal casting/cost payment.
The agent rejects `/table/events` and `/table/state` with HTTP 410. Legacy Python
mirror code remains for compatibility, but cannot change this game.

The following scanner/observer material describes a standalone diagnostic
pipeline only. It is **not connected to the Forge game** and does not decide
rules, simulate the authoritative match, or submit actions. `PhysicalGameState`
and its event annotations must not be treated as a second rules engine or as
Forge state. The only rules authority for gameplay is Forge.

Legacy diagnostic architecture (all under [physical/](physical/)):

```
local camera/scan observer → candidate identity/location observations
   → optional diagnostic tracking and annotations (not Forge game state)
   → no action or state mutation reaches Forge
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

  Pipeline knobs (capture / analysis / preview run on three separate threads,
  so these tune *their* cadences and never throttle the camera):

  | Key | Default | Meaning |
  | --- | --- | --- |
  | `camera_buffer_size` | `1` | V4L2 queue depth. `1` = every read returns the **newest** frame; the OpenCV default (4+) hands a 5 Hz consumer frames the camera captured half a second ago, which is what a "laggy" preview really is |
  | `preview_width` | `960` | the MJPEG preview is downscaled (INTER_AREA) before encoding — a 1080p JPEG encode costs ~20 ms and buys nothing on a browser-sized `<img>` |
  | `preview_fps` | `15` | preview encode cadence (this is the stream's frame rate) |
  | `preview_quality` | `70` | JPEG quality of the preview stream |
  | `analysis_fps` | `5` | detection + rectification + matching cadence |
  | `analysis_width` | `0` | analyse a downscaled frame (`0` = native). The single biggest lever on a slow machine — try `1280` if recognition is the bottleneck |

  Env equivalents: `FLYCOMMANDER_CAMERA_DEVICE`, `FLYCOMMANDER_CAMERA_WIDTH`,
  `FLYCOMMANDER_CAMERA_HEIGHT`, `FLYCOMMANDER_CAMERA_FPS`,
  `FLYCOMMANDER_CAMERA_FORMAT`, `FLYCOMMANDER_CAMERA_BUFFER_SIZE`,
  `FLYCOMMANDER_CAMERA_PREVIEW_WIDTH`, `FLYCOMMANDER_CAMERA_PREVIEW_FPS`,
  `FLYCOMMANDER_CAMERA_PREVIEW_QUALITY`, `FLYCOMMANDER_CAMERA_ANALYSIS_FPS`,
  `FLYCOMMANDER_CAMERA_ANALYSIS_WIDTH`. Inspect the effective settings without
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
- **Legacy annotation UI (diagnostic only)** — it can display proposed tap
  observations or local annotations. None of these values submit actions to
  Forge or update a Forge match.
- **Legacy private-zone view** — the standalone observer can hide hand/library
  identities in its own display. Forge's actual game view and browser adapter
  are separate and must continue to use viewer-filtered data.

Verified working: Python observer/UI tests cover the diagnostic pipeline under
its test fixtures. They are not Forge gameplay or physical synchronization tests.
The previous report that the 173-test suite was green is historical, not
verification of the current Forge human controller.
