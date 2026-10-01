# Commander table integration status

## What exists

The `/play` page accepts a Commander list for the human and three AI seats. The
Python parser checks list structure and writes Forge `.dck` files; Forge resolves
card names and remains the sole rules engine. The default launcher creates a
real browser-controlled `WebHumanController` at seat 0 and three stock Forge AI
opponents. External Fly brains are optional, not a runtime dependency. The
separate `--assisted-human` flag still selects a legacy AI-assisted controller.

The digital human adapter publishes a Forge snapshot and brokers Forge prompts.
Forge-enumerated land, spell and ability choices return through Forge's normal
controller path. The GUI proxy is partial and must fail closed when it reaches
an unsupported Forge input. A real-Forge integration harness has been added to
exercise land play, spell cast/resolution and combat damage through
`WebHumanController`, with state-transition assertions. **It has not been
compiled or run in this environment.**

## What does not exist

The web pod currently runs a **digital Forge game from the submitted deck
list**. Forge owns its virtual shuffle, hand, library, draws and zones. The
player must use the browser's digital hand and decisions; this is not yet a
physical deck gameplay experience. It must not be presented as synchronized to
cards on the table.

Physical identity, paper library order, camera-based action recognition and
physical action submission are not integrated with Forge. A scan identifies a
card only: **SCAN != PLAY**. Scanning, table location changes and vision
predictions cannot move a Forge card, play a land, cast a spell or change any
game state. The camera UI is only a local preview. This is not yet the requested
no-keyboard/no-fake-hand experience or a Spelltable/Convoke substitute.

## Required development sequence

1. Build and run a real Forge pod with a human seat plus three Forge AI seats;
   verify multiple turns against the pinned Forge version.
2. Compile and test the digital human controller against real Forge, beginning
   with land play, spell cast/resolution and combat. Extend coverage to mana,
   targets, modes, costs, abilities, priority, multiplayer and Commander prompts.
   Forge must validate and execute every decision.
3. Add identity mapping between physical card instances and Forge card IDs.
4. Add physical action recognition/submission incrementally. An observation or
   scan remains separate from the proposed action; Forge validates the action.
5. Complete Commander decisions and interactions, then test full games.

No camera or vision work should precede a working digital Forge controller.
Do not create a second rules engine: zones, legal actions, payment, stack,
combat, triggers, Commander rules and outcomes remain Forge-owned.

## Physical synchronization requirements

A version-pinned Forge hook will be required for draws and other library
operations. A physical draw acknowledgement must happen only after Forge
confirms that a draw actually occurs and before Forge moves that identified
card and processes its triggers. Replacement draws must not consume a physical
card. The protocol must cover opening hands, every mulligan, top/bottom order,
search, reveal, mill, scry, shuffle and face-down information. It must never
manufacture a card or teleport a physical identity between Forge zones. Unknown
identity is pending information, not an interchangeable card.

## Runtime and test status

The sandbox's pinned Forge source checkout is sparse and Java/Javac/Maven are
unavailable. Earlier package-mirror and Maven Central attempts failed. No Java
sources have been compiled and no Forge runtime has been launched here. Python
parser/UI/lifecycle tests are infrastructure tests only; they do not prove
Forge gameplay.

On a machine with JDK 17+, Maven and the pinned Forge resources:

```sh
git -C data/forge-source-checkout sparse-checkout disable
python3 scripts/build_forge_source.py
python3 -m pytest -q -s \
  tests/test_forge_api_contract.py \
  tests/test_forge_priority_pass.py \
  tests/test_forge_web_human_integration.py
```

The new integration pytest compiles the FlyCommander Java sources and runs a
real Forge fixture when the built JAR and `res/` are present; otherwise it
skips. A skipped test is not a pass. Only after those checks should the browser
pod be exercised through multiple turns. Do not claim physical synchronization
until the identity/action protocol and full-game tests exist.
