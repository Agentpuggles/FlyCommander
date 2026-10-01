# Forge human controller and physical-game boundary

This page tracks the required transition from a Forge-owned digital game to a
physically synchronized Commander table. Forge 2.0.15 at pinned commit
`4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676` is the only rules authority.

## Current development stage

The default four-seat browser pod now creates a real Forge human controller in
seat 0 and stock Forge AI controllers in seats 1–3. External Fly brains are an
optional advanced opponent mode (`--opponents fly`); the browser pod defaults to
`--opponents forge` and does not start Fly services.

`WebHumanController` derives choices from Forge and returns Forge abilities or
`null` for pass. `WebHumanGui` is a partial `IGuiGame` proxy that routes Forge's
input callbacks through `InputProxy`/`InputQueue`; Forge still validates and
executes the resulting choice. The browser transport carries opaque option IDs,
with bounded numeric/text fields and explicit multi-select actions. Unsupported
Forge GUI/input methods fail closed rather than being converted to a pass.

This is **not runtime-verified**: no Java compile or live Forge game has run in
this environment. A new real-Forge integration harness,
`tests/test_forge_web_human_integration.py`, compiles and runs a controlled game
that selects a land, casts/resolves a spell, and attacks through the actual
human-controller/input path. It asserts Forge zone transitions and combat life
damage. It is runnable only with JDK 17+, the built Forge 2.0.15 JAR, and the
matching `res/` tree.

## Honest product boundary

The game is presently digital Forge gameplay using the Forge-generated deck,
shuffle, opening hand, draws and library. The deck list is an input to Forge; it
does not bind physical card copies to Forge IDs. The webcam is a local preview,
and card scans identify cards only. **SCAN is not PLAY**; scanning, seeing or
moving a physical card never changes Forge state. The `/play` page now states
these limits rather than presenting its virtual hand as a synchronized paper
hand.

The intended finished experience remains a player placing a real Commander deck
on the table, with no keyboard-driven or fake digital hand, and completing a
four-player game while Forge enforces every rule. The current digital controller
is the prerequisite milestone before adding cameras; it is not that finished
experience and is not a Spelltable/Convoke substitute.

## Runtime wiring

- `scripts/run_physical_pod.py` uses `WebHumanLobbyPlayer` by default and three
  Forge AI opponents. `--opponents fly` is opt-in. `--assisted-human` selects
  the older `PhysicalTableController` path and is legacy/experimental.
- The browser-launched `/play` pod uses the digital WebHumanController and Forge
  AI by default. It does not start Fly brains.
- `TableSnapshot` is viewer-scoped for the human's Forge-generated digital hand
  and masks opponent private zones. It explicitly reports that physical
  synchronization is not installed.
- `DecisionBroker` retains one request ID and validates option IDs, bounds,
  typed values and multi-select actions. HTTP workers pass serialized answers;
  they never walk live Forge objects or mutate game state.

## Remaining work in required order

1. **Real Forge pod**: verify the default human + three Forge-AI Commander pod
   starts and runs multiple turns against the pinned runtime.
2. **Digital human controller**: compile and execute the new integration test;
   expand it for mana payment, targets, modes, alternate costs, abilities,
   priority responses, multiplayer choices and combat decisions. Keep every
   action inside Forge's normal controller/input/validation path.
3. **Physical identity**: map every physical deck instance to a Forge card ID,
   separately from any action. Forge must remain authoritative for which card
   is in each zone.
4. **Physical action submission**: add recognition and confirmation
   incrementally. Do not convert a scan, location change or vision guess into a
   game action; submit an explicit action through Forge and accept it only if
   Forge validates it.
5. **Complete Commander decisions**: cover replacement effects, library search,
   hidden information, multiplayer and Commander-specific prompts, and test
   complete games.

A version-pinned Forge hook will be needed for physical library operations.
Draw acknowledgement must happen only after Forge determines a draw occurs and
before that card is moved and draw triggers resolve. Replacement draws must not
consume a paper card. Opening hands, mulligans, known top/bottom order, searches,
reveals, mills, scry, shuffles and face-down cards need defined synchronization.
Unknown physical identity must remain pending information, not a generic card.

## Verification commands

```sh
python3 scripts/build_forge_source.py --verify-only
python3 scripts/build_forge_source.py
python3 -m pytest -q -s \
  tests/test_forge_api_contract.py \
  tests/test_forge_priority_pass.py \
  tests/test_forge_web_human_integration.py
```

The integration pytest test skips unless a JDK, a real built Forge JAR and
external `res/` are available. A skip is not evidence of compilation or game
execution. The local Forge checkout is sparse and Java/Javac/Maven are not on
PATH in this sandbox, so the adapter and new integration harness remain
uncompiled and untested here.
