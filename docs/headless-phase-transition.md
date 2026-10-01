# Headless GUI view lookup / end-of-turn regression

Pinned source: forge-2.0.15 / 4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676.
The user confirmed real priority regression success and live controller progress
through priority windows to END_OF_TURN. That run then threw on the headless
IGuiGame.getGameView lookup. This update needs another local build/runtime run.

## Inspected call path

PhaseHandler.onPhaseBegin, CLEANUP branch, calls every controller's
autoPassCancel. PlayerControllerHuman.autoPassCancel calls mayAutoPass, which
calls YieldController.shouldAutoYield. That method obtains the GUI game view:

- For stack-empty yielding, it checks `gv != null && gv.peekStack() != null`.
- Marker evaluation is inside `autoPassUntilMarker != null && gv != null`.
- No GUI view is needed for the normal, no-yield headless controller path.

YieldController.declineSuggestion returns early for a null view;
isSuggestionDeclined returns false for a null view after scope checks.
PlayerControllerHuman.skipsPromptForStackOrPhase likewise returns false when
getGameView returns null. These inspected direct lookup sites do not dereference
null. Other desktop GUI/network paths may need actual views; this patch does not
claim to implement those paths or all of IGuiGame.

WebHumanGui now returns null specifically for getGameView. It does not modify
engine Game.getView, supply an AI response, or enable an auto-yield. The default
exception for unsupported GUI methods is unchanged.

## Regression

ForgePriorityPassTest retains the original priority/stack assertions, then enters
MAIN2 using the existing controlled-state setup pattern. Real mainLoopStep calls
and broker-confirmed passes must enter END_OF_TURN and CLEANUP. The CLEANUP
phase-begin executes the previously crashing lookup. With an empty test hand,
empty stack and no auto-yield, cleanup must finish without a human discard
request and advance to the next player's UNTAP, then UPKEEP; turn number and
priority owner are asserted. The test uses the actual Forge phase implementation,
not a copied transition algorithm. Test-only APINA preference is disabled to
isolate this null-view path from local preferences.

No scanner, browser UI, physical-hand logic or launcher changes.

## Local verification (Arch/CachyOS)

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
python3 scripts/build_forge_source.py
.venv/bin/python -m pytest -q -s tests/test_forge_priority_pass.py
```

On success, the extended regression prints:

```
PASS: headless getGameView=null; MAIN2 -> END_OF_TURN -> CLEANUP -> next player's UNTAP -> UPKEEP, with Forge phase-begin callbacks
```

This line is expected output, NOT observed sandbox output.

Rerun the live digital-hand controller test, separately from the regression:

```bash
mkdir -p logs
set -o pipefail
.venv/bin/python -u scripts/run_physical_pod.py --controller-test \
  --runtime-dir '/home/flynn/Projects/Magic Fly/data/forge-built-runtime' \
  --human-deck tests/fixtures/controller-test.dck \
  --fly-deck tests/fixtures/controller-test.dck \
  --fly-deck tests/fixtures/controller-test.dck \
  --fly-deck tests/fixtures/controller-test.dck \
  2>&1 | tee logs/controller-runtime-test.log
```

Leave the existing physical-table server running and answer real Forge requests
on /play. Confirm the end-step prompt passes into later turns; return any new
exception rather than treating an unsupported decision as a pass. A later
cleanup-discard or other unsupported input may still block a longer run.

Sandbox attempts: build pin verified but build stopped at missing java/javac/mvn;
real-Forge regression skipped; live launcher stopped at missing staged runtime.
No JVM launched here, so this new transition fix is not yet runtime-verified.
