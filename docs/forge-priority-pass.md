# Priority pass regression: null is not an empty list

Pinned Forge: tag `forge-2.0.15`, tag object
`b18e110a32462958aeda4b7fbac48c06399a2c23`, source commit
`4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676`.

## Evidence and root cause

The user's real runtime log established that a broker response reached the Java
controller, followed immediately by another priority request. It also showed
`PhaseHandler: AI looped too much with: []`. These are two manifestations of the
same incorrect return contract, in the human and Fly controllers respectively.

Inspected source (line numbers refer to the unmodified pinned checkout):

- `forge-game/.../player/PlayerController.java:278`: abstract method; does not
  document the sentinel, so the signature alone is insufficient.
- `forge-gui/.../player/PlayerControllerHuman.java:1651–1719`: explicit auto-yield
  returns null. Interactive input waits, then returns `defaultInput.getChosenSa()`.
- `forge-gui/.../match/input/InputPassPriority.java:59,269–278,304–329`: chosenSa
  starts null. A confirmed pass stops input without selecting an ability; the
  getter returns null. The desktop path additionally handles macro recording and
  an optional floating-mana warning. This change reproduces the engine return
  semantics only; it does not claim that the custom UI implements all desktop UX.
- `forge-ai/.../PlayerControllerAi.java:832–834`: delegates to AiController.
- `forge-ai/.../AiController.java:1344–1398`: `singleSpellAbilityList(null)` returns
  null, not `[]`, when the AI has no selected play.
- `forge-game/.../phase/PhaseHandler.java:1039–1159`: `mainLoopStep` loops on the
  current controller. At 1066: `if (chosenSa == null) { break; // that means 'I pass' }`.
  An empty non-null list executes zero abilities, increments loopCount, and retries
  the *same* seat. AI seats eventually hit the 999 iteration warning; a human has
  no such iteration cap. After the null break, Forge gets the next player and
  either hands them priority, resolves the stack, or ends/advances the phase when
  the pass cycle closes.

## Correction

- WebHumanController returns null only after the broker accepts the explicit
  Pass response. Cancellation/unsupported inputs are not converted to passes.
- FlyPlayerController returns null on all pass/hold/no-play branches, including
  its off-main-phase gate and decision limit. Selected ability lists are unchanged.
- The legacy PhysicalTableController's pass branch uses the same sentinel.
- No Python/UI special case, manual priority mutation, or upstream Forge change.
- Controller-test remains digital-hand-only. Paper synchronization is unchanged.

## Real-engine regression

`ForgePriorityPassTest` creates a controlled four-seat Commander game with the
real WebHumanController and three instrumented FlyPlayerControllers. Setup uses
Forge dev-state methods in the same pattern as upstream `forge.ai.AITest`.

The test places the real Fountain of Youth GainLife ability on the stack as a
fixture (NOT a legal activation/payment test) and calls actual mainLoopStep:

1. Flynn blocks on a real broker priority request. One accepted pass must finish
   that loop step, clear the pending request, and hand priority to the next seat.
2. Each Fly must pass in exactly one controller call. These off-main passes need
   no brain service; they do not test brain connectivity or strategy.
3. Only the final pass may resolve the ability; life increases once, stack clears,
   and the active player receives priority in the same phase.
4. A new human pass and three Fly passes on the empty stack advance the phase.
5. A subsequent human request is legitimate, uses a new ID, and can also pass.
6. Stale/duplicate broker replies are rejected.

The old human code fails by timing out in mainLoopStep after issuing another
request. The old Fly code fails the repeated-controller-call assertion instead
of waiting for its 999-iteration safeguard. This test bypasses HTTP/browser and
opening-hand setup; those are covered by the user's separate runtime handshake.
It is not a test of an entire game or win/loss.

## Run locally

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
python3 scripts/build_forge_source.py
.venv/bin/python -m pytest -q -s tests/test_forge_priority_pass.py
```

The test compiles all patch sources and its Java regression against the real
JAR, then runs with the external `res/` directory. To use another staged location:

```bash
FORGE_TEST_JAR='/home/flynn/Projects/Magic Fly/data/forge-built-runtime/forge-gui-desktop-2.0.15-jar-with-dependencies.jar' \
FORGE_TEST_RUNTIME='/home/flynn/Projects/Magic Fly/data/forge-built-runtime' \
.venv/bin/python -m pytest -q -s tests/test_forge_priority_pass.py
```

Then rerun the existing `run_physical_pod.py --controller-test` command documented
in `controller-runtime-test.md`. Multiple human prompts remain normal when the
stack changes or a new phase begins; the bug was repeated prompts *inside the
same action loop* without a real pass. Do not suppress legitimate prompts in UI.

Sandbox result: test skipped because a JDK and built Forge runtime are not present.
No claim that this regression compiled or passed locally in the sandbox. The
previous runtime round trip is user-confirmed evidence; phase advancement after
this fix is pending a local run.
