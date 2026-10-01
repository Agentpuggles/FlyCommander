# First controller/runtime handshake — not paper gameplay

The user verified the previous API-correction commit compiles with the exact
Forge 2.0.15 reactor, and its real-JAR contract test passes. This new controller
handshake change still needs local compilation and runtime testing.

The existing launcher now accepts `--controller-test` and `--runtime-dir`.
Without --controller-test, the physical-library gate remains. With it, Forge
may deal DIGITAL test hands and call the human controller's real callbacks:
chooseStartingPlayer, mulliganKeepHand, tuckCardsViaMulligan and
chooseSpellAbilityToPlay. Each waits on DecisionBroker. Priority currently offers
only explicit Pass; it does not assert there are no other legal actions.
No AI fallback, automatic pass, physical hand synchronization or casting claim.

`awaitNextInput` and `cancelAwaitNextInput` are waiting-indicator callbacks, not
choices, and do not block. Other unsupported human input paths still fail closed.
Match.startGame failures publish a fault and retain HTTP service for inspection.

## Arch/CachyOS commands

From the FlyCommander repository root:

```bash
git pull --ff-only origin arena/01a0f6d4-flycommander
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
python3 scripts/build_forge_source.py
mkdir -p logs
set -o pipefail
.venv/bin/python -u scripts/run_physical_pod.py \
  --controller-test \
  --runtime-dir '/home/flynn/Projects/Magic Fly/data/forge-built-runtime' \
  --human-deck tests/fixtures/controller-test.dck \
  --fly-deck tests/fixtures/controller-test.dck \
  --fly-deck tests/fixtures/controller-test.dck \
  --fly-deck tests/fixtures/controller-test.dck \
  2>&1 | tee logs/controller-runtime-test.log
```

The fixture is a real-card Commander test deck (Isamaru + 99 Plains), intentionally
limited to reduce unrelated choices. Each Fly remains a separate controller and
brain, even when using the same deck. Fresh test brains avoid modifying saved
paper-session checkpoints. Forge is still responsible for starting-player logic.

Separate terminal:

```bash
.venv/bin/python scripts/physical_table.py --no-camera --port 8795
```

Open http://localhost:8795/play. If offered, select the starting player and
Confirm selection; keep/mulligan the digital hand, then explicitly pass when
Forge asks for priority. A request stays pending until the HTTP response arrives.
The same-origin API is GET /api/game and POST /api/game/decision with
`{id,selected:[optionId]}`.

## Evidence to collect

Actual logs now include initialization returned, each created seat/controller,
per-Fly HTTP /stats health response, Calling Match.startGame, human callback,
broker pending UUID, accepted UUID, and response returned to Forge. A /stats
connection is NOT a Fly decision. Later brain /stats decisionsServed increments
are needed to show real observation/decision calls. No synthetic decision probe
is sent to the learners. A human callback is evidence Forge reached that stage;
Calling Match.startGame alone is not evidence of a started game.

Please return logs/controller-runtime-test.log and the /api/game response if it
stops. UI screenshots alone cannot establish that Forge consumed a response.

## Sandbox attempt

Used the command above (without tee) with the supplied /home/flynn runtime path.
Actual output:

```
usage: run_physical_pod.py [-h] [--controller-test]
                           [--runtime-dir RUNTIME_DIR] --human-deck HUMAN_DECK
                           --fly-deck FLY_DECK
run_physical_pod.py: error: Build first: python scripts/build_forge_source.py
```

That path is on the user's machine, not this sandbox. No JVM was launched, no
seats created, and no broker request observed here. This is a preflight failure,
not a real Forge boot attempt. Runtime acceptance is pending local execution.
