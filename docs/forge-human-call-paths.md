# Human integration call paths — before implementation

Inspected unmodified Forge 2.0.15, commit
`4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676`.

| Concern | Actual source path / method | Integration boundary |
|---|---|---|
| Seat creation | `forge.player.LobbyPlayerHuman.createIngamePlayer`, `GamePlayerUtil`, `RegisteredPlayer.forCommander` | Instantiate a real PlayerControllerHuman, not PlayerControllerAi |
| Controller | `forge.game.player.PlayerController`; `forge.player.PlayerControllerHuman` constructor, `setGui` | Human controller owns InputProxy/InputQueue; supply IGuiGame adapter |
| Priority | `PlayerControllerHuman.chooseSpellAbilityToPlay` → `InputPassPriority.showAndWait` | InputQueue waits; card selection and OK go through InputProxy, not direct zone edits |
| Casting | `PlayerControllerHuman.playChosenSpellAbility` → `forge.game.player.PlaySpellAbility.playSpellAbility` | Retain Forge casting, targets, costs, rollback |
| Payment | `forge.player.HumanCostDecision`, `PlaySpellAbility`, synchronized payment inputs | Must forward card/mana selection and cancellation; never call AiCostDecision for Flynn |
| Targeting | `InputSelectTargets`, `PlayerControllerHuman.chooseTargetsFor` | Forge validates selected entities; view IDs only on wire |
| Combat | `InputAttack`, `InputBlock`, controller declare methods | Forward selection and OK; Forge validates complete declarations |
| Modal/card choices | `IGuiGame.getChoices`, `one`, `many`, `getInteger`, `confirm` | Blocking typed requests; map opaque option IDs back to original objects |
| Triggers/replacements | PlayerControllerHuman ordering/replacement choice methods → IGuiGame ordering/choice methods | Preserve ordering; no default selection |
| Draw | `Player.drawCards` → private `doDraw` | After Draw replacement but before library card selection and moveTo; source hook required |
| Mulligan | `MulliganService.perform`, `LondonMulligan.mulliganDraw`, `tuckCardsDuringMulligan` | Opening/drawn hand identity and bottom ordering need physical acknowledgements |
| Search | `ability.effects.ChangeZoneEffect.resolve` + human card choices | Library selection must obey Forge filters and physical identity |
| Mill/shuffle | `Player.mill`, `Player.shuffle` | Ordered physical acknowledgements; unknown order must not become invented card identity |
| Fly | `PlayerControllerAi`, existing FlyPlayerController/BrainClient | Keep three separate endpoints and learner instances |

## Implementation scope

New DecisionBroker and WebHumanGui cover blocking generic choices only. They are
NOT installed as the production table controller yet: InputProxy bridging,
viewer-safe snapshots and physical library source hooks remain required.
Unsupported GUI methods throw instead of returning AI/default choices.
The old assisted launcher remains a diagnostic, not the requested paper game.

## Physical identity design (not implemented Forge zone hooks)

Maintain a per-seat ledger of physical instance IDs (copies distinguished),
known printing/oracle identity or unknown, observed zone, and binding to a Forge
card ID. Unknown observations cannot become placeholder Forge cards. Camera
observations append candidate facts only. A Forge-issued operation token binds
an acknowledgement to an expected source/destination and state revision.

Draw/reveal/mill must obtain missing identity *before* Forge consumes the card;
known top/bottom constraints must be checked. A user cannot substitute an
arbitrary library instance for a known top card. Search selection is constrained
by Forge's candidate set. A shuffle invalidates order knowledge only when Forge
actually shuffles. Mulligan returns must preserve physical count and bottom order.
Forge movement commits update ledger bindings only after successful resolution.
Cancellation/uncertain delivery leaves the operation pending for reconciliation.
No generic “sync battlefield” mutation is permitted.

These source hooks are deliberately not faked by changing Card names or moving
arbitrary library cards. They need a versioned upstream patch and runtime tests.

## Decision wire contract (new broker; HTTP wiring still pending)

Request: `{id, kind, message, options:[{id,label}], min, max}`.
Response: `{id, selected:[optionId,...]}`. Selection order is meaningful.
Request IDs are UUIDs scoped to a single pending human request. Reconnect reads
that same request; it must not generate a new one. Duplicate/stale responses,
unknown IDs, duplicate selections and invalid cardinality are rejected without
waking the game thread. Zero timeout means wait indefinitely. Timeout, interrupt
or explicit cancellation terminates the wait, never passes priority. A controller
failure must stop the match and publish a fault, not resume gameplay.

Broker validates transport bounds only; Forge input objects still validate Magic
legality. HTTP must publish immutable broker requests and never expose live Forge
objects. The current `/human/decision` assisted endpoint has a different schema;
DO NOT send this contract to it. Per-game/seat routing, authorization and broker
HTTP/UI integration are unfinished. WebHumanGui uses opaque option labels until
viewer-safe rendering is implemented; it intentionally does not serialize arbitrary
objects with toString(), which could disclose hidden card identities.

## Local build/test commands (Ubuntu/Debian)

```sh
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk maven
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version
javac -version
mvn --version
python3 scripts/build_forge_source.py --verify-only
python3 scripts/build_forge_source.py
# Standalone Java broker test; no Forge dependency needed:
mkdir -p data/broker-test
javac --release 17 -d data/broker-test \
  forge_patch/src/fly/agent/DecisionBroker.java \
  forge_patch/tests/fly/agent/DecisionBrokerTest.java
java -cp data/broker-test fly.agent.DecisionBrokerTest
# Run upstream tests separately (build script skips them):
(cd data/forge-source-checkout && mvn -B -pl forge-gui-desktop -am test)
```

There is deliberately NO claimed runtime command for the requested real physical
four-player acceptance test yet. WebHumanController is not registered by
AgentMain. The existing run_paper_game.py command launches the old assisted
prototype with Forge-generated hands and stock AI opponents; it cannot satisfy
this test. Wiring a new command before physical synchronization and InputProxy
support would misrepresent the implementation.

## Remaining implementation and verification

- Implement physical operation ledger and version-pinned library hooks; add
  identity/copy/order/mulligan/search/mill reconciliation tests.
- Bridge synchronized InputProxy selection/OK/cancel, mana, combat, numeric and
  ordering requests. Current generic choice adapter is only a partial path.
- Add viewer-filtered snapshots and HTTP/UI integration for the new broker.
- Register human lobby and three independent Fly lobbies; start FlyPod, route
  per-seat observations/outcomes and checkpoint lifecycle. FlyPod tests exist,
  but no live Forge → three-brain integration has run.
- Compile all Java against pinned Forge, run Java lifecycle tests (currently
  skipped without JDK), then perform every step of the real-game acceptance test.

## Subsequent wiring checkpoint

`WebHumanLobbyPlayer` now creates WebHumanController. AgentMain's separate
`-Dfly.agent.physical=true` path registers Flynn plus three FlyLobbyPlayers with
independent brain URLs (8792–8794). `scripts/run_physical_pod.py` starts/stops
three FlyPod learners and forwards deck specs through the existing resolver.
This is a **seat-wiring diagnostic**, not gameplay: before startGame it publishes
a blocked status and waits, preventing Forge from dealing an invented hand.

WebHumanSession exposes broker requests through GET /human/state and accepts
POST /human/decision `{id,selected:[...]}`. Python forwards these unchanged via
/api/game and /api/game/decision. The /play UI supports ordered multi-selection.
The identity ledger implements copy-safe bindings and revision-bound movement
acknowledgements, but is not yet attached to Forge library/draw operations.

Local diagnostic (after building with Java 17):

```sh
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
python3 scripts/build_forge_source.py
python3 scripts/run_physical_pod.py --human-deck /absolute/path/Flynn.dck \
  --fly-deck random --fly-deck 'Atraxa AI Deck' --fly-deck random
# Separate terminal:
.venv/bin/python scripts/physical_table.py --no-camera --port 8795
# Open http://localhost:8795/play
```

Expected diagnostic result: four seats created, then blocked before dealing.
This is not the real four-player gameplay acceptance command. Generic choices
are connected, but cannot yet occur in this gated match. Source library hooks,
synchronized human inputs, viewer-safe snapshots, numeric/payment/combat inputs,
and per-Fly terminal outcome learning wiring remain unfinished.

Fixed a previously introduced source error in AgentMain.resultJson's method
signature. Java compilation remains unverified; source inspection is not a
substitute. New ledger and broker Java tests require local javac and are skipped
in this sandbox. No PR opened.
