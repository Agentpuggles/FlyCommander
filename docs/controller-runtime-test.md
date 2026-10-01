# WebHumanController integration harness

The default browser pod now uses a real `WebHumanController` for seat 0 and
stock Forge AI for seats 1–3. `--controller-test` remains an optional
runtime/diagnostic mode; it is not a physical library or card synchronization
mode. External Fly AI opponents are opt-in, not a default dependency.

The new `forge_patch/tests/fly/agent/WebHumanGameIntegrationTest.java` is a
real-Forge controlled-state harness. A background test script submits actual
`DecisionBroker` option IDs to the human controller. It exercises Forge's
normal land-play path, observes a spell on the Forge stack before passing,
asserts the spell resolves to the battlefield, then declares an attacker through
Forge's `InputAttack`/`InputProxy` callback and asserts the defending player's
life changes through Forge combat damage. It does not directly mutate those
outcomes. The fixture setup is controlled, so this is not yet a full match or a
physical-game test.

## Run against the pinned runtime

Requirements: JDK 17+, the pinned Forge 2.0.15 source/resources and Maven.
From the repository root:

```bash
git -C data/forge-source-checkout sparse-checkout disable
python3 scripts/build_forge_source.py
python3 -m pytest -q -s tests/test_forge_web_human_integration.py
```

The pytest harness compiles all FlyCommander Java sources against the real
Forge JAR and runs the integration class with Forge's external `res/` directory.
It skips if the JDK, built JAR or resources are missing; a skip is not a pass.

The adjacent regressions are separate:

```bash
python3 -m pytest -q -s \
  tests/test_forge_api_contract.py \
  tests/test_forge_priority_pass.py \
  tests/test_java_decision_broker.py
```

For the live browser pod, build first, then launch the local game server and
open `/play`. The default launch uses a Forge digital deck/hand, the new human
controller, and three Forge AI opponents. The physical camera is a preview;
physical card identity, physical library order and camera action submission are
not installed. Do not treat the controller harness as evidence of a complete
Commander game or the requested keyboard-free physical experience.

## Local sandbox status

This sandbox has a sparse pinned Forge checkout and no `java`, `javac` or Maven
on PATH. Attempts to install the JDK/Maven and reach Maven Central failed due to
network/mirror errors. Consequently neither the new integration test nor the
updated adapter has been compiled or run here. The source and runnable harness
are in the repository, but runtime acceptance is still pending on a machine
with the pinned Forge build prerequisites.
