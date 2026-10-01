# Real paper Commander integration — implementation status

The AI-assisted controller is a diagnostic prototype, not the requested game.
Do not use its Forge-generated hand as the physical-hand design.

## Verified upstream integration points (Forge tag forge-2.0.15)

- `forge.player.PlayerControllerHuman` uses `IGuiGame`, synchronized inputs and
  `forge.game.player.PlaySpellAbility`. Reuse these paths rather than inheriting
  `PlayerControllerAi` for Flynn. The web adapter must implement input selection,
  confirmation, ordering and payment interfaces, not return default answers.
- `RemoteClientGuiGame`, `NetworkGuiGame`, and the upstream test
  `HeadlessNetworkGuiGame` illustrate network GUI adaptation. Test no-op/default
  decisions must NOT become production human choices.
- `Player.doDraw` runs DrawCards/Draw replacement effects before selecting and
  moving the library card; it then records draw history and triggers. It is
  private and its callers are final: replacing only the lobby/controller cannot
  override physical draws. A version-pinned source hook is necessary.

## Physical-zone requirements for that hook

A paper draw acknowledgement must happen after Forge determines a draw actually
occurs, before Forge moves its identified card and processes draw triggers. It
must map to a remaining physical deck instance, never manufacture a card or
teleport it from hand to battlefield. Known top/bottom order must be enforced.
Replacement draws must not consume paper draws. Opening hands and each mulligan
must use the same physical protocol. Library reveal, mill, search, scry, shuffle,
and face-down cards also need explicit synchronization; a draw-only hook is not
sufficient. Unknown identity is pending information, not a generic Forge card.

Public snapshots must use viewer-filtered card views; ad-hoc `getName()` loops
are not sufficient for face-down exile and visibility-changing effects. Stop on
unimplemented required decisions rather than delegating Flynn to the AI.

## This iteration

- Added FlyPod with three separate existing MushroomBody/DopamineSystem/
  RewardComputer instances, independent episode records, random seeds and
  checkpoint files. Brain panel excludes observations/card identities.
- Added explicit per-controller brain URL support in Java (legacy default kept).
- Fixed observer capability nesting to match the existing brain encoder.
- These components are not yet wired into a playable four-player human game.
- No scanner changes.

## Runtime blocker / acceptance gate

No Forge distribution or full JDK is available here. GitHub source API access
works; release binary download and Maven TLS connections fail. jdk4py provided
only a runtime, without javac. Java patch compilation has NOT been verified.

Next: provide/install Forge 2.0.15 + full compatible JDK, compile untouched
upstream and patch, implement the source hook plus human web input adapter, then
wire FlyPod and per-seat outcomes into match lifecycle. Preserve upstream license
and attribution for copied/modified code.

Acceptance requires the user's real-runtime sequence: four seats, actual paper
opening hand, land, spell/payment, Fly turn, human response, stack/priority,
combat, and real win/loss. Python tests are component checks only.

## Pinned source checkout (subsequent update)

Cloned `https://github.com/Card-Forge/forge.git` into ignored
`data/forge-source-checkout`, detached at `forge-2.0.15`.
Verified annotated tag object `b18e110a32462958aeda4b7fbac48c06399a2c23`
and peeled commit `4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676`.
The unmodified checkout includes `forge-gui/res` (approximately 464 MB).
The earlier missing-source/resource blocker is resolved.

Reproducible build entry point:

```sh
python scripts/build_forge_source.py --verify-only
python scripts/build_forge_source.py
```

The script checks tag, commit, tracked-source cleanliness and external resources;
then invokes the desktop Maven reactor with dependencies, stages its runtime and
resource link, and compiles FlyCommander using `javac --release 17`. Upstream
Java tests are skipped by this build command and remain a separate gate. This
build sequence is not yet validated beyond prerequisite checks.

Actual attempt stopped before Maven: no full JDK or Maven installed/on PATH.
The existing jdk4py runtime has no javac. Maven Central, Apache archive and
OpenJDK binary-host probes still fail with TLS connection errors. Git clone
works but does not supply the compiler or third-party Maven artifacts.
No upstream modules or agent Java sources have been compiled; Forge has not run.
