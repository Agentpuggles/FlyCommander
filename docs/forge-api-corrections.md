# Forge 2.0.15 API corrections

Pin: tag object b18e110a32462958aeda4b7fbac48c06399a2c23;
source commit 4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676.
The user verified the desktop Maven reactor succeeds on Java 17. The patch
reported seven API errors. The corrections below are read from that source,
not another Forge version. No scanner changes or upstream modifications.

| Broken call | Pinned API / correction |
|---|---|
| Card.addCounter(type, n, true) | GameEntity.addCounter(CounterType,int,Player,GameEntityCounterTable), then GameEntityCounterTable.replaceCounterEffect(Game,SpellAbility). Add only positive quantities. |
| Negative additions | Check Card.canRemoveCounters, then Card.subtractCounter(CounterType,positiveCount,Player); retains RemoveCounter replacements and counter-removed triggers. Reject Integer.MIN_VALUE before negation. |
| Card.addDamage | Delegate complete batches through GameAction.dealDamage(boolean,CardDamageTable,CardDamageTable,GameEntityCounterTable,SpellAbility), as DamageDealEffect does. |
| CardFactory.getCard(String,...) | StaticData.instance().getCommonCards().getCard(name), variant database fallback, then CardFactory.getCard(IPaperCard,Player,Game). |
| CounterType.P1P1 / M1M1 | CounterEnumType.P1P1 / M1M1; other names still use CounterType.getType(String), including Forge custom/keyword counters. |
| Card.getCollectorNumberId | Card.getPaperCard() → IPaperCard.getCollectorNumber(); set from IPaperCard.getEdition(). Null checked; explicit printing never falls back to matching only the name. |

## Damage and legacy safety

A target/amount/deathtouch flag cannot reproduce source-dependent damage. Legacy
mark_damage is rejected rather than inventing a source/cause or duplicating
already-resolved damage. The batch helper is for Forge-prepared effect data only;
it is not wired to camera events. Source keys use Game.getChangeZoneLKIInfo in
Forge's DamageDealEffect; Forge handles prevention, replacements, damage history,
lifelink, deathtouch and damage counters. The resolution/priority loop must run
state-based actions at its normal time; no premature check is inserted into the
middle of an effect. Combat batches must remain simultaneous, not per-target calls.

Name-only token requests are rejected: creating a regular PaperCard by token name
is not token creation. Actual token effects must use Forge's token path. The
legacy /table/events and /table/state endpoints stay disabled. Counter null-cause
helpers remain legacy state edits, not legal player actions or full counter spells.

## Verification commands (Arch/CachyOS)

```sh
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH="$JAVA_HOME/bin:$PATH"
python3 scripts/build_forge_source.py
# Compiles every patch source against the real jar and runs signature checks:
.venv/bin/python -m pytest -q tests/test_forge_api_contract.py
# Use FORGE_TEST_JAR=/absolute/path/to/built.jar for a different artifact location.
```

Sandbox attempt: source pin verification passed; full build stopped before Maven
because java/javac/mvn are absent on PATH. No successful patch compilation is
claimed. Real-JAR test is skipped here, not passed. Replacement/prevention/counter
behavior requires real games; the signature smoke test is not gameplay validation.

After a successful local patch compilation, the existing runtime diagnostic is:

```sh
.venv/bin/python scripts/run_physical_pod.py --human-deck /absolute/path/Flynn.dck \
  --fly-deck random --fly-deck random --fly-deck random
```

It still deliberately blocks before dealing, pending physical library source
hooks. This change fixes API assumptions, not the unfinished human game loop.
