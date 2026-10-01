# MTG Rules Findings Relevant to FlyCommander

Primary source: **Magic: The Gathering Comprehensive Rules, effective
2025-06-06** (Wizards of the Coast; fetched 2026-10-01, local text
`/tmp/mtgrules.txt`), cross-checked against the official Commander format
page (magic.wizards.com/en/formats/commander, accessed 2026-10-01). Rule
numbers below cite the 2025-06-06 edition; re-verify after each set release
(recent sets added rules — battles created 704.5v–x, "start your engines"
704.5z, so SBA lists are *version-sensitive*).

## What the current fly state model cannot represent (gap analysis)

Verified against `GameObserver.java` + `sensory_encoder.py`: the fly's Forge
observation contains life, zone *counts*, creature/land *counts*, ≤24 board
card names + tapped bits, stack size, attacker/blocker counts, and canPlay
booleans. The following rules-relevant state is **invisible to the fly
today**, with the rules that make it matter:

| Missing state | Governing rules | Why it matters to decisions |
| --- | --- | --- |
| Power/toughness of any creature | 510.1a (damage = power); 704.5f/g (lethality) | "attack" is blind: attacking into a 5/5 with 1/1s is invisible |
| Counters (+1/+1, loyalty, shields…) | 613.1f/g; 704.5q | board "advantage" is count-based, not quality-based |
| Keywords (flying, deathtouch, trample…) | 702.x; 510.1c (deathtouch/trample assignment) | blocking/attack decisions ignore lethality & evasion |
| Mana available / mana pool | 107.3, 405 (mana abilities resolve immediately) | "play" delegates entirely to Forge; fly can't reason about curve |
| Card types beyond creature/land | 205 (types); artifacts/planeswalkers/enchantments lumped into board count | cannot value removal targets |
| Attacker→defender mapping | 506.3 (only player/planeswalker/battle can be attacked); 802 multiplayer attack restrictions | "who is being attacked" and threat assessment impossible |
| Planewalker loyalty & attackable planeswalkers | 506.3; 306 | fly sees neither |
| Phase within turn (except own main gating) | 500 turn structure; 117 priority | encoder receives `turn` number but ignores `phase` |
| Storm count / spells cast this turn | 702.50 | storm-relevant states indistinguishable |
| Whether creatures are summoning sick | 302.6 ("a creature can't attack unless it has haste… controlled continuously since turn began") | Forge AI handles it implicitly; state lacks it |
| Which commander is which / commander damage per commander | 903.10a (21+ combat damage by the *same* commander = SBA loss) | `commanderDamage` arrives as a **sum over commanders** in GameObserver (sums over opponents' command-zone cards) — cannot track the 21-damage rule per source |
| Granted/lost abilities mid-board | 613.1f layer 6 | layer-dependent states collapse |
| Auras/equipment attachments | 301/303; 704.5m | attachment graph invisible |
| Battles / prototypes / MDFC faces | 309; 712 | absent entirely |

## Verified rule facts the AI must eventually encode

1. **Layers (613.1a–g)**: copy → control → text → type → color →
   ability → P/T, with **7c/7d dependencies** and timestamp order (613.7)
   within layers. Correct reasoning requires storing: base characteristics,
   each active effect with its layer + timestamp + dependency info. A single
   "current P/T" number is insufficient for counter/replacement interactions
   (e.g., Strength-of-the-association style effects that set then boost).
2. **State-based actions (704.5a–z)** checked *whenever a player gets
   priority* (704.3), in order, repeating: 0 life (a), library-empty loses
   only when attempting to draw (b), 10 poison (c, **ten**, not twenty),
   tokens cease to exist outside battlefield (d), T≤0 creature dies (f),
   lethal-damage death (g), deathtouch death (h), planeswalker 0 loyalty
   (i), legendary duplicate (j), +1/+1 vs −1/−1 annihilation (q), Sagas
   (s), battles (v–x). Commander-specific: **903.10a 21+ combat damage from
   the same commander** is an SBA; **903.9a** commander-to-command-zone
   return is itself an SBA choice.
3. **Combat damage assignment (510.1a–c)**: attacker assigns among
   blockers (order chosen at 509.2), must assign lethal to each before the
   next (510.1c); deathtouch/trample alter "lethal"; unblocked creatures
   damage the *player/planeswalker/battle* attacked (506.3: only these can
   be attacked). Trample overassigns to defender (510.1c); double strike
   creates two damage steps (510/511).
4. **Commander (903)**: 40 starting life (103.4c); color identity includes
   back faces and alternative characteristics (903.4d/e); commander tax +2
   generic per prior cast **from the command zone only** (903.8d verified
   in the rules text: "...that cost reduction... applies only when cast
   from the command zone" — the tax increments regardless of zone only if
   cast from command zone); commander replacement to command zone applies
   from graveyard/exile as SBA (903.9a) and from hand/library as
   replacement (903.9b).
5. **Priority (117)**: players receive priority in APNAP order after most
   actions; the stack resolves top-down only when all players pass in
   succession. Fly's "hold" currently passes priority *implicitly* by
   returning an empty list — meaning it never holds up instants by choice;
   that decision is Forge AI's.
6. **Multiplayer turn order (802)**: turn order APNAP; attackers may only
   attack opponents within range of influence in some variants (802.1 FFA
   default: all opponents); a player eliminated by commander damage leaves
   the game, their spells/permanents remain but leave at next SBA cycle
   (800.4a — spells they control on the stack are exiled).
7. **Mulligan (103.4)**: London mulligan, free discard count = cards
   mulliganed, on the first mulligan in multiplayer only the first player
   skips their draw step (103.7.1). Forge AI decides mulligans for the fly
   today.

## Disagreements / version cautions

- **Duel Commander** (mtgdc.info) **removed the 21-commander-damage rule**;
  the multiplayer Commander RC rules keep it (903.10a). FlyCommander targets
  the RC rules (via Forge) — do not mix Duel-Commander rules into the model.
- **Poison**: 10 counters to lose (704.5c); "infect" deals poison in combat;
  some older texts misstate 20. Verified 10 in the 2025-06-06 rules.
- Wikipedia/MTG Wiki state "~2,500 Kenyon cells" for *insects generally*;
  the Drosophila adult number is ~2,000 (Aso 2014) — the fly KB must cite
  the specific-species number (see neuroscience/mb_facts.md).

## What FlyCommander should store for rule-level reasoning (target model)

- Objects with: identity (oracle_id), controller, owner, zone, tapped,
  counters (map), attachments (list), markers (map), damage marked
  (per-source for deathtouch), current characteristics *and* base
  characteristics, timestamp, effects applying (layer, dependency).
- Global: turn, phase/step, priority player, stack (spell + controller +
  targets), mana pools (per player), storm count, cards drawn this turn,
  commanders-cast-count per player, commander damage **per commander pair**.
- History: this-turn events (ETB order for timestamps), last-combat data.
