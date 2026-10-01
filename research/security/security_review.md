# Security Review

Status: RESEARCH DOCUMENT (audit only; no code changed). Last verified: 2026-10-01.
Threat model: **single-user local research project**. Realistic risks only;
no enterprise hardening recommendations.

---

## 1. Attack surface inventory

| Surface | Component | Risk (realistic) | Current state |
|---|---|---|---|
| HTTP server (Forge side) | AgentServer (Java) | malformed JSON crashes episode loop; oversized payload memory | unvalidated schema; no size cap observed |
| HTTP server (Python side) | [physical/server.py](../../physical/server.py) | same as above + browser client sends input | binds localhost; no auth (acceptable locally) |
| HTTP client | BrainClient → Python | Python returns malformed action JSON → Java crash loop | no schema validation observed |
| Scryfall responses | [physical/scryfall_cache.py](../../physical/scryfall_cache.py) | malicious/unexpected JSON shapes cached and consumed | cache trusts response structure |
| Bulk data files | Scryfall bulk JSONL.gz | multi-GB parse memory spikes; poisoned mirror (if URL overridden) | official URL hardcoded — verify at download time |
| Camera | browser getUserMedia | permission prompt phishing is N/A locally; frames stay local | OK for threat model |
| Browser UI | physical server pages | XSS via card names from Scryfall (arbitrary text rendered) | card names are attacker-influenced-adjacent (card *names* are user/Oracle content; sanitize before HTML insertion) |
| Filesystem | caches, checkpoints, logs | path traversal if any user-controlled string becomes a path | identifiers derive from card IDs; review before letting remote input pick file paths |
| Process lifecycle | JVM + Python pair | orphaned processes on crash (start_new_session used correctly) | OK |
| Dependencies | requirements.txt pins | supply-chain risk = npm/pip-grade background risk | pins present; no known-banned licenses |

## 2. Realistic findings (ranked)

1. **JSON schema trust.** Both directions of both HTTP links parse without
   validating required fields/types. A malformed observation (e.g.,
   `players` missing → NPE in Java; `life` as string → crash in encoder)
   kills an episode or a server. Local-only threat model makes this a
   *robustness* bug more than a security bug — but the fix is the same
   (validate at boundary, fail loud, log, continue). Claims SEC-001.
2. **Unbounded request size.** No observed Content-Length cap; a runaway
   vision client could OOM the small server. Cheap fix when hardening.
3. **Card-name-derived HTML insertion** in any browser page: escape card
   names (they can contain `<>` legitimately? — card names don't, but
   flavor/printed text fields can contain quotes/apostrophes; escape
   anyway).
4. **Cache poisoning is theoretical** (local attacker already owns the
   machine); document, don't engineer.
5. **Secrets:** none found (no tokens/keys in repo). Scryfall needs no auth.
   FlyWire/neuprint auth tokens (if connectome work starts) must NOT enter
   the repo.

## 3. What NOT to do

- Don't add auth/TLS to localhost research servers — complexity without
  realistic threat reduction for this project.
- Don't whitelist user-supplied paths without normalization + prefix check
  *when that feature exists*; today it doesn't.

## 4. Validation experiment

E-SEC-01 (roadmap): fuzz both HTTP endpoints with malformed JSON corpus
(type flips, missing fields, 100 MB bodies, deep nesting) — expected
findings: unhandled exceptions, no graceful episode abort. Security value
modest; robustness value high.
