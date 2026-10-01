#!/usr/bin/env python3
"""FlyCommander — run one or more full Commander matches end to end.

Builds the mushroom body, starts the /decide brain server, launches the Java
agent (patched Forge), polls game results, and lets the dopaminergic system
learn from each game.

Deck specs (resolved on the Java side via Forge's own Commander pool):
    random        Forge picks from its normal Commander Decks pool
    <deck name>   a deck in Forge's Commander deck storage
    <path>.dck    an explicit deck file

Usage:
    python scripts/run_match.py --games 1 --fly-deck "Atraxa AI Deck" \
        --ai-deck random
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.connectome import build_synthetic_connectome          # noqa: E402
from brain.dopamine_plasticity import DopamineSystem             # noqa: E402
from brain.mushroom_body import MushroomBody                     # noqa: E402
from flycommander.deck_specs import classify_spec                # noqa: E402
from flycommander.forge_agent_client import (                    # noqa: E402
    AgentClient,
    FlyBrainServer,
)
from flycommander.reward_shaping import RewardComputer           # noqa: E402

FORGE_DIR = Path("/home/flynn/Downloads/mtg forge")
FORGE_JAR = FORGE_DIR / "forge-gui-desktop-2.0.15-jar-with-dependencies.jar"
PATCH_CLASSES = FORGE_DIR / "forge-agent-patch" / "classes"
DECK_DIR = Path.home() / ".forge" / "decks" / "commander"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
JAVA_LOG = LOG_DIR / "java.log"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the fly brain in Forge Commander")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--fly-deck", default="random",
                    help="'random', a Forge Commander deck name, or a .dck path")
    ap.add_argument("--ai-deck", action="append", default=None,
                    help="AI opponent deck spec (repeatable; default: random)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed Forge's random pool selection (reproducible)")
    ap.add_argument("--checkpoint", default=None, help="brain checkpoint to load")
    ap.add_argument("--save", default=None, help="where to save the brain after")
    ap.add_argument("--agent-port", type=int, default=8791)
    ap.add_argument("--brain-port", type=int, default=8792)
    ap.add_argument("--java-xmx", default="3g")
    ap.add_argument("--skip-java", action="store_true",
                    help="only start the brain server (Java already running)")
    return ap.parse_args()


def build_java_command(args: argparse.Namespace, fly_spec: str,
                       ai_specs: list[str]) -> list[str]:
    """Assemble the JVM command line; deck specs are forwarded verbatim."""
    cmd = [
        "java", f"-Xmx{args.java_xmx}",
        f"-Dfly.agent.brainUrl=http://127.0.0.1:{args.brain_port}/decide",
        f"-Dfly.agent.port={args.agent_port}",
    ]
    if getattr(args, "seed", None) is not None:
        cmd.append(f"-Dfly.agent.seed={args.seed}")
    cmd += [
        "-cp", f"{PATCH_CLASSES}:{FORGE_JAR}",
        "fly.agent.AgentMain",
        str(args.games), fly_spec, *ai_specs,
    ]
    return cmd


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"match_{int(time.time())}.jsonl"

    # --- deck specs (validate local paths only; Forge does the rest) -------
    try:
        fly = classify_spec(args.fly_deck, DECK_DIR)
        ai_specs = [classify_spec(s, DECK_DIR)
                    for s in (args.ai_deck or ["random"])]
    except ValueError as exc:
        print(f"[fly] ERROR: {exc}")
        return 2

    # --- brain ------------------------------------------------------------
    connectome = build_synthetic_connectome(n_pn=64)
    mb = MushroomBody(64, connectome=connectome)
    if args.checkpoint:
        mb.load_weights(args.checkpoint)
        print(f"[fly] loaded checkpoint {args.checkpoint}")

    dopamine = DopamineSystem(mb)
    reward = RewardComputer()

    server = FlyBrainServer(mb, dopamine, reward,
                            port=args.brain_port,
                            log_path=str(log_file))
    server.start()
    print(f"[fly] brain server on :{args.brain_port} (log: {log_file})")

    # --- java agent -------------------------------------------------------
    proc = None
    java_log = None
    if not args.skip_java:
        cmd = build_java_command(args, fly.java_arg, [s.java_arg for s in ai_specs])
        print("[fly] launching:", " ".join(cmd))
        JAVA_LOG.parent.mkdir(exist_ok=True)
        java_log = open(JAVA_LOG, "a", encoding="utf-8")
        java_log.write(f"\n===== launch {time.strftime('%F %T')} =====\n")
        java_log.flush()
        proc = subprocess.Popen(cmd, cwd=str(FORGE_DIR),
                                stdout=java_log,
                                stderr=subprocess.STDOUT)

    # --- wait & poll --------------------------------------------------------
    client = AgentClient(f"http://127.0.0.1:{args.agent_port}")
    if not args.skip_java:
        print("[fly] waiting for Forge agent to become ready (boot ~40-90s)…")
        if not client.wait_until_ready(timeout_s=420):
            print("[fly] agent did not become ready in time")
            if proc:
                proc.terminate()
            return 3

    # Forge-resolved deck names, straight from the agent (Deck objects)
    h = client.health() or {}
    if h.get("flyDeck") or h.get("aiDecks"):
        print(f"[fly] Forge resolved fly deck: {h.get('flyDeck')}")
        for i, name in enumerate(h.get("aiDecks", []), start=2):
            print(f"[fly] Forge resolved ai deck {i}: {name}")

    results = []
    try:
        games_seen = 0
        last_result_str = None
        while games_seen < args.games:
            h = client.health()
            if h is None:
                print("[fly] agent connection lost")
                break
            r = client.result()
            if r and r != last_result_str and r.get("game"):
                last_result_str = r
                games_seen = max(games_seen, int(r["game"]))
                summary = server.finish_episode(r)
                results.append(summary)
                emoji = "🏆" if summary["flyWon"] else "💀"
                print(f"[fly] {emoji} learned from game {r['game']}: "
                      f"won={summary['flyWon']} turns={summary['turns']} "
                      f"steps={summary['steps']} "
                      f"reward={summary['totalReward']:.3f}")
            if h.get("gamesCompleted", 0) >= args.games:
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[fly] interrupted")
    finally:
        if proc and proc.poll() is None:
            print("[fly] stopping Java agent…")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        if java_log:
            java_log.close()
        server.stop()

    if args.save:
        mb.save(args.save)
        print(f"[fly] brain saved → {args.save}")

    wins = sum(1 for s in results if s["flyWon"])
    print(f"[fly] session: {wins}/{len(results)} wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
