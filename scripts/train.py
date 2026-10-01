#!/usr/bin/env python3
"""FlyCommander — training loop.

Runs repeated sessions of headless Commander games through the Java bridge,
letting the dopaminergic system update KC→MBON weights after every game.
Each session launches one JVM (boot ~40–90 s) and plays ``--games-per-vm``
games inside it, which amortizes Forge's startup cost.

The curriculum lives here too: early episodes use fewer AI opponents so the
fly learns land drops and combat before facing full pods.

Usage:
    python scripts/train.py --episodes 10 --games-per-vm 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.connectome import build_synthetic_connectome          # noqa: E402
from brain.dopamine_plasticity import DopamineSystem             # noqa: E402
from brain.mushroom_body import MushroomBody                     # noqa: E402
from flycommander.forge_agent_client import (                    # noqa: E402
    AgentClient,
    FlyBrainServer,
)
from flycommander.reward_shaping import RewardComputer           # noqa: E402
from flycommander.deck_specs import classify_spec                # noqa: E402
from scripts.run_match import (                                  # noqa: E402
    DECK_DIR,
    FORGE_DIR,
    FORGE_JAR,
    LOG_DIR,
    PATCH_CLASSES,
)

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the fly brain in Forge")
    ap.add_argument("--episodes", type=int, default=10,
                    help="total games to play")
    ap.add_argument("--games-per-vm", type=int, default=5,
                    help="games per JVM launch (boot is expensive)")
    ap.add_argument("--fly-deck", default="random",
                    help="'random', a Forge Commander deck name, or a .dck path")
    ap.add_argument("--ai-deck", action="append", default=None,
                    help="AI opponent deck spec (repeatable; default: random)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed Forge's random pool selection")
    ap.add_argument("--resume", default=None, help="checkpoint to load")
    ap.add_argument("--agent-port", type=int, default=8791)
    ap.add_argument("--brain-port", type=int, default=8792)
    ap.add_argument("--java-xmx", default="3g")
    return ap.parse_args()


def play_session(games: int, fly_spec: str, ai_specs: list[str],
                 mb: MushroomBody, ports: tuple[int, int],
                 java_xmx: str, log_file: Path,
                 seed: int | None = None) -> list[dict]:
    """One JVM session; returns per-episode summaries."""
    dopamine = DopamineSystem(mb)
    reward = RewardComputer()
    summaries: list[dict] = []

    server = FlyBrainServer(mb, dopamine, reward,
                            port=ports[1], log_path=str(log_file))
    server.start()

    cmd = [
        "java", f"-Xmx{java_xmx}",
        f"-Dfly.agent.brainUrl=http://127.0.0.1:{ports[1]}/decide",
        f"-Dfly.agent.port={ports[0]}",
    ]
    if seed is not None:
        cmd.append(f"-Dfly.agent.seed={seed}")
    cmd += [
        "-cp", f"{PATCH_CLASSES}:{FORGE_JAR}",
        "fly.agent.AgentMain",
        str(games), fly_spec, *ai_specs,
    ]
    proc = subprocess.Popen(cmd, cwd=str(FORGE_DIR),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    client = AgentClient(f"http://127.0.0.1:{ports[0]}")
    try:
        if not client.wait_until_ready(timeout_s=420):
            print("[train] agent never became ready")
            return summaries

        games_seen = 0
        last_result = None
        while games_seen < games and proc.poll() is None:
            h = client.health()
            if h is None:
                break
            r = client.result()
            if r and r != last_result and r.get("game"):
                last_result = r
                games_seen = max(games_seen, int(r["game"]))
                summaries.append(server.finish_episode(r))
            time.sleep(2.0)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        server.stop()
    return summaries


def main() -> int:
    args = parse_args()
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"train_{int(time.time())}.jsonl"

    connectome = build_synthetic_connectome(n_pn=64)
    mb = MushroomBody(64, connectome=connectome)
    if args.resume:
        mb.load_weights(args.resume)
        print(f"[train] resumed from {args.resume}")

    try:
        fly_spec = classify_spec(args.fly_deck, DECK_DIR).java_arg
        ai_specs = [classify_spec(s, DECK_DIR).java_arg
                    for s in (args.ai_deck or ["random"])]
    except ValueError as exc:
        print(f"[train] ERROR: {exc}")
        return 2

    results: list[dict] = []
    remaining = args.episodes
    t0 = time.time()
    while remaining > 0:
        n = min(args.games_per_vm, remaining)
        print(f"[train] JVM session: {n} game(s) "
              f"({args.episodes - remaining + 1}..{args.episodes - remaining + n}"
              f"/{args.episodes})")
        session = play_session(n, fly_spec, ai_specs, mb,
                               (args.agent_port, args.brain_port),
                               args.java_xmx, log_file, seed=args.seed)
        results.extend(session)
        remaining -= n

        wins = sum(1 for s in results if s["flyWon"])
        recent = results[-10:]
        print(f"[train] running: {wins}/{len(results)} wins "
              f"({wins / max(1, len(results)):.1%}) | recent-10: "
              f"{sum(1 for s in recent if s['flyWon'])}/{len(recent)} | "
              f"τ={mb.temperature:.3f}")

        ckpt = CKPT_DIR / f"fly_ep{len(results)}.npz"
        mb.save(str(ckpt))

    wins = sum(1 for s in results if s["flyWon"])
    turns = [s["turns"] for s in results if s.get("turns")]
    print(f"\n[train] DONE in {time.time() - t0:.0f}s: {wins}/{len(results)} "
          f"wins ({wins / max(1, len(results)):.1%}), "
          f"median game length {statistics.median(turns) if turns else '-'} turns")
    print(f"[train] final brain → {CKPT_DIR / f'fly_ep{len(results)}.npz'}")
    print(f"[train] logs → {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
