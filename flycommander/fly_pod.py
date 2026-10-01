"""Three independent Mushroom Fly agents; no shared episode or learning weights.

This module provides agent lifecycles, not Magic legality or a game simulation.
Only Forge observations/results may drive decisions and learning.
"""
from pathlib import Path

from brain.mushroom_body import MushroomBody
from brain.dopamine_plasticity import DopamineSystem
from flycommander.forge_agent_client import FlyBrainServer
from flycommander.reward_shaping import RewardComputer
from flycommander.sensory_encoder import N_SENSORY


class FlyPod:
    def __init__(self, base_port=8792, checkpoint_dir=None, seed=2049):
        if not 1 <= base_port <= 65533:
            raise ValueError('Three consecutive valid brain ports are required')
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.agents = {}
        for number in range(1, 4):
            mb = MushroomBody(N_SENSORY, seed=seed + number)
            if self.checkpoint_dir:
                checkpoint = self.checkpoint_dir / f'fly-{number}.npz'
                if checkpoint.is_file():
                    mb.load_weights(str(checkpoint))
            self.agents[number] = FlyBrainServer(
                mb, DopamineSystem(mb), RewardComputer(),
                port=base_port + number - 1)

    def start(self):
        try:
            for agent in self.agents.values():
                agent.start()
        except Exception:
            self.stop()
            raise

    def stop(self):
        for agent in self.agents.values():
            agent.stop()

    def finish_game(self, seat_results):
        """Require a separate Forge outcome for each Fly; never reuse human win."""
        if set(seat_results) != set(self.agents):
            raise ValueError('Forge results required for all three Fly seats')
        for result in seat_results.values():
            if not isinstance(result, dict) or type(result.get('flyWon')) is not bool:
                raise ValueError('Each seat requires its own boolean flyWon result')
        summaries = {n: agent.finish_episode(seat_results[n])
                     for n, agent in self.agents.items()}
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for n, agent in self.agents.items():
                agent.mb.save(str(self.checkpoint_dir / f'fly-{n}.npz'))
        return summaries

    def brain_panel(self):
        """Deliberately exclude observations, episode records and card identities."""
        return [{'seat': n, 'name': f'Fly #{n}',
                 'decisions': a.decisions_served, 'episodes': a.episodes_done,
                 'temperature': a.mb.temperature}
                for n, a in self.agents.items()]
