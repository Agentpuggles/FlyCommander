import numpy as np
import pytest
from flycommander.fly_pod import FlyPod


def test_separate_brains_and_learning_state():
    pod = FlyPod()
    a, b, c = pod.agents.values()
    assert len({id(x.mb) for x in (a,b,c)}) == 3
    assert len({id(x.dopamine) for x in (a,b,c)}) == 3
    assert len({id(x.reward) for x in (a,b,c)}) == 3
    assert len({id(x._episode) for x in (a,b,c)}) == 3
    assert not np.shares_memory(a.mb.connectome.kc_to_mbon, b.mb.connectome.kc_to_mbon)
    assert not np.array_equal(a.mb.w_mbon_action, b.mb.w_mbon_action)
    a.mb.connectome.kc_to_mbon[:] = 0
    assert np.any(b.mb.connectome.kc_to_mbon)
    assert [x.port for x in (a,b,c)] == [8792,8793,8794]


def test_private_observation_not_in_panel():
    pod = FlyPod()
    pod.agents[1]._pending_obs = {'privateHand': ['Secret card']}
    panel = pod.brain_panel()
    assert 'Secret card' not in str(panel)
    assert set(panel[0]) == {'seat','name','decisions','episodes','temperature'}


def test_separate_results_and_checkpoint_roundtrip(tmp_path):
    pod = FlyPod(checkpoint_dir=tmp_path)
    with pytest.raises(ValueError):
        pod.finish_game({1: {'flyWon': True}})
    results = {n: {'flyWon': n == 2, 'turns': 10} for n in (1,2,3)}
    summaries = pod.finish_game(results)
    assert [summaries[n]['flyWon'] for n in (1,2,3)] == [False,True,False]
    restored = FlyPod(checkpoint_dir=tmp_path)
    for n in (1,2,3):
        assert (tmp_path / f'fly-{n}.npz').is_file()
        np.testing.assert_array_equal(restored.agents[n].mb.w_mbon_action, pod.agents[n].mb.w_mbon_action)
