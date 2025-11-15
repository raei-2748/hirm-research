from hirm.envs.synthetic.heston import HestonEnv
from hirm.episodes.generator import generate_episodes


def test_generate_episodes_reproducible():
    env = HestonEnv(horizon=5)
    episodes_a = generate_episodes(env, num_episodes=2, horizon=5, seed=7)
    env2 = HestonEnv(horizon=5)
    episodes_b = generate_episodes(env2, num_episodes=2, horizon=5, seed=7)
    assert len(episodes_a) == 2
    assert len(episodes_b) == 2
    for ep_a, ep_b in zip(episodes_a, episodes_b):
        assert ep_a.length == 5
        assert ep_b.length == 5
        assert ep_a.prices.shape == ep_b.prices.shape
        assert ep_a.metadata.keys() == ep_b.metadata.keys()
        for key in ep_a.metadata:
            assert list(ep_a.metadata[key]) == list(ep_b.metadata[key])
