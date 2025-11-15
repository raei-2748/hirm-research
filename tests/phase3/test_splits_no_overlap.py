from hirm.state.splits import create_episode_splits


def test_splits_are_deterministic_and_disjoint() -> None:
    num_episodes = 25
    splits = create_episode_splits(num_episodes, seed=42)

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    combined = train | val | test
    assert combined == set(range(num_episodes))
