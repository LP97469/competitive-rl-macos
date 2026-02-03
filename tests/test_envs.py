import pytest

from competitive_rl.make_envs import make_envs


def test_make_envs_cartpole_random_rollout():
    envs = make_envs(env_id="CartPole-v0", num_envs=2, asynchronous=False)
    try:
        obs = envs.reset()
        assert obs.shape[0] == envs.num_envs

        for _ in range(5):
            actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
            obs, rewards, dones, infos = envs.step(actions)
            assert obs.shape[0] == envs.num_envs
            assert rewards.shape[0] == envs.num_envs
            assert dones.shape[0] == envs.num_envs
            assert len(infos) == envs.num_envs
    finally:
        envs.close()


def test_make_envs_pong_variants_create_and_step():
    pygame = pytest.importorskip("pygame")
    pygame.display.init()
    envs = make_envs(env_id="cPong-v0", num_envs=1, asynchronous=False)
    try:
        obs = envs.reset()
        assert obs.shape[0] == envs.num_envs

        actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
        obs, rewards, dones, infos = envs.step(actions)
        assert obs.shape[0] == envs.num_envs
        assert rewards.shape[0] == envs.num_envs
        assert dones.shape[0] == envs.num_envs
        assert len(infos) == envs.num_envs
    finally:
        envs.close()


def test_make_envs_tournament_and_double_create():
    pygame = pytest.importorskip("pygame")
    pygame.display.init()
    tournament_envs = make_envs(env_id="cPongTournament-v0", num_envs=1, asynchronous=False)
    double_envs = make_envs(env_id="cPongDouble-v0", num_envs=1, asynchronous=False)
    try:
        tournament_obs = tournament_envs.reset()
        double_obs = double_envs.reset()
        assert tournament_obs.shape[0] == tournament_envs.num_envs
        assert double_obs.shape[0] == double_envs.num_envs
    finally:
        tournament_envs.close()
        double_envs.close()


def test_make_envs_car_racing_create_and_step():
    pytest.importorskip("Box2D")
    pygame = pytest.importorskip("pygame")
    pygame.display.init()
    envs = make_envs(env_id="cCarRacing-v0", num_envs=1, asynchronous=False, frame_stack=1)
    try:
        obs = envs.reset()
        assert obs.shape[0] == envs.num_envs

        actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
        obs, rewards, dones, infos = envs.step(actions)
        assert obs.shape[0] == envs.num_envs
        assert rewards.shape[0] == envs.num_envs
        assert dones.shape[0] == envs.num_envs
        assert len(infos) == envs.num_envs
    finally:
        envs.close()
