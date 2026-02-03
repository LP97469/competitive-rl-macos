import gymnasium as gym
from competitive_rl.car_racing.car_racing_multi_players import CarRacing
from competitive_rl.utils.atari_wrappers import FrameStack, WrapPyTorch, MultipleFrameStack, \
    FlattenMultiAgentObservation
from gymnasium.envs.registration import register


def register_car_racing():
    try:
        register(
            id="cCarRacing-v0",
            entry_point=CarRacing,
            kwargs=dict(verbose=0),
            max_episode_steps=1000,
            reward_threshold=900,
            disable_env_checker=True
        )
        register(
            id="cCarRacingDouble-v0",
            entry_point=CarRacing,
            kwargs=dict(verbose=0, num_player=2),
            max_episode_steps=1000,
            reward_threshold=900,
            disable_env_checker=True
        )
        print("Register cCarRacing-v0, cCarRacingDouble-v0 environments.")
    except gym.error.Error:
        pass

    if not hasattr(gym.wrappers.TimeLimit, "seed"):
        def _seed(self, seed=None):
            if hasattr(self.env, "seed"):
                return self.env.seed(seed)
            self.env.reset(seed=seed)
            return [seed]

        gym.wrappers.TimeLimit.seed = _seed

    def _render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    gym.wrappers.TimeLimit.render = _render

    if hasattr(gym.wrappers, "OrderEnforcing"):
        def _order_render(self, *args, **kwargs):
            return self.env.render(*args, **kwargs)

        gym.wrappers.OrderEnforcing.render = _order_render

    if not hasattr(gym.wrappers.TimeLimit, "step") or gym.wrappers.TimeLimit.step.__name__ == "step":
        def _step(self, action):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            return obs, reward, done, info

        gym.wrappers.TimeLimit.step = _step


def make_car_racing(env_id, seed, rank, frame_stack=None, action_repeat=None):
    assert "CarRacing" in env_id

    def _thunk():
        env = gym.make(env_id, action_repeat=action_repeat)
        env.reset(seed=seed + rank)
        if frame_stack is not None:
            env = FrameStack(env, frame_stack)
        env = WrapPyTorch(env)
        return env

    return _thunk


def make_car_racing_double(seed, rank, frame_stack=None, action_repeat=None):
    def _thunk():
        env = gym.make("cCarRacingDouble-v0", action_repeat=action_repeat)
        env.reset(seed=seed + rank)
        if frame_stack is not None:
            env = MultipleFrameStack(env, frame_stack)
        env = FlattenMultiAgentObservation(env)
        env = WrapPyTorch(env)
        return env

    return _thunk
