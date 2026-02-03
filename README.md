

<table border="0" width=1000px align="center" style="margin-bottom: 100px;">
        <tr>
        <td align="center">
            <b>Compeititive Pong</b>
      </td>
        <td align="center">
            <b>Compeititive Car-Racing</b>
      </td>
    </tr>
    <tr>
        <td align="center">
            <img align="center" width=300px  src="resources/repo-cover-large.gif" />
      </td>
        <td align="center" width=400px>
            <img align="center" width=350px  src="resources/repo-cover-racing.gif" />
      </td>
    </tr>
</table>


# Competitive RL Environments

In this repo, we provide two interesting competitive RL environments:

1. Competitive Pong (cPong): The environment extends the classic Atari Game Pong into a competitive environment, where both side can be trainable agents.
2. Competitive Car-Racing (cCarRacing): The environment allows multiple cars to race and compete in the same map.



## Installation

```bash
pip install git+https://github.com/cuhkrlcourse/competitive-rl.git
```

### Troubleshooting

**`UserWarning: pkg_resources is deprecated as an API`**

This warning is emitted by dependencies that still import `pkg_resources` from `setuptools`. To silence it, either upgrade the offending dependency when a fix is released or temporarily pin `setuptools<81`. You can also capture the stack trace to identify the import source. This repo does not import `pkg_resources` directly.

**`objc: Class SDLApplication is implemented in both ... pygame ... and ... cv2 ...` (macOS)**

This warning appears when both `pygame` and a GUI-enabled OpenCV build load their own copies of SDL2. Ensure you install `opencv-python-headless` (which avoids SDL2) instead of `opencv-python`, or remove the duplicate SDL2 library from your environment.

If you believe you already use `opencv-python-headless`, confirm which build is actually being imported:

```bash
python -m pip show opencv-python opencv-python-headless
python - <<'PY'
import cv2
print("cv2 file:", cv2.__file__)
print("cv2 version:", cv2.__version__)
PY
```

The `cv2 file` path should point into your virtual environment. If it points elsewhere (e.g., system or Homebrew), you are importing a different OpenCV build than expected.

If the SDL warning persists, rebuilding a headless wheel from source can remove bundled SDL2:

```bash
pip uninstall opencv-python opencv-python-headless
pip install --no-binary opencv-python-headless opencv-python-headless
```


## Usage

You can easily create the vectorized environment with this function:


```python
from competitive_rl import make_envs

envs = make_envs("CompetitivePongDouble-v0", num_envs=num_envs, asynchronous=True)
```

See docs in [make_envs.py](https://github.com/ucla-rlcourse/competitive-rl/blob/master/competitive_rl/make_envs.py) for more information.

Note that for Pong environment, since it is built based on Atari Pong game, we recommand following the standard pipeline to preprocess the observation. We should convert the image to grayscale, resize it and apply frame stacking. Please refer to [this function](https://github.com/ucla-rlcourse/competitive-rl/blob/6bf7d561f924f95e659a384d38f52d6642d20878/competitive_rl/utils/atari_wrappers.py#L349) and [our wrapper](https://github.com/ucla-rlcourse/competitive-rl/blob/6bf7d561f924f95e659a384d38f52d6642d20878/competitive_rl/utils/atari_wrappers.py#L40) for more information.

If you want to create a single Gym environment instance:

```python
import gymnasium as gym
import competitive_rl

competitive_rl.register_competitive_envs()

pong_single_env = gym.make("cPong-v0")
pong_double_env = gym.make("cPongDouble-v0")

racing_single_env = gym.make("cCarRacing-v0")
racing_double_env = gym.make("cCarRacingDouble-v0")
```

The observation spaces:

1. `cPong-v0`: `Box(210, 160, 3)`
2. `cPongDouble-v0`: `Tuple(Box(210, 160, 3), Box(210, 160, 3))`
3. `cCarRacing-v0`: `Box(96, 96, 1)`
4. `cCarRacingDouble-v0`: `Box(96, 96, 1)`

The action spaces:

1. `cPong-v0`: `Discrete(3)`
2. `cPongDouble-v0`: `Tuple(Discrete(3), Discrete(3))`
3. `cCarRacing-v0`: `Box(2,)`
4. `cCarRacingDouble-v0`: `Dict(0:Box(2,), 1:Box(2,))`




## Acknowledgement

This repo is contributed by many students and alumni from CUHK: Zhenghao Peng ([@pengzhenghao](https://github.com/pengzhenghao)), Edward Hui ([@Edwardhk](https://github.com/Edwardhk)), Yi Zhang ([@1155107756](https://github.com/1155107756)), Billy Ho ([@Poiutrew1004](https://github.com/Poiutrew1004)), Joe Lam ([@JoeLamKC](https://github.com/JoeLamKC))
