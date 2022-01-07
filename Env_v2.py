# https://garage.readthedocs.io/en/v2019.10.1/_modules/index.html
import gym
import gym.spaces
import numpy as np
from collections import deque

from gym.wrappers import ResizeObservation

"""Atari environment wrapper for gym.Env."""
class AtariEnv(gym.Wrapper):
    """Atari environment wrapper for gym.Env.

    This wrapper convert the observations returned from baselines wrapped
    environment, which is a LazyFrames object into numpy arrays.

    Args:
        env (gym.Env): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return np.asarray(obs), reward, done, info


    def reset(self, **kwargs):
        """gym.Env reset function."""
        return np.asarray(self.env.reset())

from gym.wrappers import TransformObservation

"""Episodic life wrapper for gym.Env."""
class EpisodicLife(gym.Wrapper):
    """Episodic life wrapper for gym.Env.

    This wrapper makes episode end when a life is lost, but only reset
    when all lives are lost.

    Args:
        env: The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        self._lives = 0
        self._was_real_done = True

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        self._was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self._lives and lives > 0:
            done = True
        self._lives = lives
        return obs, reward, done, info


    def reset(self, **kwargs):
        """
        gym.Env reset function.

        Reset only when lives are lost.
        """
        if self._was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step
            obs, _, _, _ = self.env.step(0)
        self._lives = self.env.unwrapped.ale.lives()
        return obs

"""Max and Skip wrapper for gym.Env."""
class MaxAndSkip(gym.Wrapper):
    """Max and skip wrapper for gym.Env.

    It returns only every `skip`-th frame. Action are repeated and rewards are
    sum for the skipped frames.

    It also takes element-wise maximum over the last two consecutive frames,
    which helps algorithm deal with the problem of how certain Atari games only
    render their sprites every other game frame.

    Args:
        env: The environment to be wrapped.
        skip: The environment only returns `skip`-th frame.

    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape,
                                    dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """
        gym.Env step.

        Repeat action, sum reward, and max over last two observations.
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


    def reset(self):
        """gym.Env reset."""
        return self.env.reset()

"""Fire reset wrapper for gym.Env."""
class FireReset(gym.Wrapper):
    """Fire reset wrapper for gym.Env.

    Take action "fire" on reset.

    Args:
        env (gym.Env): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', (
            'Only use fire reset wrapper for suitable environment!')
        assert len(env.unwrapped.get_action_meanings()) >= 3, (
            'Only use fire reset wrapper for suitable environment!')

    def step(self, action):
        """gym.Env step function."""
        return self.env.step(action)


    def reset(self, **kwargs):
        """gym.Env reset function."""
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            obs = self.env.reset(**kwargs)
        return obs

"""Noop wrapper for gym.Env."""
class Noop(gym.Wrapper):
    """Noop wrapper for gym.Env.

    It samples initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    Args:
        env (gym.Env): The environment to be wrapped.
        noop_max (int): Maximum number no-op to be performed on reset.
    """

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self._noop_max = noop_max
        self._noop_action = 0
        assert noop_max > 0, 'noop_max should be larger than 0!'
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP', (
            "No-op should be the 0-th action but it's not in {}!".format(env))

    def step(self, action):
        """gym.Env step function."""
        return self.env.step(action)


    def reset(self, **kwargs):
        """gym.Env reset function."""
        obs = self.env.reset(**kwargs)
        noops = np.random.randint(1, self._noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.step(self._noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

"""Stack frames wrapper for gym.Env."""
class StackFrames(gym.Wrapper):
    """gym.Env wrapper to stack multiple frames.

    Useful for training feed-forward agents on dynamic games.
    Only works with gym.spaces.Box environment with 2D single channel frames.

    Args:
        env: gym.Env to wrap.
        n_frames: number of frames to stack.

    Raises:
        ValueError: If observation space shape is not 2 or
        environment is not gym.spaces.Box.

    """

    def __init__(self, env, n_frames):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError('Stack frames only works with gym.spaces.Box '
                             'environment.')

        if len(env.observation_space.shape) != 2:
            raise ValueError(
                'Stack frames only works with 2D single channel images')

        super().__init__(env)

        self._n_frames = n_frames
        self._frames = deque(maxlen=n_frames)

        new_obs_space_shape = env.observation_space.shape + (n_frames, )
        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._observation_space = gym.spaces.Box(
            _low,
            _high,
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype)

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _stack_frames(self):
        return np.stack(self._frames, axis=2)

    def reset(self):
        """gym.Env reset function."""
        observation = self.env.reset()
        self._frames.clear()
        for i in range(self._n_frames):
            self._frames.append(observation)

        return self._stack_frames()


    def step(self, action):
        """gym.Env step function."""
        new_observation, reward, done, info = self.env.step(action)
        self._frames.append(new_observation)

        return self._stack_frames(), reward, done, info

from gym.wrappers import TransformReward

def create_env_v2(id, clip=True):
    env = gym.make(
        f'ALE/{id}-v5', 
        obs_type='grayscale',
        frameskip=1, 
        repeat_action_probability=0,
        full_action_space=False,
        render_mode=None
    )
    env = EpisodicLife(env)
    env = ResizeObservation(env, 84)
    env = TransformObservation(env, lambda obs: obs.reshape(84, 84).astype('float32'))
    env = MaxAndSkip(env)
    if env.unwrapped.get_action_meanings()[1] == 'FIRE':
        env = FireReset(env)
    print(env.reset().dtype)
    env = Noop(env)
    env = StackFrames(env, 4)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)).astype('float32'))
    if clip:
        env = TransformReward(env, lambda r: np.clip(r, -1, 1))
    return env