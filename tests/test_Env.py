import pickle
import numpy as np
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects import Flower, Wall, Thorns

def test_observation_shape():
    config = ForagerConfig(
            size=500,
            object_types={
                'wall': Wall,
                'flower': Flower,
                'thorns': Thorns,
            },

            observation_mode='objects',
            aperture=9,
            seed=2,
        )
    forager = ForagerEnv(config)
    obs = forager.get_observation_shape()
    assert obs == (9, 9, 3), f"Expected (9, 9, 3) but got {obs}"

def test_pickleable():
    env = ForagerEnv(config=ForagerConfig(size=1, object_types={}))
    data = pickle.dumps(env)
    env2 = pickle.loads(data) # noqa
    # TODO: add equality check once api settles
    # for now, this test still ensures that things _can_ be pickled, which with numba is important

def test_init():
    # can specify sizes with integers
    config = ForagerConfig(
        size=8,
        object_types={},
        aperture=3,
    )
    env = ForagerEnv(config)
    obs_shape = env.get_observation_shape()

    assert env._state == (4, 4)
    assert env._size == (8, 8)
    assert env._ap_size == (3, 3)
    assert obs_shape == (3, 3, 0)

    # can specify sizes as uneven tuples
    config = ForagerConfig(
        size=(10, 5),
        object_types={},
        aperture=(5, 1),
    )
    env = ForagerEnv(config)
    obs_shape = env.get_observation_shape()

    assert env._state == (5, 2)
    assert env._size == (10, 5)
    assert env._ap_size == (5, 1)
    assert obs_shape == (5, 1, 0)

    # can add objects
    config = ForagerConfig(
        size=10,
        object_types={
            'flower': Flower,
        },
    )
    env = ForagerEnv(config)
    env.generate_objects(0.1, 'flower')
    obs_shape = env.get_observation_shape()

    assert len(env._obj_store) == int(100 * 0.1)
    assert obs_shape == (3, 3, 1)

    # world observation mode, apeture should be None
    config = ForagerConfig(
        size=10,
        object_types={},
        observation_mode='world'
    )
    env = ForagerEnv(config)
    obs_shape = env.get_observation_shape()

    assert env._ap_size is None
    assert obs_shape == (10, 10, 1)

def test_basic_movement():
    config = ForagerConfig(
        size=7,
        object_types={
            'wall': Wall,
        }
    )
    env = ForagerEnv(config)
    env.add_object(Wall((3, 4)))
    env.add_object(Wall((3, 5)))

    assert env._state == (3, 3)

    # stays still when bumping into a wall
    _ = env.step(0)
    assert env._state == (3, 3)

    _ = env.step(1)
    assert env._state == (4, 3)

    _ = env.step(3)
    assert env._state == (3, 3)

    _ = env.step(2)
    assert env._state == (3, 2)

def test_vision():
    config = ForagerConfig(
        size=7,
        object_types={
            'wall': Wall,
        },
        aperture=3,
    )
    env = ForagerEnv(config)
    env.add_object(Wall((3, 4)))
    env.add_object(Wall((3, 5)))
    env.add_object(Wall((0, 2)))

    assert env._state == (3, 3)

    x, _ = env.step(0)
    expected = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (3, 3)
    assert np.allclose(x[:, :, 0], expected)

    env.step(1)
    x, _ = env.step(0)
    expected = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (4, 4)
    assert np.allclose(x[:, :, 0], expected)

def test_respawn():
    config = ForagerConfig(
        size=7,
        object_types={
            'flower': Flower,
        },
        aperture=3,
        seed=1,
    )
    env = ForagerEnv(config)
    env.add_object(Flower((3, 4)))
    assert len(env._to_respawn) == 0

    _, r = env.step(0)
    assert r == 1
    assert len(env._obj_store) == 0
    assert len(env._to_respawn) == 1
    assert env._clock == 1

    for _ in range(50):
        env.step(0)

    assert len(env._to_respawn) == 1
    assert env._clock == 51

    env.step(0)
    assert env._clock == 52
    assert len(env._obj_store) == 1
    assert len(env._to_respawn) == 0

def test_wrapping_dynamics():
    config = ForagerConfig(
        size=5,
        object_types={
        },
        aperture=3,
        seed=1
    )
    env = ForagerEnv(config)

    # go up
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(0)
    assert env._state == (2, 3)
    _ = env.step(0)
    assert env._state == (2, 4)
    _ = env.step(0)
    assert env._state == (2, 0)
    _ = env.step(0)
    assert env._state == (2, 1)
    _ = env.step(0)
    assert env._state == (2, 2)

    # go down
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(2)
    assert env._state == (2, 1)
    _ = env.step(2)
    assert env._state == (2, 0)
    _ = env.step(2)
    assert env._state == (2, 4)
    _ = env.step(2)
    assert env._state == (2, 3)
    _ = env.step(2)
    assert env._state == (2, 2)

    # go right
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(1)
    assert env._state == (3, 2)
    _ = env.step(1)
    assert env._state == (4, 2)
    _ = env.step(1)
    assert env._state == (0, 2)
    _ = env.step(1)
    assert env._state == (1, 2)
    _ = env.step(1)
    assert env._state == (2, 2)

    # go left
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(3)
    assert env._state == (1, 2)
    _ = env.step(3)
    assert env._state == (0, 2)
    _ = env.step(3)
    assert env._state == (4, 2)
    _ = env.step(3)
    assert env._state == (3, 2)
    _ = env.step(3)
    assert env._state == (2, 2)

def test_wrapping_vision():
    config = ForagerConfig(
        size=5,
        object_types={
            'flower': Flower,
        },
        aperture=3,
        seed=1
    )
    env = ForagerEnv(config)
    env.add_object(Flower((0, 0)))

    expected = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    x = env.start()

    assert env._state == (2, 2)
    assert np.allclose(x[:, :, 0], expected)

    # go left
    _ = env.step(3)
    # go down
    x, _ = env.step(2)

    expected = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (1, 1)
    assert np.allclose(x[:, :, 0], expected)

    # go left , go left
    _ = env.step(3)
    x, _ = env.step(3)
    expected = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ], dtype=np.bool_)

    assert env._state == (4, 1)
    assert np.allclose(x[:, :, 0], expected)

def test_object_location():
    config = ForagerConfig(
        size=5,
        object_types={
            'flower': Flower,
        },
        aperture=3
    )
    env = ForagerEnv(config)
    f = Flower((2, 0))
    assert f.target_location == (2, 0)
    env.add_object(f)

    env.start()
    _, r = env.step(2)
    _, r = env.step(2)
    assert r == 1
    assert f.current_location == (2, 0)


def test_object_locations():
    config = ForagerConfig(
        size=(16, 8),
        object_types={
            "truffle": Truffle,
            "oyster": Oyster,
        },
        observation_mode="world",
    )
    env = ForagerEnv(config)
    size = config.size
    truffle_locations = np.zeros(size)
    truffle_locations[2:6, 2:6] = 1
    truffle_locations = np.ravel_multi_index(np.where(truffle_locations), size, order="F")

    oyster_locations = np.zeros(size)
    oyster_locations[10:14, 2:6] = 1
    oyster_locations = np.ravel_multi_index(np.where(oyster_locations), size, order="F")

    env.generate_objects_locations(2.0, "truffle", truffle_locations)
    env.generate_objects_locations(2.0, "oyster", oyster_locations)

    obs = env.start()
    assert obs.shape == (8, 16, 3)
    obs = obs.transpose(1, 0, 2)
    assert obs[8, 4, 0]
    assert obs[:, :, 0].sum() == 1
    assert obs[2:6, 2:6, 2].all()
    assert obs[:, :, 2].sum() == 16
    assert obs[10:14, 2:6, 1].all()
    assert obs[:, :, 1].sum() == 16
    assert obs.sum() == 33

def test_render():
    config = ForagerConfig(
        size=5,
        object_types={
            "flower": Flower,
        },
        aperture=3,
        seed=1,
    )
    env = ForagerEnv(config)
    env.add_object(Flower((0, 0)))
    env.start()
    rgb_array = env.render()
    assert rgb_array.shape == (5, 5, 3) and rgb_array.dtype == np.uint8
    np.testing.assert_array_equal(
        rgb_array,
        [
            [[0, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
            [[255, 255, 255], [204, 204, 255], [204, 204, 255], [204, 204, 255], [255, 255, 255]],
            [[255, 255, 255], [204, 204, 255], [0, 0, 255], [204, 204, 255], [255, 255, 255]],
            [[255, 255, 255], [204, 204, 255], [204, 204, 255], [204, 204, 255], [255, 255, 255]],
            [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
        ],
    )


def test_checkpointing():
    config = ForagerConfig(
        size=5,
        object_types={
            'flower': Flower,
            'thorns': Thorns,
        },
        aperture=3
    )
    env = ForagerEnv(config)
    env.generate_objects(0.1, 'flower')
    env.generate_objects(0.2, 'thorns')

    env.start()
    for _ in range(5):
        env.step(0)

    raw = pickle.dumps(env)
    env2 = pickle.loads(raw)

    obs, rew = env.step(1)
    obs2, rew2 = env2.step(1)

    assert np.array_equal(obs, obs2) and rew == rew2

    obs, rew = env.step(2)
    obs2, rew2 = env2.step(2)

    assert np.array_equal(obs, obs2) and rew == rew2

# ----------------------------
# -- Performance benchmarks --
# ----------------------------
def test_benchmark_vision(benchmark):
    config = ForagerConfig(
        size=7,
        object_types={
            'wall': Wall,
        },
        aperture=3,
    )
    env = ForagerEnv(config)
    env.add_object(Wall((3, 4)))
    env.add_object(Wall((3, 5)))
    env.add_object(Wall((0, 2)))

    assert env._state == (3, 3)

    x, _ = benchmark(env.step, 0)
    expected = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (3, 3)
    assert np.allclose(x[:, :, 0], expected)

def test_benchmark_creation(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=31,
    )

    def _build(config):
        env = ForagerEnv(config)
        env.generate_objects(0.05, 'wall')
        env.generate_objects(0.05, 'flower')

    benchmark(_build, config)

def test_benchmark_small_env(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=15,
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)

def test_benchmark_big_env(benchmark):
    config = ForagerConfig(
        size=10_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=61,
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)

def test_benchmark_small_env_color(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=15,
        observation_mode='colors',
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)

def test_benchmark_small_env_world(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        observation_mode='world',
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)
