# forager

[Benchmarks](https://andnp.github.io/forager/dev/bench/)

## Getting started

This environment approximately follows the RLGlue interface specification. One minor modification is that the `step` function does not emit a termination signal---this environment does not terminate!
```python
from forager.Env import ForagerEnv, ForagerConfig
from forager.objects import Wall, Flower, Thorns

config = ForagerConfig(
    size=1_000, # equivalently: (1_000, 1_000)

    # tell the env what types of objects you expect to see
    object_types={
        'wall': Wall,
        'flower': Flower,
        'thorns': Thorns,
    },

    # 'objects' mode is a (ap_x, ap_y, #objects) binary tensor
    # 'colors' mode is a (ap_x, ap_y, 3) uint8 tensor
    observation_mode='objects',

    # controls how far the agent can see around itself
    # should always be odd---this way the agent is centered
    # need not be square
    aperture=(7, 5),
)

env = ForagerEnv(config)

# place some objects randomly
env.generate_objects(freq=0.1, name='flower')
env.generate_objects(freq=0.01, name='wall')
env.generate_objects(freq=0.2, name='thorns')

obs = env.start()

# valid actions are [0, 1, 2, 3]
obs, r = env.step(0)
```

## Custom Objects
Can specify custom objects by extending the `ForagerObject` class.
Objects have a few configurable settings:
* `blocking` prevents the agent from occupying the same space as the object (think: wall)
* `collectable` means the object is removed when the agent collides. By default, collected objects are respawned after some delay.
* `color` specifies a color for the object class. **Note:** currently can only specify one color per _class_.
* `location` allows controlling the specific location of an object. If unspecified, object is placed uniform randomly.

Objects can emit a reward when the agent collides with them by specifying the `reward` function:
```python
def reward(self, rng: np.random.Generator, clock: int):
    # objects are *not* notified at every clock cycle, but can
    # simulate the passage of time by tracking how long since last contact
    time_since_last_collision = clock - self.last_collision
    r = np.sin(time_since_last_collision)

    # objects are handed an rng owned by the global environment
    # this can be used in combination with `clock` for simulations
    # or to add noise to the reward signal
    eps = rng.normal(0, 1)
    return r + eps
```

Finally, objects can control how much time passes until they respawn:
```python
def regen_delay(self, rng: np.random.Generator, clock: int):
    # prevents the object from respawning, this object is permanently removed changing the env forever
    # use with caution!
    return None

# default implementation:
def regen_delay(self, rng: np.random.Generator, clock: int):
    # a random number of timesteps into the future before the object reappears
    # self.location can be modified here to control _where_ the object reappears
    # otherwise if self.location is None, then the object will be randomly placed
    return rng.integers(10, 100)
```

## Design philosophy
This implementation should be fast and memory efficient. To do so, many intermediate representations are cached in `numba` objects which are not immediately serializable. This comes at a slight additional cost when initially building the environment and whenever the environment is serialized/deserialized (e.g. during checkpointing) while saving a significant amount of time on every `env.step` call. This design also comes at a mild readability and flexibility cost in order to significantly speed up observation generating code.

## Known limitations
* `color` can only be set once for every class and cannot be changed on a per-instance basis
* Random placement of objects on a huge gridworld can be costly. Remember, `size=1_000_000` creates a 1M*1M gridworld. Even with only 1% density of objects, this means 1_000_000_000 (1B) object instances need to be created and stored. To alleviate this problem, objects are created **lazily** when the agent collides with them for the first time. However, still finding 1B locations and earmarking them for later object generation takes some time.
* Serialization/deserialization is not yet implemented. This is highest dev priority
* Visualization of the domain is not implemented.
