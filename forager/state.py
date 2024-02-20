import numpy as np
import numba as nb
import forager._utils.numba as nbu
import forager._utils.config as cu

from collections import defaultdict
from typing import Dict, List

from forager.config import ForagerConfig
from forager.interface import Size
from forager.objects import ForagerObject
from forager.ObjectStorage import ObjectStorage

class ForagerState:
    def __init__(
        self,
        config: ForagerConfig,
    ):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # parse config
        self.size: Size = cu.to_tuple(config.size)
        self.ap_size = cu.to_tuple(config.aperture) if config.aperture is not None else None

        # build object storage
        self.colors = cu.not_none(config.colors)
        self.objects = ObjectStorage(self.size, config.object_types, self.colors, self.rng)

        # build respawn timers
        self.to_respawn: Dict[int, List[ForagerObject]] = defaultdict(list)

        # ensure object types have a consistent object dimension
        _names = set(config.object_types.keys())

        if config.observation_mode == 'world':
            _names |= {'agent'}

        self.names = nbu.List(sorted(_names))
        self.names_to_dims = nbu.Dict.empty(
            key_type=nb.typeof(''),
            value_type=nb.typeof(int(1)),
            n_keys=len(_names),
        )

        for i, n in enumerate(self.names):
            self.names_to_dims[n] = i

        # build state information
        self.clock = 0
        self.agent_state = (
            int(self.size[0] // 2),
            int(self.size[1] // 2),
        )
