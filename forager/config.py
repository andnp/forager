import json
import forager._utils.config as cu

from dataclasses import dataclass
from typing import Callable, Dict

from forager.colors import Palette
from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Size
from forager.logger import logger
from forager.objects import ForagerObject

ObjectFactory = Callable[[], ForagerObject]

@dataclass
class ForagerConfig:
    size: int | Size
    object_types: Dict[str, ObjectFactory]

    seed: int = 0
    colors: Palette | None = None
    observation_mode: str = 'objects'
    aperture: int | Size = 3

def load_config(path: str) -> ForagerConfig:
    with open(path, 'r') as f:
        d = json.load(f)

    config = ForagerConfig(**d)
    return config

def sanity_check(config: ForagerConfig) -> ForagerConfig:
    # Fatal and unfixable issues
    _assert_valid_observation_mode(config)

    # Fixable issues
    config = _maybe_fix_aperture(config)
    config = _default_palette(config)

    return config

def _assert_valid_observation_mode(config: ForagerConfig) -> None:
    valid_modes = ['objects', 'colors', 'world']
    if config.observation_mode not in valid_modes:
        raise ForagerInvalidConfigException(f'Observation mode must be one of {valid_modes}')

def _maybe_fix_aperture(config: ForagerConfig) -> ForagerConfig:
    ap = cu.to_tuple(config.aperture)

    new_ap = (
        cu.nearest_odd(ap[0]),
        cu.nearest_odd(ap[1]),
    )
    if ap[0] % 2 == 0 or ap[1] % 2 == 0:
        logger.warning(f'Aperture sizes must be odd. Resizing from {ap} to {new_ap}')
        config.aperture = new_ap

    return config

def _default_palette(config: ForagerConfig) -> ForagerConfig:
    if config.colors is None:
        config.colors = Palette(len(config.object_types), config.seed)

    return config
