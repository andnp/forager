import json
import forager._utils.config as cu

from dataclasses import dataclass
from typing import List

from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Size, ForagerObject
from forager.logger import logger

@dataclass
class ForagerConfig:
    size: int | Size
    objects: List[ForagerObject]

    observation_mode: str = 'objects'
    aperture: int | Size = 3

def load_config(path: str) -> ForagerConfig:
    with open(path, 'r') as f:
        d = json.load(f)

    config = ForagerConfig(**d)
    return config

def sanity_check(config: ForagerConfig) -> ForagerConfig:
    # Fatal and unfixable issues
    _assert_sane_object_frequencies(config)
    _assert_valid_observation_mode(config)
    _assert_objects_placeable(config)

    # Fixable issues
    config = _maybe_fix_aperture(config)

    return config

def _assert_sane_object_frequencies(config: ForagerConfig) -> None:
    freqs = [obj.freq for obj in config.objects if obj.freq is not None]

    if not all((f <= 1 for f in freqs)):
        raise ForagerInvalidConfigException('Object frequencies must each be less than 1')

    if sum(freqs) > 1:
        raise ForagerInvalidConfigException('Sum of object frequencies must be < 1')

def _assert_objects_placeable(config: ForagerConfig) -> None:
    for obj in config.objects:
        if obj.location is None and obj.freq is None:
            raise ForagerInvalidConfigException(f'Cannot determine how to place object: {obj.name}')

def _assert_valid_observation_mode(config: ForagerConfig) -> None:
    valid_modes = ['objects', 'colors']
    if config.observation_mode not in valid_modes:
        raise ForagerInvalidConfigException(f'Observation mode must be one of {valid_modes}')

def _maybe_fix_aperture(config: ForagerConfig) -> ForagerConfig:
    ap = cu.to_tuple(config.aperture)

    new_ap = (
        cu.nearest_odd(ap[0]),
        cu.nearest_odd(ap[1]),
    )
    if ap[0] % 2 == 0 or ap[1] % 2 == 0:
        logger.warn(f'Aperture sizes must be odd. Resizing from {ap} to {new_ap}')
        config.aperture = new_ap

    return config
