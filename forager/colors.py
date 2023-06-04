import colorsys
import numpy as np
import forager._utils.numba as nbu

from typing import Any, Dict
from forager.interface import RawRGB

class Palette:
    def __init__(self, classes: int, seed: int = 0) -> None:
        self.classes = classes

        # stateful
        self._rng = np.random.default_rng(seed)
        self._colors: Dict[str, np.ndarray] = {}
        self._last_salt = self._rng.random()

    def register(self, label: str, color: np.ndarray | RawRGB):
        color = np.asarray(color)
        self._colors[label] = color
        return color

    def generate(self, label: str):
        if label in self._colors:
            return self._colors[label]

        h = self._last_salt
        rgb = hsv_to_rgb(h, 0.8, 0.8)

        self._last_salt = (h + (1 / self.classes)) % 1

        return self.register(label, rgb)

    def brighten(self, label: str, amount: float, ma: float = 1.0):
        self._colors[label] = brighten(self._colors[label], amount, ma)

    def darken(self, label: str, amount: float, mi: float = 0.2):
        self._colors[label] = darken(self._colors[label], amount, mi)


@nbu.njit
def brighten(c: np.ndarray, amount: float, ma: float):
    assert amount >= 0 and amount <= 1
    r, g, b = c
    h, s, v = rgb_to_hsv(r, g, b)

    v = min(v * (1 + amount), ma)
    rgb = hsv_to_rgb(h, s, v)

    return np.asarray(rgb)

@nbu.njit
def darken(c: np.ndarray, amount: float, mi: float):
    assert amount >= 0 and amount <= 1
    r, g, b = c
    h, s, v = rgb_to_hsv(r, g, b)

    v = max(v * (1 - amount), mi)
    rgb = hsv_to_rgb(h, s, v)

    return np.asarray(rgb)

rgb_to_hsv: Any = nbu.njit(colorsys.rgb_to_hsv, inline='always')
hsv_to_rgb: Any = nbu.njit(colorsys.hsv_to_rgb, inline='always')
