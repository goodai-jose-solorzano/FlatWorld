import abc
from typing import List, Tuple

import numpy as np

from flat_world.helpers.PrecalculatedOrientations import PrecalculatedOrientations


class AbstractEnvEntity(abc.ABC):
    def __init__(self):
        pass
