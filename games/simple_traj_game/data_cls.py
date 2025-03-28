from dataclasses import dataclass

import numpy as np
from shapely import Polygon

from typing import Tuple, List, Union, Optional


@dataclass
class SquareObstacle:
    center_xy: Tuple[float, float]  # down-left_corner=(0.0, 0.0)
    width: float
    angle: float  # ∈[0.0, 1.0), where 0.0 is up and rotates clockwise

    def __post_init__(self):
        self.angle -= int(self.angle)


@dataclass
class Attributes:
    map_size: Tuple[float, float]  # w, h
    self_radius: float
    obstacles: List[SquareObstacle]
    start_xy: Tuple[float, float]  # down-left_corner=(0.0, 0.0)
    target_xy: Tuple[float, float]  # down-left_corner=(0.0, 0.0)

    # caches
    _obs_plgs: Optional[List[Polygon]] = None  # obstacles MultiPolygons
    _obs_union_plgs: Optional[List[Polygon]] = None


_TrajType = Union[List[Tuple[float, float]], np.ndarray]

@dataclass
class Answer:
    traj: _TrajType

    # caches
    _vs_plgs: Optional[List[Polygon]] = None  # vertices MultiPolygons
    _paths_plgs: Optional[List[Polygon]] = None  # paths MultiPolygons
    _union_plgs: Optional[List[Polygon]] = None


@dataclass
class AnsEval:
    hit_infos: List[Tuple[Tuple[int, int], int]]  # [((traj_idx_start, traj_idx_end), obstacle_idx)]
