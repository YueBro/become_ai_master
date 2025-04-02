from dataclasses import dataclass
from copy import deepcopy

import numpy as np
from shapely import Polygon

from typing import Tuple, List, Union, Optional


@dataclass
class SquareObstacle:
    center_xy: Tuple[float, float]  # down-left_corner=(0.0, 0.0)
    width: float
    angle: float  # ∈[0.0, 1.0), where 0.0 is (x=1.0,y=0.0) direction and 0.0 -> 1.0 is counter-clockwise

    def __post_init__(self):
        self.angle -= int(self.angle)
    
    def make_copy(self):
        return SquareObstacle(
            center_xy=deepcopy(self.center_xy),
            width=self.width,
            angle=self.angle,
        )


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

    def make_copy(self):
        return Attributes(
            map_size=deepcopy(self.map_size),
            self_radius=self.self_radius,
            obstacles=[obs.make_copy() for obs in self.obstacles],
            start_xy=deepcopy(self.start_xy),
            target_xy=deepcopy(self.target_xy),
        )


_TrajType = Union[List[Tuple[float, float]], np.ndarray]

@dataclass
class Answer:
    traj: _TrajType

    # caches
    _vs_plgs: Optional[List[Polygon]] = None  # vertices MultiPolygons
    _paths_plgs: Optional[List[Polygon]] = None  # paths MultiPolygons
    _union_plgs: Optional[List[Polygon]] = None

    def make_copy(self):
        return Answer(
            traj=deepcopy(self.traj),
        )


@dataclass
class AnsEval:
    hit_infos: List[Tuple[Tuple[int, int], int]]  # [((traj_idx_start, traj_idx_end), obstacle_idx)]
    out_of_map_infos: List[bool]  # [True/False] * num_traj_points
    global_score: float
    traj_scores: np.ndarray  # [score] * num_traj_points. Each one is score of the path, scores[0] is score of point 0

    def make_copy(self):
        return AnsEval(
            hit_infos=deepcopy(self.hit_infos),
            out_of_map_infos=deepcopy(self.out_of_map_infos),
            global_score=self.global_score,
            traj_scores=self.traj_scores.copy(),
        )
