import random
from copy import deepcopy

import matplotlib.pyplot as plt
from shapely import Polygon, Point, union_all, MultiPolygon, overlaps
import numpy as np

from .data_cls import Attributes, Answer, SquareObstacle, AnsEval
from .utils import interp, get_square_vertices, combination, extend_line, get_distance, get_angle_between_vecs

from typing import Optional, List, Tuple


class Game:
    def __init__(self):
        map_h = 1.0
        map_w = 1.0
        self_size = 0.01

        self.attr = Attributes(
            map_size=(map_w, map_w),
            self_radius=self_size,
            obstacles=[SquareObstacle(
                center_xy=(
                    interp(random.random(), (0.0, 1.0), (map_w * 0.2, map_w * 0.8)),
                    interp(random.random(), (0.0, 1.0), (map_h * 0.2, map_h * 0.8))
                ),
                width=interp(random.random(), (0.0, 1.0), (min(map_w, map_h) * 0.05, min(map_w, map_h) * 0.15)),
                angle=random.random(),
            ) for _ in range(5)],
            start_xy=(0.1, 0.1),
            target_xy=(map_w - 0.1, map_h - 0.1),
        )

        self.ans: Optional[Answer] = None
    
    def make_copy(self) -> "Game":
        new_self = Game()
        new_self.attr = self.attr.make_copy()
        new_self.ans = self.ans.make_copy() if (self.ans is not None) else None
        return new_self

    def get_attributes(self) -> Attributes:
        return self.attr

    def render_img(self, output_path: str):
        plt.figure(figsize=(10.0, 10.0), num=1, clear=True)

        # obstacles
        vs_x = []
        vs_y = []
        obs_union_plgs = self._get_obs_union_plg()
        for plg in obs_union_plgs:
            xs, ys = plg.exterior.xy
            vs_x += [*xs, xs[0], None]
            vs_y += [*ys, ys[0], None]
        plt.plot(vs_x, vs_y, color="black")

        # start/end point
        plt.scatter(*self.attr.start_xy, 100, marker="x", color="green", label="start", zorder=100)
        plt.scatter(*self.attr.target_xy, 100, marker="x", color="blue", label="target", zorder=100)

        # traj
        if self.ans is not None:
            traj_union_plgs = self._get_traj_union_plgs()
            for plg in traj_union_plgs:
                xs, ys = plg.exterior.xy
                plt.plot(xs, ys, "-.", color="red")
            plt.plot([x for x, y in self.ans.traj], [y for x, y in self.ans.traj], "-o", color="red")

        plt.axis((0.0, self.attr.map_size[0], 0.0, self.attr.map_size[1]))
        plt.legend(loc="upper left")
        plt.savefig(output_path)

    def apply_answer(self, ans: Answer):
        self.ans = ans

    def evaluate_ans(self) -> AnsEval:
        if self.ans is None:
            raise ValueError("No answer yet!")

        hit_infos = self._calc_hit_infos()
        out_of_map_infos = self._calc_out_of_map_infos()
        global_score, traj_scores = self._get_scores(hit_infos, out_of_map_infos)
        return AnsEval(
            hit_infos=hit_infos,
            out_of_map_infos=out_of_map_infos,
            global_score=global_score,
            traj_scores=traj_scores,
        )

    def _calc_hit_infos(self) -> List[Tuple[Tuple[int, int], int]]:
        '''
        @return: [((traj_idx_start, traj_idx_end), obstacle_idx)]
        '''
        if self.ans is None:
            raise ValueError("No answer yet!")

        hits: List[Tuple[Tuple[int, int], int]] = []

        # vertices hit
        for (idx_tr, tr_geo), (idx_obs, obs_geo) in combination(
            self._get_traj_ver_plgs(), self._get_obs_plgs(), with_idx=True,
        ):
            if overlaps(obs_geo, tr_geo):
                hits.append(((idx_tr, idx_tr), idx_obs))
        
        # traj_paths hit
        for (idx_tr, tr_geo), (idx_obs, obs_geo) in combination(
            self._get_traj_path_plgs(), self._get_obs_plgs(), with_idx=True,
        ):
            if overlaps(obs_geo, tr_geo):
                hits.append(((idx_tr, idx_tr+1), idx_obs))

        return hits

    def _calc_out_of_map_infos(self) -> List[bool]:
        '''
        @return: [True/False] * num_traj_points
        '''
        if self.ans is None:
            raise ValueError("No answer yet!")

        return [
            (not (0.0 <= x <= self.attr.map_size[0])) or (not (0.0 <= y <= self.attr.map_size[1]))
            for x, y in self.ans.traj
        ]

    def _get_obs_plgs(self) -> List[Polygon]:
        if self.attr._obs_plgs is None:
            self.attr._obs_plgs = [
                Polygon(get_square_vertices(obs.center_xy, obs.width, obs.angle))
                for obs in self.attr.obstacles
            ]
        return self.attr._obs_plgs

    def _get_obs_union_plg(self) -> List[Polygon]:
        if self.attr._obs_union_plgs is None:
            union_polygons = union_all(self._get_obs_plgs())
            if isinstance(union_polygons, MultiPolygon):
                self.attr._obs_union_plgs = list(union_polygons.geoms)
            elif isinstance(union_polygons, Polygon):
                self.attr._obs_union_plgs = [union_polygons]
            else:
                raise TypeError(f"{type(union_polygons)=}")
        return self.attr._obs_union_plgs

    def _get_traj_ver_plgs(self) -> List[Polygon]:
        if self.ans is None:
            raise ValueError("No answer yet!")
        if self.ans._vs_plgs is None:
            self.ans._vs_plgs = [Point(tr_xy).buffer(self.attr.self_radius) for tr_xy in self.ans.traj]
        return self.ans._vs_plgs

    def _get_traj_path_plgs(self) -> List[Polygon]:
        if self.ans is None:
            raise ValueError("No answer yet!")
        if self.ans._paths_plgs is None:
            self.ans._paths_plgs = [
                Polygon(extend_line(self.ans.traj[idx], self.ans.traj[idx+1], radius=self.attr.self_radius))
                for idx in range(len(self.ans.traj) - 1)
                if get_distance(self.ans.traj[idx], self.ans.traj[idx+1]) > 1e-6
            ]
        return self.ans._paths_plgs

    def _get_traj_union_plgs(self) -> List[Polygon]:
        if self.ans is None:
            raise ValueError("No answer yet!")
        if self.ans._union_plgs is None:
            union_polygons = union_all(self._get_traj_ver_plgs() + self._get_traj_path_plgs())
            if isinstance(union_polygons, MultiPolygon):
                self.ans._union_plgs = list(union_polygons.geoms)
            elif isinstance(union_polygons, Polygon):
                self.ans._union_plgs = [union_polygons]
            else:
                raise TypeError(f"{type(union_polygons)=}")
        return self.ans._union_plgs

    def _get_scores(
        self, hit_infos: List[Tuple[Tuple[int, int], int]], out_of_map_infos: List[bool],
    ) -> Tuple[float, np.ndarray]:
        if self.ans is None:
            raise ValueError("No answer yet!")
        
        global_score = 0.0
        traj_scores = np.zeros(len(self.ans.traj))

        # some cache values to np
        traj = np.array(self.ans.traj)
        delta_xys = traj[1:] - traj[:-1]
        target_xy = np.array(self.attr.target_xy)
        start_xy = np.array(self.attr.start_xy)
        out_of_map = np.array(out_of_map_infos)

        # hits
        recorded_hits = set()
        for (traj_start_idx, traj_end_idx), obs_idx in hit_infos:
            if (traj_end_idx, obs_idx) in recorded_hits:
                continue
            traj_scores[traj_end_idx] -= 5.0

        # out of map
        traj_scores += np.where(out_of_map, -5.0, 0.0)

        # long trajectory
        if len(traj) > 32:
            global_score -= 1.0
        
        # distance to target
        dist_to_target = np.linalg.norm(traj[-1] - target_xy)
        if dist_to_target > 1e-3:
            global_score += -2.0 * (dist_to_target / np.linalg.norm(target_xy - start_xy)).item()
        else:
            global_score += 1.0
        
        # # segment too long or short
        # traj_lens = np.linalg.norm(delta_xys, axis=1)
        # traj_scores[1:] += np.where((0.05 <= traj_lens) & (traj_lens <= 0.15), 0.0, -0.2)

        # correct direction
        delta_xys_to_target = target_xy[None,:] - traj
        angle_diffs = get_angle_between_vecs(delta_xys_to_target[:1,:], delta_xys)
        traj_scores[1:] += np.where(angle_diffs <= (10.0 / 180.0 * np.pi), +0.2, 0.0)

        return global_score, traj_scores
