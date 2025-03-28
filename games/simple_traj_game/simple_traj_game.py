import random

import matplotlib.pyplot as plt
from shapely import Polygon, Point, union_all, MultiPolygon, overlaps

from .data_cls import Attributes, Answer, SquareObstacle, AnsEval
from .utils import interp, get_square_vertices, combination, extend_line

from typing import Optional, List, Tuple


class Game:
    def __init__(self):
        map_h = 1.0
        map_w = 1.0
        self_size = 0.03

        self.attr = Attributes(
            map_size=(1.0, 1.0),
            self_radius=self_size,
            obstacles=[SquareObstacle(
                center_xy=(
                    interp(random.random(), (0.0, 1.0), (map_w * 0.2, map_w * 0.8)),
                    interp(random.random(), (0.0, 1.0), (map_h * 0.2, map_h * 0.8))
                ),
                width=interp(random.random(), (0.0, 1.0), (min(map_w, map_h) * 0.05, min(map_w, map_h) * 0.15)),
                angle=random.random(),
            ) for _ in range(5)],
            start_xy=(0.0, 0.0),
            target_xy=(1.0, 1.0),
        )

        self.ans: Optional[Answer] = None

    def get_attributes(self) -> Attributes:
        return self.attr

    def render_img(self, output_path: str):
        plt.figure(figsize=(10.0, 10.0))

        # obstacles
        vs_x = []
        vs_y = []
        obs_union_plgs = self._get_obs_union_plg()
        for plg in obs_union_plgs:
            xs, ys = plg.exterior.xy
            vs_x += [*xs, xs[0], None]
            vs_y += [*ys, ys[0], None]
        plt.plot(vs_x, vs_y, color="black")

        # traj
        if self.ans is not None:
            traj_union_plgs = self._get_traj_union_plgs()
            for plg in traj_union_plgs:
                xs, ys = plg.exterior.xy
                plt.plot(xs, ys, "-o", color="red")

        plt.axis((0.0, self.attr.map_size[0], 0.0, self.attr.map_size[1]))
        plt.savefig(output_path)

    def apply_answer(self, ans: Answer):
        self.ans = ans

    def evaluate_ans(self) -> AnsEval:
        if self.ans is None:
            raise ValueError("No answer yet!")

        return AnsEval(
            hit_infos=self._calc_hit_infos(),
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
