import math

import numpy as np

from typing import Tuple, List


def interp(val: float, src_range: Tuple[float, float], dst_range: Tuple[float, float]):
    return (val - src_range[0]) / (src_range[1] - src_range[0]) * (dst_range[1] - dst_range[0]) + dst_range[0]


def get_square_vertices(center_xy: Tuple[float, float], width: float, angle: float) -> List[Tuple[float, float]]:
    '''
    @param: angle
        ∈[0.0, 1.0), where 0.0 is (x=1.0,y=0.0) direction and 0.0 -> 1.0 is counter-clockwise
    '''
    r = width / 2 * (2 ** 0.5)
    a = 2 * math.pi * angle

    # rotate
    vs = [
        (math.cos(a) * r, math.sin(a) * r),
        (math.cos(a + math.pi / 2) * r, math.sin(a + math.pi / 2) * r),
        (math.cos(a + math.pi) * r, math.sin(a + math.pi) * r),
        (math.cos(a - math.pi / 2) * r, math.sin(a - math.pi / 2) * r),
    ]

    # shift
    vs = [(x + center_xy[0], y + center_xy[1]) for x, y, in vs]

    return vs


def combination(arr1, arr2, *arrs, with_idx=False):
    if len(arrs) == 0:
        for i1, v1 in enumerate(arr1):
            for i2, v2 in enumerate(arr2):
                yield (v1, v2) if (not with_idx) else ((i1, v1), (i2, v2))
    else:
        for i, v in enumerate(arr1):
            for others in combination(*arrs, with_idx=with_idx):
                yield (v, *others) if (not with_idx) else ((i, v), *others)


def extend_line(xy0: Tuple[float, float], xy1: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
    '''
    Extend line to rectangle along the line direction:
                          ·------·
        ·------·    to    |      |
                          ·------·
    and return vertices
    '''
    dy = xy1[1] - xy0[1]
    dx = xy1[0] - xy0[0]
    
    # swapping dy and dx, the angle is 90° rotated!
    dx, dy = dy, -dx

    normalize_scale = (1 / (dx ** 2 + dy ** 2)) ** 0.5
    dx = dx * normalize_scale * radius
    dy = dy * normalize_scale * radius

    return [
        (xy0[0] + dx, xy0[1] + dy),
        (xy0[0] - dx, xy0[1] - dy),
        (xy1[0] - dx, xy1[1] - dy),
        (xy1[0] + dx, xy1[1] + dy),
    ]


def get_distance(xy0: Tuple[float, float], xy1: Tuple[float, float], l_lv=2) -> float:
    dy = xy1[1] - xy0[1]
    dx = xy1[0] - xy0[0]
    return (dx ** l_lv + dy ** l_lv) ** (1 / l_lv)


def get_angle_between_vecs(xy0s: np.ndarray, xy1s: np.ndarray):
    uvec_0 = xy0s / ((xy0s[...,0] ** 2 + xy0s[...,1] ** 2) ** 0.5)[...,None]
    uvec_1 = xy1s / ((xy1s[...,0] ** 2 + xy1s[...,1] ** 2) ** 0.5)[...,None]
    uvec_diff = uvec_1 - uvec_0
    lens_uvec_diff = (uvec_diff[...,0] ** 2 + uvec_diff[...,1] ** 2) ** 0.5
    return np.arcsin(lens_uvec_diff / 2) * 2
