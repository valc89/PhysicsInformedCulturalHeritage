import torch

from copy import copy
from typing import List

from pina import LabelTensor
from pina.geometry import Location, CartesianDomain

class TemporalBlender(Location):

    def __init__(self, spatial_domain : Location, limiti_temporali : List[float]):
        super().__init__()
        self.spatial_domain = spatial_domain
        self.temporal_domain = CartesianDomain({'t': limiti_temporali})
        self.t_min, self.t_max = limiti_temporali
        self.variables = copy(spatial_domain.variables)
        self.variables.append('t')

    def is_inside(self, pts, check_border = True):
        spat_pts = pts.extract(self.variables[:-1])
        temp_pts = pts.extract(['t']).tensor.item()
        if check_border:
            if self.spatial_domain.is_inside(spat_pts) and self.t_min <= temp_pts <= self.t_max:
                return True
        if not check_border:
            if self.spatial_domain.is_inside(spat_pts) and self.t_min < temp_pts < self.t_max:
                return True
        return False

    def sample(self, n : int, mode="random", variables="all"):
        sampled_points = list()

        while len(sampled_points) < n:
            t_value = (self.t_max - self.t_min)*torch.rand(1) + self.t_min
            pts = copy(self.spatial_domain.sample(1, mode, variables).squeeze().tolist())
            pts.append(t_value)
            sampled_points.append(copy(pts))
            pts.clear()

        return LabelTensor(torch.tensor(sampled_points), self.variables)