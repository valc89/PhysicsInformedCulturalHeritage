import torch
import random

from typing import List
from modelaquisition.acquisition.face import Face
from pina import LabelTensor
from pina.geometry import Location, Union


class MyUnion(Union):

    def __init__(self, geometries : List[Face], objects = None):
        super().__init__(geometries)
        self.objects = objects
        self.range_len = range(len(self.objects))

    def is_inside(self, point, check_border=False):
        pass

    def other_is_inside(self, pts, i):
        for j in self.range_len:
            if j != i:
                app_el = self.objects[j].intern()
                if app_el.is_inside_bound(pts):
                    return True
        return False

    def sample(self, n, mode="random", variables="all"):
        sampled_points = []
        tot_elements = len(self.objects)
        num_points = n // tot_elements
        remainder = n % tot_elements
        for i in self.range_len:
            target = num_points + int(i<remainder)
            partial_points = 0
            bound = self.objects[i].boundary()
            while partial_points < target:
                pts = bound.sample(1)
                if not self.other_is_inside(pts, i):
                    sampled_points.append(
                        pts
                    )
                    partial_points += 1

        return LabelTensor(torch.cat(sampled_points), labels=self.variables)