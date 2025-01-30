import torch

from copy import copy
from typing import List
from modelaquisition.acquisition.internal_face import InternalFace
from pina import LabelTensor
from pina.geometry import Location

class VolumeAcquisition(Location):

    def __init__(self, faces : List[InternalFace], bar : LabelTensor, ampiezza : float):
        super().__init__()
        self.faces = faces
        self.bar = bar
        self.variables = copy(bar.labels)
        self.ampiezza = ampiezza
        self.normals = self._normals_in_lt()
        self.bars = self._bars_in_lt()
    
    def is_inside(self, point, check_border=False):
        distances = torch.norm(point.tensor - self.bars.tensor, dim=1)
        _, idx = torch.topk(distances, k=3, largest=False)
        direction_slope = []
        k = []
        for i in range(idx.shape[0]):
            direction_slope.append(point - self.bars[idx[i]])
            k.append(torch.dot(direction_slope[i].tensor.squeeze(), self.normals[idx[i]].tensor.squeeze()))
        if all([k1 < 0 for k1 in k]):
            return True
        return False

    def sample(self, n, mode="random", variables="all"):
        sampled_points = []

        while len(sampled_points) < n:
            theta = 2*torch.pi*torch.rand(1)
            gamma = 2*torch.pi*torch.rand(1)
            rho = 2*self.ampiezza*torch.rand(1)
            x = self.bar.tensor[0, 0].item() + rho*torch.sin(theta)*torch.cos(gamma)
            y = self.bar.tensor[0, 1].item() + rho*torch.sin(theta)*torch.sin(gamma)
            z = self.bar.tensor[0, 2].item() + rho*torch.cos(theta)
            lst = [x.item(), y.item(), z.item()]
            point = LabelTensor(torch.tensor([[x.item(), y.item(), z.item()]]), labels=self.variables)
            if self.is_inside(point):
                sampled_points.append(copy(lst))
            lst.clear()
        return LabelTensor(torch.tensor(sampled_points), labels=self.variables)

    def _normals_in_lt(self):
        lst_normals = [el.acq_normal.tensor for el in self.faces]
        lst_normals = torch.cat(lst_normals, axis=0)
        return LabelTensor(lst_normals, labels=self.variables)
    
    def _bars_in_lt(self):
        lst_bars = [el.bar.tensor for el in self.faces]
        lst_bars = torch.cat(lst_bars, axis=0)
        return LabelTensor(lst_bars, labels=self.variables)