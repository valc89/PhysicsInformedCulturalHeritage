import torch

from copy import copy
from typing import List
from modelaquisition.acquisition.internal_face import InternalFace

from pina import LabelTensor
from pina.geometry import Location

class MyIntersection(Location):
    """
    Personalized intersection for rock
    """

    def __init__(self, faces : List[InternalFace], bar : LabelTensor, ampiezza : float):
        super().__init__()
        self.faces = faces
        self.bar = bar
        self.variables = copy(bar.labels)
        self.ampiezza = ampiezza

    def is_inside(self, point, check_border=False):
        results = [element.is_inside(point) for element in self.faces]
        return all(results)

    def sample(self, n, mode="random", variables="all"):
        sampled_points = list()

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

    def sample_fixed_value(self, n : int, variable : str, fixed_value : float):
        if variable not in self.variables:
            raise ValueError("Input variable not in the set of domain variables")
        if variable == self.variables[0]:
            return self._sampled_fixed_x(n, fixed_value)
        if variable == self.variables[1]:
            return self._sampled_fixed_y(n, fixed_value)
        if variable == self.variables[2]:
            return self._sampled_fixed_z(n, fixed_value)

    def _sampled_fixed_x(self, n :int, fixed_value_x : float) -> LabelTensor:
        sampled_points = list()

        while len(sampled_points) < n:
            y = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            z = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            control = [fixed_value_x, y, z]
            if self.is_inside(LabelTensor(torch.tensor([control]), self.variables)):
                sampled_points.append(copy(control))
            control.clear()
        return LabelTensor(torch.tensor(sampled_points), self.variables)

    def _sampled_fixed_y(self, n :int, fixed_value_y : float) -> LabelTensor:
        sampled_points = list()

        while len(sampled_points) < n:
            x = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            z = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            control = [x, fixed_value_y, z]
            if self.is_inside(LabelTensor(torch.tensor([control]), self.variables)):
                sampled_points.append(copy(control))
            control.clear()
        return LabelTensor(torch.tensor(sampled_points), self.variables)

    def _sampled_fixed_z(self, n :int, fixed_value_z : float) -> LabelTensor:
        sampled_points = list()

        while len(sampled_points) < n:
            x = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            y = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            control = [x, y, fixed_value_z]
            if self.is_inside(LabelTensor(torch.tensor([control]), self.variables)):
                sampled_points.append(copy(control))
            control.clear()
        return LabelTensor(torch.tensor(sampled_points), self.variables)