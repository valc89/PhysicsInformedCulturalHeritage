"""Custom intersection sampling module."""

import torch

from copy import copy
from typing import List
from model_acquisition.acquisition import InternalFace

from pina import LabelTensor
from pina.geometry import Location

class MyIntersection(Location):
    """
    Custom intersection domain defined by multiple :class:`InternalFace` objects.

    A point is considered inside the domain if it satisfies the ``is_inside``
    condition for all the provided faces. Sampling is performed by generating
    random candidate points around a given barycenter and filtering them through
    the intersection check.
    """

    def __init__(self, faces : List[InternalFace], bar : LabelTensor, ampiezza : float):
        """
        Initialize the :class:`MyIntersection` class.

        :param list[InternalFace] faces: List of faces defining the intersection
            constraints. A point must be inside each face to be accepted.
        :param LabelTensor bar: Reference point (barycenter) used as center for
            random sampling.
        :param float ampiezza: Amplitude used to bound the sampling region.
        """
        super().__init__()
        self.faces = faces
        self.bar = bar
        self.variables = copy(bar.labels)
        self.ampiezza = ampiezza

    def is_inside(self, point, check_border=False):
        """
        Check whether a point lies inside the intersection domain.

        The point is inside if :meth:`InternalFace.is_inside` returns ``True``
        for all faces.

        :param LabelTensor point: Point to test.
        :param bool check_border: Border handling flag (present in signature
            but not used in this implementation). Default is ``False``.
        :return: ``True`` if the point satisfies all face constraints,
            ``False`` otherwise.
        :rtype: bool
        """
        results = [element.is_inside(point) for element in self.faces]
        return all(results)

    def sample(self, n, mode="random", variables="all"):
        """
        Sample points inside the intersection domain.

        Candidate points are generated around ``self.bar`` using spherical
        coordinates (theta, gamma, rho) within a radius bounded by ``ampiezza``.
        Points are accepted only if they pass :meth:`is_inside`.

        :param int n: Number of points to sample.
        :param str mode: Sampling mode parameter (present for compatibility with
            the expected interface, but not used in this implementation).
            Default is ``"random"``.
        :param str variables: Variable selection parameter (present for
            compatibility with the expected interface, but not used in this
            implementation). Default is ``"all"``.
        :return: Sampled points as a :class:`~pina.LabelTensor` with labels
            matching ``bar.labels``.
        :rtype: LabelTensor
        """
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
        """
        Sample points inside the domain by fixing one coordinate.

        The method selects one of the private sampling routines depending on the
        chosen variable label (first, second, or third label in ``self.variables``).

        :param int n: Number of points to sample.
        :param str variable: Variable name to fix. Must be contained in
            ``self.variables``.
        :param float fixed_value: Value assigned to the fixed variable.
        :raises ValueError: If ``variable`` is not in ``self.variables``.
        :return: Sampled points with one fixed coordinate.
        :rtype: LabelTensor
        """
        if variable not in self.variables:
            raise ValueError("Input variable not in the set of domain variables")
        if variable == self.variables[0]:
            return self._sampled_fixed_x(n, fixed_value)
        if variable == self.variables[1]:
            return self._sampled_fixed_y(n, fixed_value)
        if variable == self.variables[2]:
            return self._sampled_fixed_z(n, fixed_value)

    def _sampled_fixed_x(self, n :int, fixed_value_x : float) -> LabelTensor:
        """
        Sample points inside the domain with fixed x coordinate.

        :param int n: Number of points to sample.
        :param float fixed_value_x: Fixed x value.
        :return: Sampled points with x fixed to ``fixed_value_x``.
        :rtype: LabelTensor
        """
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
        """
        Sample points inside the domain with fixed y coordinate.

        :param int n: Number of points to sample.
        :param float fixed_value_y: Fixed y value.
        :return: Sampled points with y fixed to ``fixed_value_y``.
        :rtype: LabelTensor
        """
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
        """
        Sample points inside the domain with fixed z coordinate.

        :param int n: Number of points to sample.
        :param float fixed_value_z: Fixed z value.
        :return: Sampled points with z fixed to ``fixed_value_z``.
        :rtype: LabelTensor
        """
        sampled_points = list()

        while len(sampled_points) < n:
            x = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            y = torch.distributions.Uniform(-self.ampiezza, self.ampiezza).sample((1, 1))
            control = [x, y, fixed_value_z]
            if self.is_inside(LabelTensor(torch.tensor([control]), self.variables)):
                sampled_points.append(copy(control))
            control.clear()
        return LabelTensor(torch.tensor(sampled_points), self.variables)