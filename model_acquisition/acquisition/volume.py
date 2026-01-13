"""Volume acquisition module."""
import torch

from copy import copy
from typing import List
from pina import LabelTensor
from pina.geometry import Location
from model_acquisition.acquisition import InternalFace

class VolumeAcquisition(Location):

    def __init__(self, faces : List[InternalFace], bar : LabelTensor, ampiezza : float):
        """
        Volume-like domain built from a set of :class:`InternalFace` elements.

        The class stores face barycenters and associated acquisition normals in
        :class:`~pina.LabelTensor` form. A point is classified as inside by checking
        its orientation with respect to the normals of the nearest face barycenters
        (k-nearest selection).
        """
        super().__init__()
        self.faces = faces
        self.bar = bar
        self.variables = copy(bar.labels)
        self.ampiezza = ampiezza
        # Normals and barycenters as LabelTensors
        self.normals = self._normals_in_lt()
        self.bars = self._bars_in_lt()
    
    def is_inside(self, point, check_border=False, v_k = 5):
        """
        Check whether a point is inside the volume.

        The method:
        - computes distances between the point and all stored face barycenters,
        - selects the ``v_k`` closest barycenters,
        - computes dot products between (point - barycenter) and the corresponding
          face normals,
        - classifies the point as inside if all dot products are strictly negative.

        :param LabelTensor point: Point to test.
        :param bool check_border: Border handling flag (present in signature
            but not used in this implementation). Default is ``False``.
        :param int v_k: Number of nearest barycenters/normals used in the test.
            Default is ``5``.
        :return: ``True`` if the point is considered inside, ``False`` otherwise.
        :rtype: bool
        """
        distances = torch.norm(point.tensor - self.bars.tensor, dim=1)
        _, idx = torch.topk(distances, k=v_k, largest=False)
        direction_slope = []
        k = []
        for i in range(idx.shape[0]):
            direction_slope.append(point - self.bars[idx[i]])
            k.append(torch.dot(direction_slope[i].tensor.squeeze(), self.normals[idx[i]].tensor.squeeze()))
        if all([k1 < 0 for k1 in k]):
            return True
        return False
    
    def is_inside_bound(self, point, check_border=False, v_k = 5):
        """
        Check whether a point is inside the volume including the boundary.

        This method mirrors :meth:`is_inside` but uses a non-strict condition:
        the point is accepted if all dot products are ``<= 0``.

        :param LabelTensor point: Point to test.
        :param bool check_border: Border handling flag (present in signature
            but not used in this implementation). Default is ``False``.
        :param int v_k: Number of nearest barycenters/normals used in the test.
            Default is ``5``.
        :return: ``True`` if the point is considered inside (or on boundary),
            ``False`` otherwise.
        :rtype: bool
        """
        distances = torch.norm(point.tensor - self.bars.tensor, dim=1)
        _, idx = torch.topk(distances, k=v_k, largest=False)
        direction_slope = []
        k = []
        for i in range(idx.shape[0]):
            direction_slope.append(point - self.bars[idx[i]])
            k.append(torch.dot(direction_slope[i].tensor.squeeze(), self.normals[idx[i]].tensor.squeeze()))
        if all([k1 <= 0 for k1 in k]):
            return True
        return False

    def sample(self, n, mode="random", variables="all"):
        """
        Sample points inside the volume.

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
        """
        Build a :class:`~pina.LabelTensor` containing all face acquisition normals.

        The normals are extracted from each face as ``face.acq_normal.tensor``
        and concatenated along the first dimension.

        :return: A :class:`~pina.LabelTensor` containing stacked normals.
        :rtype: LabelTensor
        """
        lst_normals = [el.acq_normal.tensor for el in self.faces]
        lst_normals = torch.cat(lst_normals, axis=0)
        return LabelTensor(lst_normals, labels=self.variables)
    
    def _bars_in_lt(self):
        """
        Build a :class:`~pina.LabelTensor` containing all face barycenters.

        The barycenters are extracted from each face as ``face.bar.tensor`` and
        concatenated along the first dimension.

        :return: A :class:`~pina.LabelTensor` containing stacked barycenters.
        :rtype: LabelTensor
        """
        lst_bars = [el.bar.tensor for el in self.faces]
        lst_bars = torch.cat(lst_bars, axis=0)
        return LabelTensor(lst_bars, labels=self.variables)