"""Spatio-temporal domain wrapper module."""
import torch

from copy import copy
from typing import List

from pina import LabelTensor
from pina.geometry import Location, CartesianDomain

class TemporalBlender(Location):
    """
    Spatio-temporal wrapper for a spatial :class:`~pina.geometry.Location`.

    This class combines an input spatial domain with a 1D temporal domain
    defined over a variable ``t``. Points are considered inside if their spatial
    part lies inside the provided spatial domain and their temporal coordinate
    lies within the specified time interval.
    """

    def __init__(self, spatial_domain : Location, limiti_temporali : List[float]):
        """
        Initialize the :class:`TemporalBlender` class.

        :param Location spatial_domain: Spatial domain used for the spatial
            inside check and spatial sampling.
        :param list[float] limiti_temporali: Time interval limits for ``t`` as
            ``[t_min, t_max]``.
        """
        super().__init__()
        self.spatial_domain = spatial_domain
        self.temporal_domain = CartesianDomain({'t': limiti_temporali})
        self.t_min, self.t_max = limiti_temporali
        # Build variable list by extending spatial variables with 't'
        self.variables = copy(spatial_domain.variables)
        self.variables.append('t')

    def is_inside(self, pts, check_border = True):
        """
        Check whether spatio-temporal points are inside the domain.

        The method:
        - extracts the spatial coordinates and checks them against
          ``self.spatial_domain.is_inside``,
        - extracts the temporal coordinate ``t`` and checks it against the
          interval ``[t_min, t_max]`` (inclusive if ``check_border=True``,
          exclusive otherwise).

        :param LabelTensor pts: Points to test.
        :param bool check_border: If ``True`` the time interval is checked
            inclusively (``t_min <= t <= t_max``). If ``False`` the interval
            is checked strictly (``t_min < t < t_max``). Default is ``True``.
        :return: ``True`` if the point satisfies both spatial and temporal
            constraints, ``False`` otherwise.
        :rtype: bool
        """
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
        """
        Sample spatio-temporal points from the combined domain.

        The method samples spatial points using ``self.spatial_domain.sample(1, ...)``
        and independently samples a time value ``t`` uniformly in ``[t_min, t_max]``.
        Each returned point is the concatenation of the spatial coordinates and ``t``.

        :param int n: Number of points to sample.
        :param str mode: Sampling mode parameter forwarded to the spatial domain
            sampling. Default is ``"random"``.
        :param str variables: Variable selection parameter forwarded to the
            spatial domain sampling. Default is ``"all"``.
        :return: Sampled spatio-temporal points as a :class:`~pina.LabelTensor`
            with labels equal to spatial variables plus ``'t'``.
        :rtype: LabelTensor
        """
        sampled_points = list()

        while len(sampled_points) < n:
            t_value = (self.t_max - self.t_min)*torch.rand(1) + self.t_min
            pts = copy(self.spatial_domain.sample(1, mode, variables).squeeze().tolist())
            pts.append(t_value)
            sampled_points.append(copy(pts))
            pts.clear()

        return LabelTensor(torch.tensor(sampled_points), self.variables)