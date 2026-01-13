"""Custom union sampling module."""
import torch

from typing import List
from pina import LabelTensor
from pina.geometry import Union
from model_acquisition.acquisition import Face


class MyUnion(Union):
    """
    Custom union geometry based on a list of face-like objects.

    The class extends :class:`~pina.geometry.Union` and provides a sampling
    routine that draws points from the boundary of each object while avoiding
    points that lie inside the boundary of the other objects.
    """

    def __init__(self, geometries : List[Face], objects = None):
        """
        Initialize the :class:`MyUnion` class.

        :param list[Face] geometries: List of geometries passed to the parent
            :class:`~pina.geometry.Union` constructor.
        :param objects: Sequence of objects used by the custom sampling logic.
            Each element is expected to provide ``boundary()``, ``intern()``,
            and the returned geometries are expected to provide
            ``sample(...)`` and ``is_inside_bound(...)`` methods, as used in
            this implementation. Default is ``None``.
        """
        super().__init__(geometries)
        self.objects = objects
        self.range_len = range(len(self.objects))

    def is_inside(self, point, check_border=False):
        """
        Placeholder for a custom inside check.

        This method is currently not implemented in the provided code.

        :param LabelTensor point: Point to test.
        :param bool check_border: Border handling flag (present in signature).
        """
        pass

    def other_is_inside(self, pts, i):
        """
        Check whether a sampled point is inside the boundary of any other object.

        The method iterates over all objects except the one at index ``i`` and
        tests the point against the internal geometry boundary check.

        :param LabelTensor pts: Point(s) to test.
        :param int i: Index of the object currently being sampled.
        :return: ``True`` if the point is inside the boundary of any other
            object, ``False`` otherwise.
        :rtype: bool
        """
        for j in self.range_len:
            if j != i:
                app_el = self.objects[j].intern()
                if app_el.is_inside_bound(pts):
                    return True
        return False

    def sample(self, n, mode="random", variables="all"):
        """
        Sample points from the union boundary while avoiding overlaps.

        The method splits the total number of requested points across the
        available objects. For each object, it samples points from its boundary
        and keeps them only if they are not inside the boundary of any other
        object (checked via :meth:`other_is_inside`).

        :param int n: Total number of points to sample.
        :param str mode: Sampling mode parameter (present for compatibility with
            the expected interface, but not used in this implementation).
            Default is ``"random"``.
        :param str variables: Variable selection parameter (present for
            compatibility with the expected interface, but not used in this
            implementation). Default is ``"all"``.
        :return: Sampled points as a single :class:`~pina.LabelTensor`.
        :rtype: LabelTensor
        """
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