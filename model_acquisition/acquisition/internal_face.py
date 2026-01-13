"""Internal face sampling module."""

import torch
import matplotlib.pyplot as plt

from copy import copy
from typing import List

from pina import LabelTensor
from pina.geometry import Location

class InternalFace(Location):
    """
    Class for sampling points on the internal side of a triangular face.

    The face is defined by three vertices. The class computes:
    - the triangle barycenter,
    - two in-plane generating vectors,
    - the face normal (via cross product).

    Sampling is performed by generating random displacements on the triangle
    plane and then shifting points along the normal direction to move "inside"
    according to the orientation check implemented in :meth:`is_inside`.
    """

    def __init__(self, points : List[LabelTensor], limits : List[List[float]], ampiezza : float, acq_normal : list, margine : float = 2.):
        """
        Initialize the :class:`InternalFace` sampler.

        :param list[LabelTensor] points: List of three vertices defining the
            triangular face. Each vertex is expected to contain labels
            including ``'x'``, ``'y'``, ``'z'``.
        :param list[list[float]] limits: Axis limits of the overall geometry,
            structured as ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``.
        :param float ampiezza: Amplitude used as bound for random displacements
            both in-plane and along the normal direction.
        :param list acq_normal: Acquisition normal stored as a
            :class:`~pina.LabelTensor` with the same labels as the vertices.
        :param float margine: Margin parameter stored in the object (not used
            by the provided sampling implementation). Default is ``2.0``.
        :raises AssertionError: If ``points`` does not have length 3, if
            ``limits`` does not have length 3, or if any element in ``limits``
            does not have length 2.
        """
        super().__init__()
        assert len(points) == 3, "The length of points should be 3"
        assert len(limits) == 3, "The length of limits should be 3"
        for el in limits:
            assert len(el) == 2, "The length of elements in limits should be 2"
        self.limits = limits
        self.margine = margine
        self.ampiezza = ampiezza
        self.points = points
        # Store variable labels from the first point (e.g., ['x', 'y', 'z'])
        self.variables = copy(self.points[0].labels)
        # Store acquisition normal as a LabelTensor
        self.acq_normal = LabelTensor(torch.tensor([acq_normal]), labels=self.variables)
        # Pre-compute barycenter and face vectors
        self.bar = self._calculate_bar()
        self.generate_vec1, self.generate_vec2, self.normal_vec = self._calculate_vectors()

    def _calculate_bar(self):
        """
        Compute the barycenter of the triangular face.

        The barycenter is computed as the arithmetic mean of the three vertices.

        :return: The barycenter as a :class:`~pina.LabelTensor`.
        :rtype: LabelTensor
        """
        in_bar = [0, 0, 0]
        len_bar = 3
        for elem in self.points:
            for i in range(len_bar):
                in_bar[i] += elem.tensor[0, i].item()
        bar = [in_bar[i]/3 for i in range(len_bar)]
        return LabelTensor(torch.tensor([bar]), labels=self.variables)

    def _calculate_vectors(self):
        """
        Compute triangle generating vectors and the face normal.

        The method builds:
        - ``generate_vec1 = p1 - p0``
        - ``generate_vec2 = p2 - p0``
        - ``normal_vec = cross(generate_vec1, generate_vec2)``

        :return: The two generating vectors and the normal vector.
        :rtype: tuple[LabelTensor, LabelTensor, torch.Tensor]
        """
        generate_vec1 = self.points[1] - self.points[0]
        generate_vec2 = self.points[2] - self.points[0]
        normal_vec = torch.linalg.cross(generate_vec1, generate_vec2).squeeze()
        return generate_vec1, generate_vec2, normal_vec

    def is_inside(self, point : LabelTensor, check_border : bool =False):
        """
        Check whether a point is on the "internal" side of the face.

        The check is performed by:
        - computing the vector from the barycenter to the point,
        - taking the dot product with the face normal,
        - returning ``True`` if the dot product is ``<= 0``.

        :param LabelTensor point: Point to test.
        :param bool check_border: Border handling flag (present in signature
            but not used by this implementation). Default is ``False``.
        :return: ``True`` if the point is considered inside, ``False`` otherwise.
        :rtype: bool
        """
        vet = point.tensor.squeeze() - self.bar.tensor.squeeze()
        if torch.dot(vet, self.normal_vec) <= 0:
            return True
        else:
            return False

    def sample(self, n, mode="random", variables="all"):
        """
        Sample points on the internal side of the face.

        The method generates candidate points by:
        - applying random displacements along the two in-plane generating vectors,
        - subtracting a random displacement along the face normal to move points
          to one side of the plane,
        - filtering candidates using :meth:`is_inside`.

        :param int n: Number of points to sample.
        :param str mode: Sampling mode parameter (present for compatibility with
            the expected interface, but not used in this implementation).
            Default is ``"random"``.
        :param str variables: Variable selection parameter (present for
            compatibility with the expected interface, but not used in this
            implementation). Default is ``"all"``.
        :return: Sampled points as a :class:`~pina.LabelTensor` with the same
            labels as the input vertices.
        :rtype: LabelTensor
        """
        sampled_points = list()

        while len(sampled_points) < n:
            # Generic displacement on the plane
            lambda1 = 2*self.ampiezza*torch.rand(1) - self.ampiezza
            lambda2 = 2*self.ampiezza*torch.rand(1) - self.ampiezza
            # Generic displacement along the normal direction
            space_mov = 2*self.ampiezza*torch.rand(1)
            add_point = (
                self.bar.tensor
                + lambda1*(self.generate_vec1/torch.norm(self.generate_vec1))
                + lambda2*(self.generate_vec2/torch.norm(self.generate_vec2))
                - space_mov*self.normal_vec).tolist()
            point = LabelTensor(torch.tensor(add_point), labels=self.variables)
            if self.is_inside(point):
                sampled_points.append(add_point[0])
        return LabelTensor(torch.tensor(sampled_points), labels=self.variables)

    def _plot_scatter(self, ax, pts):
        """
        Plot a 3D scatter of sampled points on a given Matplotlib axis.

        :param matplotlib.axes.Axes ax: A 3D Matplotlib axis.
        :param LabelTensor pts: Points to scatter plot.
        """
        ax.scatter(
            pts.extract('x'),
            pts.extract('y'),
            pts.extract('z'),
            color='blue',
            alpha=0.5
        )

    def plot_samples(self, ax, samples : LabelTensor, fig_size = (6, 6)):
        """
        Plot the triangle vertices and the sampled points in 3D.

        :param matplotlib.axes.Axes ax: A 3D Matplotlib axis where the data
            will be plotted.
        :param LabelTensor samples: Sampled points returned by :meth:`sample`.
        :param tuple fig_size: Figure size parameter (present in signature but
            not used by this implementation). Default is ``(6, 6)``.
        :return: None
        :rtype: None
        """
        for el in self.points:
            ax.scatter(el.extract('x'), el.extract('y'), el.extract('z'), c="r")
        self._plot_scatter(ax, samples)
        ax.set_title("Face")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")