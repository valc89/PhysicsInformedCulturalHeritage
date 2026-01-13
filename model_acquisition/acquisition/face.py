"""Face sampling module."""

import math
import torch
import matplotlib.pyplot as plt

from copy import copy
from typing import List

from pina import LabelTensor
from pina.geometry import Location

class Face(Location):
    """
    Class for sampling points on a single triangular face.

    The face is defined by three 3D points (triangle vertices). The class builds
    two generating vectors lying on the triangle plane and the associated normal.
    Sampling is performed by generating candidate points on the plane and then
    keeping only those that lie inside the triangle (via barycentric coordinates).
    """

    def __init__(self, points : List[LabelTensor], limits : List[List[float]], margine : float = 2.):
        """
        Initialize the :class:`Face` sampler.

        :param list[LabelTensor] points: List of three vertices defining the
            triangular face. Each vertex is expected to contain labels
            including ``'x'``, ``'y'``, ``'z'``.
        :param list[list[float]] limits: Axis limits of the overall geometry,
            structured as ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``.
        :param float margine: Multiplicative margin used to enlarge the
            search range for candidate points. Default is ``2.0``.
        :raises AssertionError: If ``points`` does not have length 3, if
            ``limits`` does not have length 3, or if any element in ``limits``
            does not have length 2.
        """
        super().__init__()
        assert len(points) == 3, "The length of points should be 3"
        assert len(limits) == 3, "The length of limits should be 3"
        for el in limits:
            assert len(el) == 2, "The length of elements in limits should be 2"
        torch.set_default_dtype(torch.float64)
        self.limits = limits
        self.margine = margine
        self.points = points
        # Store variable labels from the first point (e.g., ['x', 'y', 'z'])
        self.variables = copy(self.points[0].labels)
        # Pre-compute sampling radius and face basis vectors
        self.rho = self._calculate_rho()
        self.generate_vec1, self.generate_vec2, self.normal_vec = self._calculate_vectors()

    def _calculate_rho(self):
        """
        Compute the sampling radius ``rho``.

        The radius is derived from:
        - the average vertex location of the triangle,
        - the maximum absolute coordinate bounds in ``limits``,
        - the ``margine`` scaling factor.

        :return: The computed sampling radius.
        :rtype: float | torch.Tensor
        """
        x_bar_partial = 0; y_bar_partial = 0; z_bar_partial = 0
        for element in self.points:
            x_bar_partial += element.extract('x')
            y_bar_partial += element.extract('y')
            z_bar_partial += element.extract('z')
        x_bar_partial /= len(self.points)
        y_bar_partial /= len(self.points)
        z_bar_partial /= len(self.points)
        x_min, x_max = self.limits[0]
        y_min, y_max = self.limits[1]
        z_min, z_max = self.limits[2]
        maximum_x = max([abs(x_min), abs(x_max)])
        maximum_y = max([abs(y_min), abs(y_max)])
        maximum_z = max([abs(z_min), abs(z_max)])
        rho_partial = math.sqrt(
            (maximum_x + x_bar_partial)**2
            + (maximum_y + y_bar_partial)**2
            + (maximum_z + z_bar_partial)**2
        )
        return rho_partial * self.margine

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
        return generate_vec1, generate_vec2, torch.unsqueeze(normal_vec, 1)

    def is_inside(self):
        """
        Placeholder for the :meth:`Location.is_inside` interface.

        This method is currently not implemented in the provided code.
        """
        pass

    def _my_is_inside(self, alpha : float, beta : float, gamma : float):
        """
        Check if barycentric coordinates correspond to a point inside the triangle.

        A point lies inside (or on the edges of) a triangle if its barycentric
        coordinates satisfy:
        ``0 <= alpha, beta, gamma <= 1``.

        :param float alpha: Barycentric weight for vertex 0.
        :param float beta: Barycentric weight for vertex 1.
        :param float gamma: Barycentric weight for vertex 2.
        :return: ``True`` if the point is inside the triangle, ``False`` otherwise.
        :rtype: bool
        """
        if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
            return True
        else:
            return False

    def sample(self, n, mode="random", variables="all"):
        """
        Sample points on the triangular face.

        The method repeatedly generates random candidate points on the triangle
        plane using the two generating vectors. For each candidate point it
        computes barycentric coordinates (alpha, beta, gamma) and keeps the
        point only if it lies inside the triangle.

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
            lambda1 = 2*self.rho*torch.rand(1) - self.rho
            lambda2 = 2*self.rho*torch.rand(1) - self.rho
            point = (
                self.points[0]
                + lambda1*(self.generate_vec1/torch.norm(self.generate_vec1))
                + lambda2*(self.generate_vec2/torch.norm(self.generate_vec2))
            ).squeeze()
            gamma = (
                (torch.linalg.cross(self.generate_vec1, point - self.points[0])
                @ self.normal_vec)
                 / (
                    torch.transpose(self.normal_vec, 1, 0) @ self.normal_vec)
            )
            beta = (
                torch.linalg.cross(self.generate_vec2, point - self.points[0])
                @ self.normal_vec
                ) / (
                    torch.transpose(self.normal_vec, 1, 0) @ self.normal_vec
            )
            alpha = 1 - gamma - beta
            if self._my_is_inside(alpha, beta, gamma):
                point = alpha*self.points[0] + beta*self.points[1] + gamma*self.points[2]
                # Check that the point lies on the face plane (within float eps)
                if torch.abs((point - self.points[0]) @ self.normal_vec) < torch.finfo(torch.float64).eps:
                    sampled_points.append([point.squeeze()[0].item(), point.squeeze()[1].item(), point.squeeze()[2].item()])
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

    def plot_samples(self, samples : LabelTensor, fig_size = (6, 6)):
        """
        Plot the triangle vertices and the sampled points in 3D.

        :param LabelTensor samples: Sampled points returned by :meth:`sample`.
        :param tuple fig_size: Figure size passed to Matplotlib. Default is ``(6, 6)``.
        :return: None
        :rtype: None
        """
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(projection="3d")
        for el in self.points:
            ax.scatter(el.extract('x'), el.extract('y'), el.extract('z'), c="r")
        self._plot_scatter(ax, samples)
        ax.set_title("Face")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()