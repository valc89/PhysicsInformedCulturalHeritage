import math
import torch
import matplotlib.pyplot as plt

from copy import copy
from typing import List

from pina import LabelTensor
from pina.geometry import Location

class Face(Location):
    """
    Classe per campionare la singola faccia dell'acquisizione
    """

    def __init__(self, points : List[LabelTensor], limits : List[List[float]], margine : float = 2.):
        super().__init__()
        assert len(points) == 3, "The length of points should be 3"
        assert len(limits) == 3, "The length of limits should be 3"
        for el in limits:
            assert len(el) == 2, "The length of elements in limits should be 2"
        torch.set_default_dtype(torch.float64)
        self.limits = limits
        self.margine = margine
        self.points = points
        self.variables = copy(self.points[0].labels)
        self.rho = self._calculate_rho()
        self.generate_vec1, self.generate_vec2, self.normal_vec = self._calculate_vectors()

    def _calculate_rho(self):
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
        generate_vec1 = self.points[1] - self.points[0]
        generate_vec2 = self.points[2] - self.points[0]
        normal_vec = torch.linalg.cross(generate_vec1, generate_vec2).squeeze()
        return generate_vec1, generate_vec2, torch.unsqueeze(normal_vec, 1)

    def is_inside(self):
        pass

    def _my_is_inside(self, alpha : float, beta : float, gamma : float):
        if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
            return True
        else:
            return False

    def sample(self, n, mode="random", variables="all"):
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
                # Verifica che appartenga al piano
                if torch.abs((point - self.points[0]) @ self.normal_vec) < torch.finfo(torch.float64).eps:
                    sampled_points.append([point.squeeze()[0].item(), point.squeeze()[1].item(), point.squeeze()[2].item()])
        return LabelTensor(torch.tensor(sampled_points), labels=self.variables)

    def _plot_scatter(self, ax, pts):
        ax.scatter(
            pts.extract('x'),
            pts.extract('y'),
            pts.extract('z'),
            color='blue',
            alpha=0.5
        )

    def plot_samples(self, samples : LabelTensor, fig_size = (6, 6)):
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