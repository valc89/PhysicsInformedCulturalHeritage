import torch
import matplotlib.pyplot as plt

from copy import copy
from typing import List

from pina import LabelTensor
from pina.geometry import Location

class InternalFace(Location):
    """
    Classe per campionare la singola faccia dell'acquisizione
    """

    def __init__(self, points : List[LabelTensor], limits : List[List[float]], ampiezza : float, acq_normal : list, margine : float = 2.):
        super().__init__()
        assert len(points) == 3, "The length of points should be 3"
        assert len(limits) == 3, "The length of limits should be 3"
        for el in limits:
            assert len(el) == 2, "The length of elements in limits should be 2"
        self.limits = limits
        self.margine = margine
        self.ampiezza = ampiezza
        self.points = points
        self.variables = copy(self.points[0].labels)
        self.acq_normal = LabelTensor(torch.tensor([acq_normal]), labels=self.variables)
        self.bar = self._calculate_bar()
        self.generate_vec1, self.generate_vec2, self.normal_vec = self._calculate_vectors()

    def _calculate_bar(self):
        in_bar = [0, 0, 0]
        len_bar = 3
        for elem in self.points:
            for i in range(len_bar):
                in_bar[i] += elem.tensor[0, i].item()
        bar = [in_bar[i]/3 for i in range(len_bar)]
        return LabelTensor(torch.tensor([bar]), labels=self.variables)

    def _calculate_vectors(self):
        generate_vec1 = self.points[1] - self.points[0]
        generate_vec2 = self.points[2] - self.points[0]
        normal_vec = torch.linalg.cross(generate_vec1, generate_vec2).squeeze()
        return generate_vec1, generate_vec2, normal_vec

    def is_inside(self, point : LabelTensor, check_border : bool =False):
        vet = point.tensor.squeeze() - self.bar.tensor.squeeze()
        if torch.dot(vet, self.normal_vec) <= 0:
            return True
        else:
            return False

    def sample(self, n, mode="random", variables="all"):
        sampled_points = list()

        while len(sampled_points) < n:
            # movimento generico nel piano
            lambda1 = 2*self.ampiezza*torch.rand(1) - self.ampiezza
            lambda2 = 2*self.ampiezza*torch.rand(1) - self.ampiezza
            # movimento generico nello spazio
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
        ax.scatter(
            pts.extract('x'),
            pts.extract('y'),
            pts.extract('z'),
            color='blue',
            alpha=0.5
        )

    def plot_samples(self, ax, samples : LabelTensor, fig_size = (6, 6)):
        for el in self.points:
            ax.scatter(el.extract('x'), el.extract('y'), el.extract('z'), c="r")
        self._plot_scatter(ax, samples)
        ax.set_title("Face")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")