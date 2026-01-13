"""Blend-to-PINA conversion module."""
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt

from copy import copy

from model_acquisition.acquisition import (
    Blender, Face, InternalFace, MyUnion, 
    TemporalBlender, VolumeAcquisition
)

from pina import LabelTensor
from pina.geometry import Union

torch.set_default_dtype(torch.float64)

class Blend2Pina(Blender):
    """
    Class to convert an acquired Blender model into PINA geometries.

    The class inherits from :class:`~model_acquisition.acquisition.Blender` to
    open a ``.blend`` file and extract mesh dictionaries. For each acquired mesh
    object, a :class:`SingleBlender` is created. The class exposes:
    - a boundary geometry (union of face geometries),
    - an internal geometry (volume-like acquisition based on internal faces),
    - optional spatio-temporal wrapping via :class:`TemporalBlender`,
    - basic plotting and CSV export utilities for sampled points.
    """

    total_variables = ['x', 'y', 'z']

    def __init__(self, filename):
        """
        Initialize the :class:`Blend2Pina` class.

        :param str filename: Path to the Blender ``.blend`` file.
        """
        super().__init__(filename)
        self.lst_model_dict = self.acquire_model()
        self.len_lst = len(self.lst_model_dict)
        # Variables are inferred from vertex dimensionality (plus one index in the original code)
        self.variables = self.total_variables[:(len(self.lst_model_dict[0]["vertices"][0])+1)]
        # Build a SingleBlender helper for each acquired object dictionary
        self.single_obj_lst = [SingleBlender(el, self.variables) for el in self.lst_model_dict]
    
    def boundary(self, time_interval = None):
        """
        Return the boundary geometry for the acquired model.

        If ``time_interval`` is provided as a list of two elements, the returned
        boundary geometry is wrapped by :class:`TemporalBlender`.

        :param list[float] | None time_interval: Optional time interval
            ``[t_min, t_max]`` used to create a spatio-temporal boundary.
        :raises RuntimeError: If ``time_interval`` is not a list or does not
            have exactly two elements.
        :return: Boundary geometry (possibly wrapped in :class:`TemporalBlender`).
        :rtype: Location
        """
        if time_interval == None:
            return self._single_boundary()
        else:
            if isinstance(time_interval, list):
                if len(time_interval) == 2:
                    return TemporalBlender(self._single_boundary(), time_interval)
                else:
                    raise  RuntimeError("time_interval has not 2 elements")
            else:
                raise  RuntimeError("time_interval has to be a list")
    
    def _single_boundary(self):
        """
        Build the boundary geometry without temporal wrapping.

        If only one object is present, returns its boundary directly.
        Otherwise returns a :class:`MyUnion` of all object boundaries.

        :return: Boundary geometry for the model.
        :rtype: Location
        """
        if self.len_lst == 1:
            return self.single_obj_lst[0].boundary()
        else:
            return MyUnion([el.boundary() for el in self.single_obj_lst], [el for el in self.single_obj_lst])
    
    def intern(self, time_interval = None):
        """
        Return the internal geometry for the acquired model.

        If ``time_interval`` is provided as a list of two elements, the returned
        internal geometry is wrapped by :class:`TemporalBlender`.

        :param list[float] | None time_interval: Optional time interval
            ``[t_min, t_max]`` used to create a spatio-temporal internal domain.
        :raises RuntimeError: If ``time_interval`` is not a list or does not
            have exactly two elements.
        :return: Internal geometry (possibly wrapped in :class:`TemporalBlender`).
        :rtype: Location
        """
        if time_interval == None:
            return self._single_intern()
        else:
            if isinstance(time_interval, list):
                if len(time_interval) == 2:
                    return TemporalBlender(self._single_intern(), time_interval)
                else:
                    raise  RuntimeError("time_interval has not 2 elements")
            else:
                raise  RuntimeError("time_interval has to be a list")
    
    def _single_intern(self):
        """
        Build the internal geometry without temporal wrapping.

        If only one object is present, returns its internal geometry directly.
        Otherwise returns a :class:`~pina.geometry.Union` of all internal geometries.

        :return: Internal geometry for the model.
        :rtype: Location
        """
        if self.len_lst == 1:
            return self.single_obj_lst[0].intern()
        else:
            return Union([el.intern() for el in self.single_obj_lst])
    
    def _plot_scatter(self, ax, pts):
        """
        Plot a 3D scatter of points on a given Matplotlib axis.

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
    
    def plot(self, pts : LabelTensor, fig_size = (8, 6)):
        """
        Plot 3D points using Matplotlib.

        :param LabelTensor pts: Points to plot.
        :param tuple fig_size: Figure size passed to Matplotlib.
            Default is ``(8, 6)``.
        :return: None
        :rtype: None
        """
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(projection="3d")
        self._plot_scatter(ax, pts)
        ax.set_title("Face")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    
    def to_csv(self, pts, filepath, sep = ";"):
        """
        Export points to a CSV file.

        :param LabelTensor pts: Points to export.
        :param str filepath: Output path prefix used to create the CSV file name.
        :param str sep: CSV separator. Default is ``";"``.
        :return: None
        :rtype: None
        """
        df = pd.DataFrame(
            data=pts.tensor.detach().numpy(),
            columns=self.variables
        )
        df.to_csv(filepath + "csv", sep, index=False)


class SingleBlender(object):
    """
    Helper class to build PINA geometries from a single acquired Blender object.

    The class:
    - computes basic features (barycenter, radius-like value, coordinate intervals),
    - builds face geometries (:class:`Face`) and internal face geometries
      (:class:`InternalFace`) from the object faces,
    - exposes a boundary union of faces and an internal :class:`VolumeAcquisition`.
    """
    
    def __init__(self, model_dict, variables):
        """
        Initialize the :class:`SingleBlender` helper.

        :param dict model_dict: Object dictionary produced by :class:`Blender`.
        :param list[str] variables: Labels used to build :class:`LabelTensor`
            objects (e.g., ``['x', 'y', 'z']``).
        """
        self.model_dict = model_dict
        self.variables = variables
        self.features = self._acquire_features_model()
        self.total_point_lst, self.total_faces_lst, self.total_int_faces_lst = self._acquire_total_points_and_faces()
        self.torch_bar = LabelTensor(torch.tensor([self.features['baricenter']]), labels=self.variables)
    
    def _acquire_features_model(self) -> dict:
        """
        Compute simple geometric features from the acquired object.

        The method iterates over all vertices, applying scale and translation
        (as implemented in the provided code) to compute:
        - the barycenter of vertices,
        - a radius-like value ``rho`` based on max absolute coordinates and barycenter,
        - coordinate intervals for x/y/z.

        :return: A dictionary containing ``baricenter``, ``rho`` and ``interval``.
        :rtype: dict
        """
        x_bar , y_bar, z_bar = 0, 0, 0
        for el in self.model_dict["vertices"]:
            x, y, z = el
            x = x * self.model_dict["scale"][0] + self.model_dict["location"][0]
            y = y * self.model_dict["scale"][1] + self.model_dict["location"][1]
            z = z * self.model_dict["scale"][2] + self.model_dict["location"][2]
            x_bar += x
            y_bar += y
            z_bar += z
            if self.model_dict["vertices"].index(el) == 0:
                x_max_all, y_max_all, z_max_all = abs(x), abs(y), abs(z)
                x_max , y_max, z_max = x, y, z
                x_min , y_min, z_min = x, y, z
            else:
                x_max_all = max(x_max_all, abs(x))
                y_max_all = max(y_max_all, abs(y))
                z_max_all = max(z_max_all, abs(z))
                x_max = max(x_max, x)
                x_min = min(x_min, x)
                y_max = max(y_max, y)
                y_min = min(y_min, y)
                z_max = max(z_max, z)
                z_min = min(z_min, z)
        tot_points = len(self.model_dict["vertices"])
        x_bar /= tot_points
        y_bar /= tot_points
        z_bar /= tot_points
        rho = math.sqrt((x_max_all-x_bar)**2 + (y_max_all-y_bar)**2 + (z_max_all-z_bar)**2)
        return {
            'baricenter' : [x_bar, y_bar, z_bar],
            'rho' : rho,
            'interval' : {
                'x' : [x_min, x_max],
                'y' : [y_min, y_max],
                'z' : [z_min, z_max]
            }
        }
    
    def print_features(self):
        """
        Print the computed features dictionary to stdout.

        :return: None
        :rtype: None
        """
        for key, value in self.features.items():
            if not isinstance(value, dict):
                print(f"{key}: {value}")
            else:
                print(key)
                for key_int, value_int in value.items():
                    print(f"{key_int}: {value_int}")
    
    def _acquire_total_points_and_faces(self):
        """
        Build per-face point lists and corresponding face geometry objects.

        For each face in ``model_dict["faces"]`` the method:
        - retrieves the vertex indices,
        - computes scaled+translated coordinates for the three face vertices,
        - builds three :class:`LabelTensor` vertices,
        - creates one :class:`InternalFace` and one :class:`Face`,
        - stores them in lists returned at the end.

        :return: Tuple containing:
            - ``total_points_lst``: list of per-face coordinate dictionaries,
            - ``total_faces_lst``: list of :class:`Face` objects,
            - ``total_int_faces_lst``: list of :class:`InternalFace` objects.
        :rtype: tuple[list, list, list]
        """
        total_points_lst = list()
        total_faces_lst = list()
        total_int_faces_lst = list()
        for i in range(len(self.model_dict["faces"])):
            idx_lst = copy(self.model_dict["faces"][i])
            x = [self.model_dict["scale"][0] * self.model_dict["vertices"][idx][0] + self.model_dict["location"][0] for idx in idx_lst]
            y = [self.model_dict["scale"][1] * self.model_dict["vertices"][idx][1] + self.model_dict["location"][1] for idx in idx_lst]
            z = [self.model_dict["scale"][2] * self.model_dict["vertices"][idx][2] + self.model_dict["location"][2] for idx in idx_lst]
            total_points_lst.append(
                {
                    "x" : copy(x),
                    "y" : copy(y),
                    "z" : copy(z),
                    "xlim" : [min(x), max(x)],
                    "ylim" : [min(y), max(y)],
                    "zlim" : [min(z), max(z)]
                }
            )
            limits = [
                copy([min(copy(x)), max(copy(x))]),
                copy([min(copy(y)), max(copy(y))]),
                copy([min(copy(z)), max(copy(z))])
            ]
            input_values = [
                LabelTensor(
                    torch.tensor([[x[0], y[0], z[0]]]),
                    self.variables
                ),
                LabelTensor(
                    torch.tensor([[x[1], y[1], z[1]]]),
                    self.variables
                ),
                LabelTensor(
                    torch.tensor([[x[2], y[2], z[2]]]),
                    self.variables
                )
            ]
            total_int_faces_lst.append(
                InternalFace(
                    copy(input_values),
                    copy(limits),
                    self.features['rho'],
                    copy(self.model_dict["normals"][i])
                )
            )
            total_faces_lst.append(
                Face(
                    copy(input_values),
                    copy(limits)
                )
            )
            # Cleaning of lists
            input_values.clear(); limits.clear()
            x.clear(); y.clear(); z.clear()
        return total_points_lst, total_faces_lst, total_int_faces_lst
    
    def boundary(self):
        """
        Return the boundary geometry for the single object.

        :return: Union of all :class:`Face` objects built from the faces list.
        :rtype: Union
        """
        return Union(self.total_faces_lst)
    
    def intern(self):
        """
        Return the internal geometry for the single object.

        :return: A :class:`VolumeAcquisition` built from internal faces, using
            the object barycenter and ``rho`` as sampling amplitude.
        :rtype: VolumeAcquisition
        """
        return VolumeAcquisition(
            self.total_int_faces_lst,
            bar=self.torch_bar,
            ampiezza=self.features['rho']
        )