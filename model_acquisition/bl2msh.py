"""Blend-to-Gmsh meshing module."""
import os
import gmsh
import numpy as np
import pandas as pd

from copy import copy
from model_acquisition.acquisition import Blender

class Blend2Mesh(Blender):
    """
    Class to generate a Gmsh mesh starting from a Blender acquisition.

    The class inherits from :class:`~model_acquisition.acquisition.Blender` to:
    - open a ``.blend`` file and extract mesh data as dictionaries,
    - build the corresponding Gmsh OCC geometry (points, lines, surfaces, volumes),
    - generate and save a ``.msh`` file.

    The class also stores basic mesh node information after meshing.
    """

    def __init__(self, filename, model_name = "model"):
        """
        Initialize the :class:`Blend2Mesh` class.

        :param str filename: Path to the Blender ``.blend`` file.
        :param str model_name: Name used for the Gmsh model and output mesh file.
            Default is ``"model"``.
        """
        super().__init__(filename)
        self.model_name = model_name
        self.lst_model_dict = self.acquire_model()
        self._num_points_msh = None
        self._coord_points_msh = None

    def create_mesh(self, lc=1e-14, len_msh=.1):
        """
        Create a single mesh by adding all acquired objects to one Gmsh model.

        The method:
        - initializes Gmsh,
        - for each acquired object dictionary adds points, curves, surfaces and a volume,
        - generates a 3D mesh and writes ``<model_name>.msh``,
        - stores the total number of mesh nodes and their coordinates.

        :param float lc: Local characteristic length assigned to points
            (passed as ``meshSize`` in :func:`gmsh.model.occ.addPoint`).
            Default is ``1e-14``.
        :param float len_msh: Global maximum characteristic length set via
            ``Mesh.CharacteristicLengthMax``. Default is ``0.1``.
        :return: None
        :rtype: None
        """
        dim = 3
        tot_acquired = 0
        lst_elements = []

        # Inizialize gmsh
        gmsh.initialize()
        gmsh.model.add(self.model_name)
        
        for el in self.lst_model_dict:
            # Add of points
            lst_tag_points = self._add_points(el, lc)

            # Inclusion of curves
            lst_tag_curves = self._add_curves(el, tot_acquired)
            tot_acquired += len(lst_tag_points)

            # Incluson of surfaces
            lst_surfaces = self._add_surfaces(lst_tag_curves)

            # Inclusion of volums
            idx_volum = self._add_volume(lst_surfaces)
            lst_elements.append(idx_volum)
            gmsh.model.occ.synchronize()

        # Create mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", len_msh)
        gmsh.model.mesh.generate(dim)

        # Store mesh node information
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        self._num_points_msh = len(nodeTags)  # Numero totale di punti della mesh
        self._coord_points_msh = nodeCoords

        # Save mesh
        gmsh.write(self.model_name + ".msh")

        # Finalize gmsh
        gmsh.finalize()
    
    def create_single_meshes(self, lc=1e-14, len_msh=.1):
        """
        Create one mesh per object and then merge all meshes into a single file.

        The method:
        - iterates over the acquired objects, creating a mesh file per object,
        - records which meshes were successfully generated,
        - merges the successful meshes into ``<model_name>.msh``,
        - writes errors to ``Errors_file.txt``,
        - stores the total number of mesh nodes and their coordinates after merging.

        :param float lc: Local characteristic length assigned to points.
            Default is ``1e-14``.
        :param float len_msh: Global maximum characteristic length used for meshing.
            Default is ``0.1``.
        :return: None
        :rtype: None
        """
        dim = 3
        num_el = 0
        lst_merging = []
        lst_problems = []
        for el in self.lst_model_dict:
            gmsh.initialize()
            gmsh.model.add(str(num_el).zfill(2) + "_" + self.model_name)

            _ = self._add_points(el, lc)
            lst_tag_curves = self._add_curves(el)
            lst_surfaces = self._add_surfaces(lst_tag_curves)
            idx_volume = self._add_volume(lst_surfaces)
            gmsh.model.occ.synchronize()
            _ = self._add_phy(idx_volume)
            gmsh.model.occ.synchronize()
            # Create mesh
            try:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", len_msh)
                gmsh.model.mesh.generate(dim)
                # Trigger mesh retrieval (used as a basic check)
                _ = gmsh.model.mesh.getNodes()
                _ = gmsh.model.mesh.getElements(dim=3)
                gmsh.write(str(num_el).zfill(2) + "_" + self.model_name + ".msh")
                gmsh.finalize()
                lst_merging.append(num_el)
            except:
                lst_problems.append(
                    f"Error in the mesh {num_el}, object " + el["name"] + "\n"
                )
            num_el+=1
        
        # Merging
        gmsh.initialize()
        gmsh.model.add(self.model_name)
        for val in lst_merging:
            gmsh.merge(str(val).zfill(2) + "_" + self.model_name + ".msh")
            os.remove(str(val).zfill(2) + "_" + self.model_name + ".msh")
            gmsh.model.occ.synchronize()
        
        gmsh.model.mesh.generate(dim)
        # Store mesh node information
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        self._num_points_msh = len(nodeTags)
        self._coord_points_msh = nodeCoords
        gmsh.write(self.model_name + ".msh")
        gmsh.finalize()

        with open("Errors_file.txt", "w") as fp:
            for stringa in lst_problems:
                fp.writelines(stringa)
            fp.close()
    
    @staticmethod
    def _add_points(el, lc):
        """
        Add transformed Blender vertices as Gmsh OCC points.

        The method applies, in order:
        - scaling,
        - rotation (Euler angles X/Y/Z),
        - translation,
        to each vertex in ``el["vertices"]`` and then adds it to the Gmsh model.

        :param dict el: Object dictionary produced by :class:`Blender`, expected
            to contain keys ``location``, ``scale``, ``rotation``, ``vertices``.
        :param float lc: Local characteristic length assigned to each point.
        :return: List of Gmsh point tags.
        :rtype: list[int]
        """
        # Acquire location, scale, and rotation
        tr_x, tr_y, tr_z = copy(el["location"])
        sc_x, sc_y, sc_z = copy(el["scale"])
        rot_x, rot_y, rot_z = copy(el["rotation"])  # Angoli di rotazione (in radianti)

        # Build rotation matrices
        rot_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)]
        ])
        
        rot_matrix_y = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        
        rot_matrix_z = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1]
        ])

        # Total rotation
        rotation_matrix = rot_matrix_z @ rot_matrix_y @ rot_matrix_x

        lst_tags_points = []
        for ptx in el["vertices"]:
            # Scala il vertice
            scaled_ptx = np.array([sc_x * ptx[0], sc_y * ptx[1], sc_z * ptx[2]])
            
            # Ruota il vertice
            rotated_ptx = rotation_matrix @ scaled_ptx
            
            # Trasla il vertice
            final_ptx = rotated_ptx + np.array([tr_x, tr_y, tr_z])
            
            # Aggiungi il punto a GMSH
            lst_tags_points.append(
                gmsh.model.occ.addPoint(
                    final_ptx[0], final_ptx[1], final_ptx[2], meshSize=lc
                )
            )
        return lst_tags_points

    
    @staticmethod
    def _add_curves(el, tot_acquired=0):
        """
        Add triangular edges as Gmsh OCC lines.

        For each face (assumed to be a triangle) in ``el["faces"]``, the method
        creates three lines connecting its three vertex indices. Vertex indices
        are shifted by ``tot_acquired`` to account for previously added points.

        :param dict el: Object dictionary expected to contain key ``faces`` as a
            list of triangular index triplets.
        :param int tot_acquired: Offset added to vertex indices. Default is ``0``.
        :return: List of line-tag triplets (one per face).
        :rtype: list[list[int]]
        """
        lst_cicli = []
        for elm in el["faces"]:
            fc = copy(elm)
            lst_cicli.append(
                [
                    gmsh.model.occ.addLine(fc[0] + 1 + tot_acquired, fc[1] + 1 + tot_acquired),
                    gmsh.model.occ.addLine(fc[1] + 1 + tot_acquired, fc[2] + 1 + tot_acquired),
                    gmsh.model.occ.addLine(fc[2] + 1 + tot_acquired, fc[0] + 1 + tot_acquired),
                ]
            )
        tot_acquired += len(lst_cicli)
        return lst_cicli

    @staticmethod
    def _add_surfaces(lst_tag_curves):
        """
        Create planar surfaces from curve loops.

        For each set of curve tags, the method:
        - creates a curve loop,
        - creates a plane surface bounded by that loop.

        :param list[list[int]] lst_tag_curves: List of curve-tag triplets.
        :return: List of created surface tags.
        :rtype: list[int]
        """
        lst_surfaces = []
        for elm in lst_tag_curves:
            el_loop = gmsh.model.occ.addCurveLoop(copy(elm))
            lst_surfaces.append(
                gmsh.model.occ.addPlaneSurface([el_loop])
            )
        return lst_surfaces
    
    @staticmethod
    def _add_volume(lst_surfaces):
        """
        Create a Gmsh OCC volume from a list of surfaces.

        :param list[int] lst_surfaces: Surface tags used to build a surface loop.
        :return: Volume tag.
        :rtype: int
        """
        vol = gmsh.model.occ.addSurfaceLoop(copy(lst_surfaces))
        return gmsh.model.occ.addVolume([vol])
    
    @staticmethod
    def _add_phy(idx_volume):
        """
        Add a physical group for a volume.

        :param int idx_volume: Volume tag.
        :return: Physical group tag.
        :rtype: int
        """
        dim = 3
        return gmsh.model.addPhysicalGroup(dim, [idx_volume], idx_volume, "Element "+ str(idx_volume))
    
    @staticmethod
    def print_edge_lengths(mesh_file):
        """
        Load a mesh and print minimum and maximum edge lengths for one entity.

        The method:
        - opens a ``.msh`` file in Gmsh,
        - gets 3D entities and selects the maximum entity tag,
        - uses ``getElementQualities`` with ``minEdge`` and ``maxEdge``,
        - prints the minimum and maximum values.

        :param str mesh_file: Path to the mesh file to load.
        :return: None
        :rtype: None
        """
        gmsh.initialize()

        gmsh.open(mesh_file)

        gmsh.model.geo.synchronize()

        entities = gmsh.model.getEntities(dim=3)

        max_tag = max([tag for _, tag in entities])

        min_edge_qualities = gmsh.model.mesh.getElementQualities([max_tag], qualityName="minEdge")
        max_edge_qualities = gmsh.model.mesh.getElementQualities([max_tag], qualityName="maxEdge")

        if min_edge_qualities and max_edge_qualities:
            min_edge_length = min(min_edge_qualities)
            max_edge_length = max(max_edge_qualities)
            print(f"Min length of edges: {min_edge_length}")
            print(f"Max length of edges: {max_edge_length}")
        else:
            print("Impossible obtaining length of edges.")

        gmsh.finalize()
    
    @staticmethod
    def refine(mesh_file, num_refine):
        """
        Refine an existing mesh file in-place.

        If ``num_refine`` is not zero, the method:
        - opens the mesh file,
        - refines the mesh ``num_refine`` times,
        - writes the refined mesh back to the same file.

        :param str mesh_file: Path to the mesh file to refine.
        :param int num_refine: Number of refinement iterations.
        :return: None
        :rtype: None
        """
        if num_refine != 0:
            gmsh.initialize()

            gmsh.open(mesh_file)

            entities = gmsh.model.getEntities()

            # Store original topology (not used further in the provided code)
            element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()

            gmsh.model.mesh.setOrder(2)
            for _ in range(num_refine):
                gmsh.model.mesh.refine()

            max_tag = max(entities[0])
            print(gmsh.model.mesh.getElementQualities([max_tag], qualityName="minEdge"))
            print(gmsh.model.mesh.getElementQualities([max_tag], qualityName="maxEdge"))

            new_element_types, new_element_tags, new_element_nodes = gmsh.model.mesh.getElements()

            gmsh.write(mesh_file)

            gmsh.finalize()
    
    @staticmethod
    def export_vertices(mesh_file, filename = "mesh_points", sep=";", num = None, random_state = 42):
        """
        Export mesh node coordinates to a CSV file.

        The method:
        - opens the mesh file,
        - retrieves node coordinates from Gmsh,
        - builds a DataFrame with columns ``x, y, z``,
        - optionally samples a subset of rows,
        - writes the result to ``<filename>.csv``.

        :param str mesh_file: Path to the mesh file to load.
        :param str filename: Output file name without extension. Default is ``"mesh_points"``.
        :param str sep: CSV separator. Default is ``";"``.
        :param int | None num: If provided and valid, number of random nodes to export.
            If ``None`` all nodes are exported. Default is ``None``.
        :param int random_state: Random seed used when sampling rows. Default is ``42``.
        :return: None
        :rtype: None
        """
        gmsh.initialize()

        gmsh.open(mesh_file)

        _, node_coords, _ = gmsh.model.mesh.getNodes()

        x_coords = node_coords[::3]
        y_coords = node_coords[1::3]
        z_coords = node_coords[2::3]

        df = pd.DataFrame(
            {
                "x": x_coords,
                "y": y_coords,
                "z": z_coords
            }
        )

        gmsh.finalize()
        if num == None or num > len(df) or num <= 0:
            df.to_csv(filename + ".csv", sep, index=False)
        else:
            df = df.sample(n=num, random_state=random_state)
            df.to_csv(filename + ".csv", sep, index=False)
    

    @property
    def num_points_msh(self):
        """
        Return the number of mesh nodes produced by the last meshing call.

        :raises AttributeError: If the mesh has not been created yet.
        :return: Number of mesh nodes.
        :rtype: int
        """
        if self._num_points_msh is None:
            raise AttributeError("Attribute not defined. Call method - create mesh.")
        return self._num_points_msh
    
    @property
    def coord_points_msh(self):
        """
        Return the flattened mesh node coordinates produced by the last meshing call.

        The returned array is the raw ``nodeCoords`` output of
        :func:`gmsh.model.mesh.getNodes`.

        :raises AttributeError: If the mesh has not been created yet.
        :return: Mesh node coordinates.
        :rtype: list[float] | numpy.ndarray
        """
        if self._coord_points_msh is None:
            raise AttributeError("Attribute not defined. Call method - create mesh.")
        return self._coord_points_msh