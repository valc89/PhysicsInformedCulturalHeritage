"""Gmsh .msh to XDMF/HDF5 conversion module."""
import os
import gmsh
import h5py
import meshio
import numpy as np
import pandas as pd

class Msh2Xdmf(object):
    """
    Utility class to convert a Gmsh ``.msh`` file into XDMF/HDF5 files.

    The class uses Gmsh to load a mesh, extracts node coordinates and tetrahedral
    connectivity, stores them in an HDF5 file, and creates an XDMF file pointing
    to the HDF5 datasets. It also provides utilities to:
    - refine a mesh with Gmsh,
    - export node coordinates to CSV,
    - create additional XDMF/HDF5 files including static or time-dependent
      solution datasets.
    """

    def __init__(self, filename, model_name = "model"):
        """
        Initialize the :class:`Msh2Xdmf` class.

        :param str filename: Path to the input ``.msh`` file.
        :param str model_name: Base name used for output files
            (e.g., ``<model_name>.h5`` and ``<model_name>.xdmf``).
            Default is ``"model"``.
        """
        self.filename = filename
        self.model_name = model_name
        self._num_mesh_points = None
        self._coord_mesh_points = None
    
    def _initialize_gmsh(self):
        """
        Initialize the Gmsh API.

        :return: None
        :rtype: None
        """
        gmsh.initialize()
    
    def _load_mesh(self, dim=3):
        """
        Load the mesh in Gmsh and store node information.

        The method:
        - opens ``self.filename`` with Gmsh,
        - retrieves mesh nodes and stores their count and coordinates,
        - generates the mesh for the specified dimension.

        :param int dim: Mesh dimension used in ``gmsh.model.mesh.generate``.
            Default is ``3``.
        :return: None
        :rtype: None
        """
        gmsh.open(self.filename)
        nodeTags, coord, _ = gmsh.model.mesh.getNodes()
        self._num_mesh_points = len(nodeTags)
        self._coord_mesh_points = coord
        gmsh.model.mesh.generate(dim)

    def _extract_mesh_data(self, dim=3, index_element=4):
        """
        Extract points and tetrahedral connectivity from the loaded mesh.

        The method:
        - reads nodes and reshapes coordinates to ``(N, dim)``,
        - reads mesh elements and selects the element block whose type matches
          ``index_element``,
        - reshapes the tetrahedral connectivity to ``(M, 4)`` and converts it
          from 1-based to 0-based indexing.

        :param int dim: Spatial dimension used to reshape node coordinates.
            Default is ``3``.
        :param int index_element: Gmsh element type code to select. Default is
            ``4`` (tetrahedra in Gmsh).
        :return: Tuple ``(points, tetra_elements)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = node_coords.reshape(-1, dim)

        element_types, _, element_nodes = gmsh.model.mesh.getElements()
        tetra_index = np.where(element_types == index_element)[0][0]
        tetra_elements = element_nodes[tetra_index].reshape(-1, 4)
        
        tetra_elements -= 1
        
        return points, tetra_elements

    def _refine_mesh(self, num_refine):
        """
        Refine the current mesh using Gmsh.

        After refinement, the method updates the stored number of nodes and
        node coordinates.

        :param int num_refine: Number of refinement iterations.
        :return: None
        :rtype: None
        """
        if num_refine > 0:
            for _ in range(num_refine):
                gmsh.model.mesh.refine()
            nodeTags, coord, _ = gmsh.model.mesh.getNodes()
            self._num_mesh_points = len(nodeTags)
            self._coord_mesh_points = coord

    def _extract_physical_groups(self):
        """
        Extract physical groups and their entity tags from the loaded mesh.

        For each physical group, the method retrieves its name. If the name is
        empty, a default name is assigned. The returned dictionary maps group
        names to entity tags.

        :return: Dictionary mapping physical group names to entity tags.
        :rtype: dict[str, list[int]]
        """
        physical_groups = gmsh.model.getPhysicalGroups()
        physical_entities = {}

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            if not name:
                name = f"unnamed_group_dim{dim}_tag{tag}"
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            physical_entities[name] = entity_tags

        return physical_entities

    def _save_hdf5(self, points, tetra_elements, physical_groups):
        """
        Save mesh data and physical groups into an HDF5 file.

        The method creates:
        - ``/Mesh/mesh/geometry`` containing node coordinates,
        - ``/Mesh/mesh/topology`` containing tetra connectivity,
        - ``/Mesh/physical_groups`` containing datasets for physical groups.

        :param numpy.ndarray points: Node coordinates array.
        :param numpy.ndarray tetra_elements: Tetrahedral connectivity array.
        :param dict physical_groups: Physical groups mapping (name -> entities).
        :return: None
        :rtype: None
        """
        with h5py.File(self.model_name + ".h5", "w") as f:
            mesh_group = f.create_group("/Mesh/mesh")
            mesh_group.create_dataset("geometry", data=points)
            mesh_group.create_dataset("topology", data=tetra_elements)
            physical_group_grp = f.create_group("/Mesh/physical_groups")
            for name, entities in physical_groups.items():
                physical_group_grp.create_dataset(name.encode('utf-8'), data=np.array(entities))

    def _create_xdmf_content(self, points, tetra_elements, dim=3):
        """
        Create the XDMF content referencing the generated HDF5 datasets.

        The generated XDMF includes a single Uniform Grid with:
        - a tetrahedral topology referencing ``/Mesh/mesh/topology``,
        - a geometry referencing ``/Mesh/mesh/geometry``.

        :param numpy.ndarray points: Node coordinates array.
        :param numpy.ndarray tetra_elements: Tetrahedral connectivity array.
        :param int dim: Spatial dimension written in the geometry DataItem.
            Default is ``3``.
        :return: None
        :rtype: None
        """
        self.xdmf_content = f"""<?xml version="1.0" ?>
        <Xdmf Version="3.0">
          <Domain>
            <Grid Name="mesh" GridType="Uniform">
              <Topology TopologyType="Tetrahedron" NumberOfElements="{tetra_elements.shape[0]}">
                <DataItem Format="HDF" Dimensions="{tetra_elements.shape[0]} 4" Name="partition:0">
                  {self.model_name}.h5:/Mesh/mesh/topology
                </DataItem>
              </Topology>
              <Geometry GeometryType="XYZ">
                <DataItem Format="HDF" Dimensions="{points.shape[0]} {dim}" Name="partition:0">
                  {self.model_name}.h5:/Mesh/mesh/geometry
                </DataItem>
              </Geometry>
            </Grid>
          </Domain>
        </Xdmf>
        """

    def _save_xdmf(self):
        """
        Save the generated XDMF content to disk.

        :return: None
        :rtype: None
        """
        with open(self.model_name + ".xdmf", "w") as f:
            f.write(self.xdmf_content)
    
    def _save_msh_file(self, refined_mesh_file):
        """
        Save the current Gmsh mesh to a ``.msh`` file.

        :param str refined_mesh_file: Output ``.msh`` file path.
        :return: None
        :rtype: None
        """
        gmsh.write(refined_mesh_file)

    def _finalize_gmsh(self):
        """
        Finalize the Gmsh API session.

        :return: None
        :rtype: None
        """
        gmsh.finalize()

    def to_xdmf(self, dim=3, index_element=4, num_refine=0):
        """
        Run the full conversion pipeline from ``.msh`` to XDMF/HDF5.

        The method:
        - initializes Gmsh and loads the mesh,
        - optionally refines the mesh and overwrites the ``.msh`` file,
        - extracts points and tetra connectivity,
        - extracts physical groups,
        - writes the HDF5 and XDMF output files,
        - finalizes Gmsh.

        :param int dim: Spatial dimension. Default is ``3``.
        :param int index_element: Gmsh element type code. Default is ``4``.
        :param int num_refine: Number of refinement iterations. Default is ``0``.
        :return: None
        :rtype: None
        """
        self._initialize_gmsh()
        self._load_mesh(dim)
        if num_refine > 0:
            self._refine_mesh(num_refine)
            self._save_msh_file(self.filename)
        points, tetra_elements = self._extract_mesh_data(dim, index_element)
        physical_groups = self._extract_physical_groups()
        self._save_hdf5(points, tetra_elements, physical_groups)
        self._create_xdmf_content(points, tetra_elements, dim)
        self._save_xdmf()
        self._finalize_gmsh()
    
    def export_vertices(self, mesh_file, filename = "mesh_points", sep=";", num = None, random_state = 42):
        """
        Export mesh node coordinates to a CSV file.

        The method loads a ``.msh`` file with Gmsh, retrieves the nodes, builds a
        DataFrame with columns ``x, y, z``, optionally samples ``num`` rows, and
        writes the output to ``<filename>.csv``.

        :param str mesh_file: Path to the input mesh file.
        :param str filename: Output file name without extension.
            Default is ``"mesh_points"``.
        :param str sep: CSV separator. Default is ``";"``.
        :param int | None num: If provided and valid, number of random nodes to export.
            If ``None`` all nodes are exported. Default is ``None``.
        :param int random_state: Random seed used when sampling. Default is ``42``.
        :return: None
        :rtype: None
        """
        gmsh.initialize()
        gmsh.open(mesh_file)
        nodeTags, node_coords, _ = gmsh.model.mesh.getNodes()
        self._num_mesh_points = len(nodeTags)
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
    
    def _check_solution_consistency(self, solution, num_nodes):
        """
        Check that a solution array matches the number of mesh nodes.

        :param numpy.ndarray solution: Solution array to validate.
        :param int num_nodes: Expected number of nodes.
        :raises ValueError: If the solution length does not match ``num_nodes``.
        :return: ``True`` if consistent.
        :rtype: bool
        """
        if solution.shape[0] != num_nodes:
            raise ValueError(f"Il numero di nodi nella soluzione ({solution.shape[0]}) non corrisponde al numero di nodi nella mesh ({num_nodes}).")
        return True
    
    def add_solution(self, solution_steps, add_info, solution_name="solution", steps=None):
        """
        Add a static or time-dependent solution and generate new XDMF/HDF5 files.

        If ``steps`` is ``None``, the method treats the input as a static solution.
        Otherwise it creates a time series solution where each step is stored
        under a different HDF5 group.

        :param numpy.ndarray | list solution_steps: Solution data. The expected
            shape depends on whether the solution is static or time-dependent.
        :param str add_info: String appended to output file names.
        :param str solution_name: Dataset name used to store the solution.
            Default is ``"solution"``.
        :param list | int | None steps: Time steps. If ``int``, it is converted
            to ``range(steps)``. If ``None``, the solution is static.
        :raises ValueError: If the number of provided steps does not match the
            number of solutions.
        :return: None
        :rtype: None
        """
        if steps is None:
            if solution_steps.ndim == 1:
                print("Addition of static solution")
                self._add_solution_static(solution_steps, add_info, "Scalar", solution_steps.ndim, solution_name)
            else:
                self._add_solution_static(solution_steps, add_info, "Vector", solution_steps.ndim, solution_name)
        else:
            if isinstance(steps, int):
                steps = range(steps)
            if not isinstance(solution_steps, list):
                if solution_steps.shape[0] == len(steps):
                    print("Addition of time-dependent solution")
                    self._add_solution_time(solution_steps, add_info, "Scalar", 1, solution_name, steps)
                else:
                    raise ValueError("Number of time steps not coherent with number of solutions.")
            else:
                if solution_steps[0].shape[1] == len(steps):
                    print("Addition of vectorial time-dependent solution.")
                    self._add_solution_time(solution_steps, add_info, "Vector", len(solution_steps), solution_name, steps)
                else:
                    raise ValueError("Number of time steps not coherent with number of solutions.")

    
    def _add_solution_static(self, solution_step, add_info, attribute, leng_input, solution_name="solution"):
        """
        Add a static solution to a mesh and generate a new XDMF/HDF5 pair.

        The method reads the base mesh from ``<model_name>.xdmf``, writes a new
        HDF5 file containing the mesh and the solution, and generates a new XDMF
        file referencing the stored datasets.

        :param numpy.ndarray solution_step: Static solution array.
        :param str add_info: String appended to output file names.
        :param str attribute: XDMF AttributeType (e.g., ``"Scalar"``, ``"Vector"``).
        :param int leng_input: Number of components per node written in XDMF.
        :param str solution_name: Dataset name used to store the solution.
            Default is ``"solution"``.
        :return: None
        :rtype: None
        """
        mesh = meshio.read(self.model_name + ".xdmf")
        points = mesh.points
        tetra_elements = mesh.cells_dict["tetra"]
        h5_filepath = self.model_name + add_info + ".h5"
        with h5py.File(h5_filepath, "w") as f:
            mesh_group = f.create_group("/Mesh/mesh")
            mesh_group.create_dataset("geometry", data=points)
            mesh_group.create_dataset("topology", data=tetra_elements)
            self._check_solution_consistency(solution_step, points.shape[0])
            vis_group = f.create_group("/VisualisationVector")
            vis_group.create_dataset(solution_name, data=solution_step)
        
        self._create_static_xdmf_file_with_template(solution_name, add_info, tetra_elements, points, attribute, leng_input)

    def _create_static_xdmf_file_with_template(self, solution_name, add_info, tetra_elements, points, attribute, leng_input):
        """
        Create an XDMF file including a static solution attribute.

        :param str solution_name: Dataset name used for the solution.
        :param str add_info: String appended to output file names.
        :param numpy.ndarray tetra_elements: Tetrahedral connectivity array.
        :param numpy.ndarray points: Node coordinates array.
        :param str attribute: XDMF AttributeType (e.g., ``"Scalar"``, ``"Vector"``).
        :param int leng_input: Number of components per node written in XDMF.
        :return: None
        :rtype: None
        """
        xdmf_filepath = self.model_name + add_info + ".xdmf"
        h5_filename = self.model_name + add_info + ".h5"

        grid_content = f"""
        <Grid Name="mesh" GridType="Uniform">
            <Topology NumberOfElements="{tetra_elements.shape[0]}" TopologyType="Tetrahedron" NodesPerElement="4">
                <DataItem Dimensions="{tetra_elements.shape[0]} 4" NumberType="UInt" Format="HDF">
                    {h5_filename}:/Mesh/mesh/topology
                </DataItem>
            </Topology>
            <Geometry GeometryType="XYZ">
                <DataItem Dimensions="{points.shape[0]} 3" Format="HDF">
                    {h5_filename}:/Mesh/mesh/geometry
                </DataItem>
            </Geometry>
            <Attribute Name="{solution_name}" AttributeType="{attribute}" Center="Node">
                <DataItem Dimensions="{points.shape[0]} {leng_input}" Format="HDF">
                    {h5_filename}:/VisualisationVector/{solution_name}
                </DataItem>
            </Attribute>
        </Grid>
        """

        xdmf_content = f"""<?xml version="1.0"?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
            <Domain>
                {grid_content}
            </Domain>
        </Xdmf>
        """

        with open(xdmf_filepath, "w") as f:
            f.write(xdmf_content)
    

    def _add_solution_time(self, solution_steps, add_info, attribute, leng_input, solution_name="solution", steps=None):
        """
        Add a time-dependent solution to a mesh and generate a new XDMF/HDF5 pair.

        The method reads the base mesh from ``<model_name>.xdmf`` and writes a new
        HDF5 file where each time step stores the mesh under ``/Mesh/<step>/mesh``
        and the solution under ``/VisualisationVector/<step>/``.

        :param numpy.ndarray | list solution_steps: Solution data (array or list
            of arrays depending on the calling code path).
        :param str add_info: String appended to output file names.
        :param str attribute: XDMF AttributeType (e.g., ``"Scalar"``, ``"Vector"``).
        :param int leng_input: Number of components per node written in XDMF.
        :param str solution_name: Dataset name used to store the solution.
            Default is ``"solution"``.
        :param list | None steps: Time step values.
        :return: None
        :rtype: None
        """
        mesh = meshio.read(self.model_name + ".xdmf")
        points = mesh.points
        tetra_elements = mesh.cells_dict["tetra"]

        h5_filepath = self.model_name + add_info + ".h5"
        with h5py.File(h5_filepath, "w") as f:
            if isinstance(solution_steps, list):
                step = 0.
                for i in range(solution_steps[0].shape[1]):
                    step_group_path = f"/Mesh/{int(step)}/mesh"
                    mesh_group = f.create_group(step_group_path)

                    mesh_group.create_dataset("geometry", data=points)
                    mesh_group.create_dataset("topology", data=tetra_elements)

                    for el in solution_steps:
                        self._check_solution_consistency(el, points.shape[0])

                    vis_group_path = f"/VisualisationVector/{int(step)}"
                    vis_group = f.create_group(vis_group_path)

                    vis_group.create_dataset(f"{solution_name}", data=np.concatenate([el[:, i].reshape(el.shape[0], 1) for el in solution_steps], axis=-1))
                    step += 1
            else:
                for i, (step, step_solution) in enumerate(zip(steps, solution_steps)):
                    step_group_path = f"/Mesh/{int(step)}/mesh"
                    mesh_group = f.create_group(step_group_path)

                    mesh_group.create_dataset("geometry", data=points)
                    mesh_group.create_dataset("topology", data=tetra_elements)

                    self._check_solution_consistency(step_solution, points.shape[0])
                    
                    vis_group_path = f"/VisualisationVector/{int(step)}"
                    vis_group = f.create_group(vis_group_path)

                    dataset_name = f"{solution_name}"
                    vis_group.create_dataset(dataset_name, data=step_solution)

        self._create_xdmf_file_with_template(solution_name, steps, add_info, tetra_elements, points, attribute, leng_input)

    def _create_xdmf_file_with_template(self, solution_name, steps, add_info, tetra_elements, points, attribute, leng_input):
        """
        Create an XDMF time series file including a solution attribute per step.

        :param str solution_name: Dataset name used for the solution.
        :param list steps: Time step values used in the XDMF ``Time`` tags.
        :param str add_info: String appended to output file names.
        :param numpy.ndarray tetra_elements: Tetrahedral connectivity array.
        :param numpy.ndarray points: Node coordinates array.
        :param str attribute: XDMF AttributeType (e.g., ``"Scalar"``, ``"Vector"``).
        :param int leng_input: Number of components per node written in XDMF.
        :return: None
        :rtype: None
        """
        xdmf_filepath = self.model_name + add_info + ".xdmf"
        h5_filename = self.model_name + add_info + ".h5"
        
        grids = ""
        for i, step in enumerate(steps):
            grid_content = f"""
            <Grid Name="mesh" GridType="Uniform">
                <Topology NumberOfElements="{tetra_elements.shape[0]}" TopologyType="Tetrahedron" NodesPerElement="4">
                    <DataItem Dimensions="{tetra_elements.shape[0]} 4" NumberType="UInt" Format="HDF">
                        {h5_filename}:/Mesh/{int(i)}/mesh/topology
                    </DataItem>
                </Topology>
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="{points.shape[0]} 3" Format="HDF">
                        {h5_filename}:/Mesh/{int(i)}/mesh/geometry
                    </DataItem>
                </Geometry>
                <Time Value="{step:.15e}" />
                <Attribute Name="{solution_name}" AttributeType="{attribute}" Center="Node">
                    <DataItem Dimensions="{points.shape[0]} {leng_input}" Format="HDF">
                        {h5_filename}:/VisualisationVector/{i}/{solution_name}
                    </DataItem>
                </Attribute>
            </Grid>
            """
            grids += grid_content

        xdmf_content = f"""<?xml version="1.0"?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
            <Domain>
                <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
                    {grids}
                </Grid>
            </Domain>
        </Xdmf>
        """

        with open(xdmf_filepath, "w") as f:
            f.write(xdmf_content)


    def reset_files(self, add_info):
        """
        Delete XDMF and HDF5 files generated with the given suffix.

        :param str add_info: String appended to the output file names to select
            which files must be removed.
        :return: None
        :rtype: None
        """
        xdmf_filepath = self.model_name + add_info + ".xdmf"
        h5_filepath = self.model_name + add_info + ".h5"

        if os.path.exists(xdmf_filepath):
            os.remove(xdmf_filepath)
            print(f"File {xdmf_filepath} deleted.")
        else:
            print(f"File {xdmf_filepath} not found.")

        if os.path.exists(h5_filepath):
            os.remove(h5_filepath)
            print(f"File {h5_filepath} deleted.")
        else:
            print(f"File {h5_filepath} not found.")
    
    @property
    def num_mesh_points(self):
        """
        Return the number of mesh points stored by the last mesh loading/refinement.

        :raises AttributeError: If the attribute was not set yet.
        :return: Number of mesh nodes.
        :rtype: int
        """
        if self._num_mesh_points is None:
            raise AttributeError("Attribute not defined.")
        return self._num_mesh_points

    @property
    def coord_mesh_points(self):
        """
        Return the flattened mesh node coordinates stored by the last mesh loading/refinement.

        :raises AttributeError: If the attribute was not set yet.
        :return: Mesh node coordinates.
        :rtype: list[float] | numpy.ndarray
        """
        if self._coord_mesh_points is None:
            raise AttributeError("Attribute not defined.")
        return self._coord_mesh_points
