import os
import gmsh
import h5py
import meshio
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

class Msh2Xdmf(object):

    def __init__(self, filename, model_name = "Model"):
        self.filename = filename
        self.model_name = model_name
        self._num_mesh_points = None
        self._coord_mesh_points = None
    
    def _initialize_gmsh(self):
        # Inizializza gmsh
        gmsh.initialize()
    
    def _load_mesh(self, dim=3):
        # Carica il file .msh in gmsh
        gmsh.open(self.filename)
        nodeTags, coord, _ = gmsh.model.mesh.getNodes()
        self._num_mesh_points = len(nodeTags)
        self._coord_mesh_points = coord
        # Sincronizza il modello in modo che tutte le modifiche siano visibili
        gmsh.model.mesh.generate(dim)  # 3 indica mesh 3D

    def _extract_mesh_data(self, dim=3, index_element=4):
        # Recupera i nodi (vertici) e le celle (elementi) dalla mesh di gmsh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # Estrarre le coordinate dei nodi e organizzarle in un array (x, y, z)
        points = node_coords.reshape(-1, dim)

        # Recupera le connettività delle celle (elementi) di tipo tetraedrico (4 nodi per cella)
        element_types, _, element_nodes = gmsh.model.mesh.getElements()

        # Trova l'indice degli elementi tetraedrici (che in GMSH è il tipo 4 per tetraedri)
        tetra_index = np.where(element_types == index_element)[0][0]  # 4 è il codice GMSH per tetraedro

        # Estrai i nodi degli elementi tetraedrici e riorganizza in array di (4 nodi per elemento)
        tetra_elements = element_nodes[tetra_index].reshape(-1, 4)
        
        # Se necessario, sottrai 1 per convertire da 1-based a 0-based (usato in meshio)
        tetra_elements -= 1
        
        return points, tetra_elements

    def _refine_mesh(self, num_refine):
        if num_refine > 0:
            for _ in range(num_refine):
                gmsh.model.mesh.refine()
            nodeTags, coord, _ = gmsh.model.mesh.getNodes()
            self._num_mesh_points = len(nodeTags)
            self._coord_mesh_points = coord

    def _extract_physical_groups(self):
        # Recupera i gruppi fisici definiti nella mesh
        physical_groups = gmsh.model.getPhysicalGroups()
        physical_entities = {}

        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            if not name:  # Se il nome è vuoto, assegna un nome predefinito
                name = f"unnamed_group_dim{dim}_tag{tag}"
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            physical_entities[name] = entity_tags

        return physical_entities

    def _save_hdf5(self, points, tetra_elements, physical_groups):
        # Creare un file HDF5 con la struttura corretta
        with h5py.File(self.model_name + ".h5", "w") as f:
            # Creare il gruppo root -> Mesh -> mesh
            mesh_group = f.create_group("/Mesh/mesh")
            
            # Salvare le coordinate (geometry) e la connettività (topology)
            mesh_group.create_dataset("geometry", data=points)
            mesh_group.create_dataset("topology", data=tetra_elements)
            
            # Salvare i gruppi fisici
            physical_group_grp = f.create_group("/Mesh/physical_groups")
            for name, entities in physical_groups.items():
                # Convertire il nome in bytes
                physical_group_grp.create_dataset(name.encode('utf-8'), data=np.array(entities))

    def _create_xdmf_content(self, points, tetra_elements, dim=3):
        # Creare il contenuto XDMF con la dicitura partition:0, senza includere i gruppi fisici
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
        # Salvare il file XDMF
        with open(self.model_name + ".xdmf", "w") as f:
            f.write(self.xdmf_content)
    
    def _save_msh_file(self, refined_mesh_file):
        gmsh.write(refined_mesh_file)

    def _finalize_gmsh(self):
        # Chiudi gmsh
        gmsh.finalize()

    def to_xdmf(self, dim=3, index_element=4, num_refine=0):
        # Processo completo di conversione
        self._initialize_gmsh()
        self._load_mesh(dim)
        # Raffina la mesh se richiesto
        if num_refine > 0:
            self._refine_mesh(num_refine)
            # Salva il file .msh raffinato, se richiesto
            self._save_msh_file(self.filename)
        points, tetra_elements = self._extract_mesh_data(dim, index_element)
        physical_groups = self._extract_physical_groups()
        self._save_hdf5(points, tetra_elements, physical_groups)
        self._create_xdmf_content(points, tetra_elements, dim)
        self._save_xdmf()
        self._finalize_gmsh()
    
    def export_vertices(self, mesh_file, filename = "mesh_points", sep=";", num = None, random_state = 42):
        # Inizializzazione di gmsh
        gmsh.initialize()

        # Caricamento del file .msh (sostituire con il percorso del proprio file)
        gmsh.open(mesh_file)

        # Recuperare i nodi (vertici) della mesh
        nodeTags, node_coords, _ = gmsh.model.mesh.getNodes()
        self._num_mesh_points = len(nodeTags)

        # Dividere le coordinate in coordinate x, y, z
        x_coords = node_coords[::3]
        y_coords = node_coords[1::3]
        z_coords = node_coords[2::3]

        # Creare un DataFrame con le coordinate
        df = pd.DataFrame(
            {
                "x": x_coords,
                "y": y_coords,
                "z": z_coords
            }
        )

        # Terminare gmsh
        gmsh.finalize()
        if num == None or num > len(df) or num <= 0:
            df.to_csv(filename + ".csv", sep, index=False)
        else:
            df = df.sample(n=num, random_state=random_state)
            df.to_csv(filename + ".csv", sep, index=False)
    
    # Metodo per verificare la coerenza della soluzione con la mesh
    def _check_solution_consistency(self, solution, num_nodes):
        if solution.shape[0] != num_nodes:
            raise ValueError(f"Il numero di nodi nella soluzione ({solution.shape[0]}) non corrisponde al numero di nodi nella mesh ({num_nodes}).")
        return True
    
    def add_solution(self, solution_steps, add_info, solution_name="solution", steps=None):
        """
        Aggiunge una soluzione, temporale o statica, utilizzando la mesh da un file XDMF esistente.
        Se `steps` è fornito, viene considerata una soluzione temporale, altrimenti una soluzione statica.

        Args:
            solution_steps: array numpy con soluzioni (per ciascuno step temporale o una statica).
            add_info: stringa aggiuntiva per il nome del file salvato.
            solution_name: il nome del dataset per la soluzione.
            steps: lista o array di valori temporali che definiscono gli step temporali (opzionale).
        """
        if steps is None:
            # Soluzione statica
            if solution_steps.ndim == 1:
                print("Aggiunta soluzione statica")
                self._add_solution_static(solution_steps, add_info, "Scalar", solution_steps.ndim, solution_name)
            else:
                self._add_solution_static(solution_steps, add_info, "Vector", solution_steps.ndim, solution_name)
        else:
            # Soluzione temporale
            if isinstance(steps, int):
                steps = range(steps)
            if not isinstance(solution_steps, list):
                if solution_steps.shape[0] == len(steps):
                    print("Aggiunta soluzione temporale")
                    self._add_solution_time(solution_steps, add_info, "Scalar", 1, solution_name, steps)
                else:
                    raise ValueError("Numero di steps temporali non corrisponde al numero di soluzioni.")
            else:
                if solution_steps[0].shape[1] == len(steps):
                    print("Aggiunta soluzione temporale e vettoriale")
                    self._add_solution_time(solution_steps, add_info, "Vector", len(solution_steps), solution_name, steps)
                else:
                    raise ValueError("Numero di steps temporali non corrisponde al numero di soluzioni.")

    
    def _add_solution_static(self, solution_step, add_info, attribute, leng_input, solution_name="solution"):
        """
        Aggiunge una soluzione statica utilizzando la mesh da un file XDMF esistente,
        e aggiorna i file HDF5 e XDMF. Il nome del file salvato include `add_info`.

        Args:
            solution_step: array numpy con la soluzione statica (unica).
            add_info: stringa aggiuntiva per il nome del file salvato.
            solution_name: il nome del dataset per la soluzione.
        """
        # Carica la mesh dal file XDMF esistente
        mesh = meshio.read(self.model_name + ".xdmf")

        # Leggere la geometria e la topologia dalla mesh caricata
        points = mesh.points
        tetra_elements = mesh.cells_dict["tetra"]

        # Creare il file HDF5 con la mesh e la soluzione statica
        h5_filepath = self.model_name + add_info + ".h5"
        with h5py.File(h5_filepath, "w") as f:
            # Creare il gruppo /Mesh/mesh per la mesh statica
            mesh_group = f.create_group("/Mesh/mesh")

            # Salvare la geometria (coordinate dei nodi) e la topologia
            mesh_group.create_dataset("geometry", data=points)
            mesh_group.create_dataset("topology", data=tetra_elements)

            # Verifica che la soluzione abbia la dimensione corretta (stessa dei nodi)
            self._check_solution_consistency(solution_step, points.shape[0])
            
            # Creare il gruppo VisualisationVector per la soluzione statica
            vis_group = f.create_group("/VisualisationVector")

            # Salvare la soluzione
            vis_group.create_dataset(solution_name, data=solution_step)

        # Aggiorna il file XDMF con la mesh e la soluzione statica
        self._create_static_xdmf_file_with_template(solution_name, add_info, tetra_elements, points, attribute, leng_input)

    def _create_static_xdmf_file_with_template(self, solution_name, add_info, tetra_elements, points, attribute, leng_input):
        """
        Crea o aggiorna un file XDMF con la struttura fornita per includere la mesh e la soluzione statica.
        Il nome del file XDMF include `add_info`.

        Args:
            solution_name: nome del dataset della soluzione.
            add_info: stringa aggiuntiva per il nome del file salvato.
            tetra_elements: la connettività della mesh (topologia).
            points: le coordinate dei nodi della mesh (geometria).
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

        # Struttura finale del file XDMF
        xdmf_content = f"""<?xml version="1.0"?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
            <Domain>
                {grid_content}
            </Domain>
        </Xdmf>
        """

        # Salvare il contenuto nel file XDMF
        with open(xdmf_filepath, "w") as f:
            f.write(xdmf_content)
    

    def _add_solution_time(self, solution_steps, add_info, attribute, leng_input, solution_name="solution", steps=None):
        """
        Aggiunge una soluzione per ciascuno step temporale utilizzando la mesh da un file XDMF esistente,
        e aggiorna i file HDF5 e XDMF. Il nome del file salvato include `add_info`.
        
        Args:
            xdmf_input_file: file XDMF esistente contenente solo la mesh.
            solution_steps: un array numpy con soluzioni per ogni step temporale.
            add_info: stringa aggiuntiva per il nome del file salvato.
            solution_name: il nome del dataset per la soluzione.
            steps: lista o array di valori temporali che definiscono gli step.
        """
        # Carica la mesh dal file XDMF esistente
        mesh = meshio.read(self.model_name + ".xdmf")

        # Leggere la geometria e la topologia dalla mesh caricata
        points = mesh.points
        tetra_elements = mesh.cells_dict["tetra"]

        # Creare il file HDF5 con la mesh e le soluzioni
        h5_filepath = self.model_name + add_info + ".h5"
        with h5py.File(h5_filepath, "w") as f:
            if isinstance(solution_steps, list):
                step = 0.
                for i in range(solution_steps[0].shape[1]):
                    # Creare il gruppo /Mesh/{step}/mesh per ogni step
                    step_group_path = f"/Mesh/{int(step)}/mesh"
                    mesh_group = f.create_group(step_group_path)

                    # Salvare la geometria (coordinate dei nodi) e la topologia per ogni step
                    mesh_group.create_dataset("geometry", data=points)
                    mesh_group.create_dataset("topology", data=tetra_elements)

                    # Verifica che ogni matrice di soluzione abbia la dimensione corretta
                    for el in solution_steps:
                        self._check_solution_consistency(el, points.shape[0])

                    # Creare il gruppo VisualisationVector per ogni soluzione in questo step
                    vis_group_path = f"/VisualisationVector/{int(step)}"
                    vis_group = f.create_group(vis_group_path)

                    # Salvare la singola matrice di soluzione per questo step
                    vis_group.create_dataset(f"{solution_name}", data=np.concatenate([el[:, i].reshape(el.shape[0], 1) for el in solution_steps], axis=-1))
                    step += 1
            else:
                for i, (step, step_solution) in enumerate(zip(steps, solution_steps)):
                    # Creare il gruppo /Mesh/{step}/mesh per ogni step
                    step_group_path = f"/Mesh/{int(step)}/mesh"
                    mesh_group = f.create_group(step_group_path)

                    # Salvare la geometria (coordinate dei nodi) e la topologia per ogni step
                    mesh_group.create_dataset("geometry", data=points)
                    mesh_group.create_dataset("topology", data=tetra_elements)

                    # Verifica che la soluzione abbia la dimensione corretta (stessa dei nodi)
                    self._check_solution_consistency(step_solution, points.shape[0])
                    
                    # Creare il gruppo VisualisationVector per ogni step
                    vis_group_path = f"/VisualisationVector/{int(step)}"
                    vis_group = f.create_group(vis_group_path)

                    # Salvare la soluzione per questo step
                    dataset_name = f"{solution_name}"
                    vis_group.create_dataset(dataset_name, data=step_solution)

        # Aggiorna il file XDMF con la mesh e le soluzioni per ciascun step
        self._create_xdmf_file_with_template(solution_name, steps, add_info, tetra_elements, points, attribute, leng_input)

    def _create_xdmf_file_with_template(self, solution_name, steps, add_info, tetra_elements, points, attribute, leng_input):
        """
        Crea o aggiorna un file XDMF con la struttura fornita per includere la mesh e le soluzioni per ciascun step temporale.
        Il nome del file XDMF include `add_info`.

        Args:
            solution_name: nome del dataset della soluzione.
            steps: lista o array di valori temporali per gli step.
            add_info: stringa aggiuntiva per il nome del file salvato.
            tetra_elements: la connettività della mesh (topologia).
            points: le coordinate dei nodi della mesh (geometria).
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

        # Struttura finale del file XDMF con il tag "Collection" per gli steps temporali
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

        # Salvare il contenuto nel file XDMF
        with open(xdmf_filepath, "w") as f:
            f.write(xdmf_content)


    def reset_files(self, add_info):
        """
        Cancella i file XDMF e HDF5 associati al modello, utilizzando la stringa aggiuntiva `add_info`.

        Args:
            add_info: stringa aggiuntiva per il nome del file da cancellare.
        """
        # Costruire i percorsi dei file da cancellare
        xdmf_filepath = self.model_name + add_info + ".xdmf"
        h5_filepath = self.model_name + add_info + ".h5"

        # Cancellare il file XDMF se esiste
        if os.path.exists(xdmf_filepath):
            os.remove(xdmf_filepath)
            print(f"File {xdmf_filepath} cancellato.")
        else:
            print(f"File {xdmf_filepath} non trovato.")

        # Cancellare il file HDF5 se esiste
        if os.path.exists(h5_filepath):
            os.remove(h5_filepath)
            print(f"File {h5_filepath} cancellato.")
        else:
            print(f"File {h5_filepath} non trovato.")
    
    @property
    def num_mesh_points(self):
        if self._num_mesh_points is None:
            raise AttributeError("Attribute not defined.")
        return self._num_mesh_points

    @property
    def coord_mesh_points(self):
        if self._coord_mesh_points is None:
            raise AttributeError("Atribute not defined.")
        return self._coord_mesh_points
