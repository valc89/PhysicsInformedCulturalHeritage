import os
import gmsh
import numpy as np
import pandas as pd

from copy import copy
from modelaquisition.acquisition.blender import Blender

class Blend2Mesh(Blender):

    def __init__(self, filename, model_name = "Model"):
        super().__init__(filename)
        self.model_name = model_name
        self.lst_model_dict = self.acquire_model()
        self._num_points_msh = None
        self._coord_points_msh = None

    def create_mesh(self, lc=1e-14, len_msh=.1):
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

        # Acquisizione numero di punti della mesh
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        self._num_points_msh = len(nodeTags)  # Numero totale di punti della mesh
        self._coord_points_msh = nodeCoords

        # Saving mesh
        gmsh.write(self.model_name + ".msh")

        # Finalize gmsh
        gmsh.finalize()
    
    def create_single_meshes(self, lc=1e-14, len_msh=.1):
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
                # Ottieni i nodi ed elementi per verificare la mesh
                nodes = gmsh.model.mesh.getNodes()
                elements = gmsh.model.mesh.getElements(dim=3)
                gmsh.write(str(num_el).zfill(2) + "_" + self.model_name + ".msh")
                gmsh.finalize()
                lst_merging.append(num_el)
            except:
                lst_problems.append(
                    f"Errore nella mesh {num_el}, object " + el["name"] + "\n"
                )
            num_el+=1
        
        # Merging
        gmsh.initialize()
        gmsh.model.add(self.model_name)
        for val in lst_merging:
            gmsh.merge(str(val).zfill(2) + "_" + self.model_name + ".msh")
            os.remove(str(val).zfill(2) + "_" + self.model_name + ".msh")
            gmsh.model.occ.synchronize()

        # Sincronizza il modello per aggiornare le entità
        gmsh.model.mesh.generate(dim)
        # Acquisizione numero di punti della mesh
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        self._num_points_msh = len(nodeTags)  # Numero totale di punti della mesh
        self._coord_points_msh = nodeCoords
        gmsh.write(self.model_name + ".msh")
        gmsh.finalize()

        with open("Errors_file.txt", "w") as fp:
            for stringa in lst_problems:
                fp.writelines(stringa)
            fp.close()
    
    @staticmethod
    def _add_points(el, lc):
        # Acquire location, scale, and rotation
        tr_x, tr_y, tr_z = copy(el["location"])
        sc_x, sc_y, sc_z = copy(el["scale"])
        rot_x, rot_y, rot_z = np.radians(copy(el["rotation"]))  # Angoli di rotazione (in radianti)

        # Matrici di rotazione attorno agli assi X, Y e Z
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x), 0],
            [0, np.sin(rot_x), np.cos(rot_x), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y), 0],
            [0, 1, 0, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0, 0],
            [np.sin(rot_z), np.cos(rot_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Matrice di trasformazione totale (T * Rz * Ry * Rx)
        R = Rz @ Ry @ Rx

        lst_tags_points = []
        for ptx in el["vertices"]:
            # Scala il vertice
            p = np.array([ptx[0], ptx[1], ptx[2], 1])
            
            # Ruota il vertice
            final_ptx = R @ p
            
            # Aggiungi il punto a GMSH
            lst_tags_points.append(
                gmsh.model.occ.addPoint(
                    final_ptx[0] + tr_x + sc_x,
                    final_ptx[1] + tr_x + sc_x,
                    final_ptx[2] + tr_x + sc_x,
                    meshSize=lc
                )
            )
        return lst_tags_points

    
    @staticmethod
    def _add_curves(el, tot_acquired=0):
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
        lst_surfaces = []
        for elm in lst_tag_curves:
            el_loop = gmsh.model.occ.addCurveLoop(copy(elm))
            lst_surfaces.append(
                gmsh.model.occ.addPlaneSurface([el_loop])
            )
        return lst_surfaces
    
    @staticmethod
    def _add_volume(lst_surfaces):
        vol = gmsh.model.occ.addSurfaceLoop(copy(lst_surfaces))
        return gmsh.model.occ.addVolume([vol])
    
    @staticmethod
    def _add_phy(idx_volume):
        dim = 3
        return gmsh.model.addPhysicalGroup(dim, [idx_volume], idx_volume, "Element "+ str(idx_volume))
    
    @staticmethod
    def print_edge_lengths(mesh_file):
        """
        Funzione per caricare una mesh da un file e stampare la lunghezza minima e massima degli edge
        utilizzando la funzione getElementQualities di Gmsh.
        
        Args:
            mesh_file (str): Il percorso del file della mesh da caricare.
        """
        # Inizializza Gmsh
        gmsh.initialize()

        # Carica la mesh da file
        gmsh.open(mesh_file)

        # Sincronizza la geometria
        gmsh.model.geo.synchronize()

        # Ottieni tutte le entità della mesh
        entities = gmsh.model.getEntities(dim=3)

        # Trova il massimo tag di entità
        max_tag = max([tag for _, tag in entities])

        # Ottieni la qualità degli edge minimi e massimi per l'entità con il tag massimo
        min_edge_qualities = gmsh.model.mesh.getElementQualities([max_tag], qualityName="minEdge")
        max_edge_qualities = gmsh.model.mesh.getElementQualities([max_tag], qualityName="maxEdge")

        # Stampa la lunghezza minima e massima degli edge
        if min_edge_qualities and max_edge_qualities:
            min_edge_length = min(min_edge_qualities)
            max_edge_length = max(max_edge_qualities)
            print(f"Lunghezza minima degli edge: {min_edge_length}")
            print(f"Lunghezza massima degli edge: {max_edge_length}")
        else:
            print("Impossibile ottenere le lunghezze degli edge.")

        # Finalizza Gmsh
        gmsh.finalize()
    
    @staticmethod
    def refine(mesh_file, num_refine):
        if num_refine != 0:
            # Inizializza Gmsh
            gmsh.initialize()

            # Carica la mesh dal file .msh
            gmsh.open(mesh_file)

            # Ottieni le entità della mesh
            entities = gmsh.model.getEntities()

            # Salva la topologia originale (connettività degli elementi)
            element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()

            # Raffina la mesh
            gmsh.model.mesh.setOrder(2)  # Imposta l'ordine di meshing a 2 (opzionale)
            for _ in range(num_refine):
                gmsh.model.mesh.refine()

            # Verifica le qualità degli elementi (ad esempio minEdge, maxEdge)
            max_tag = max(entities[0])
            print(gmsh.model.mesh.getElementQualities([max_tag], qualityName="minEdge"))
            print(gmsh.model.mesh.getElementQualities([max_tag], qualityName="maxEdge"))

            # Ottieni nuovamente gli elementi dopo il raffinamento
            new_element_types, new_element_tags, new_element_nodes = gmsh.model.mesh.getElements()

            # Puoi fare il confronto tra `element_nodes` e `new_element_nodes` per preservare l'ordine
            # oppure gestire manualmente il mapping se l'ordine viene modificato.

            # Salva la mesh raffinata
            gmsh.write(mesh_file)

            # Pulisci le risorse di Gmsh
            gmsh.finalize()
    
    @staticmethod
    def export_vertices(mesh_file, filename = "mesh_points", sep=";", num = None, random_state = 42):
        # Inizializzazione di gmsh
        gmsh.initialize()

        # Caricamento del file .msh (sostituire con il percorso del proprio file)
        gmsh.open(mesh_file)

        # Recuperare i nodi (vertici) della mesh
        _, node_coords, _ = gmsh.model.mesh.getNodes()

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
    

    @property
    def num_points_msh(self):
        if self._num_points_msh is None:
            raise AttributeError("Attribute not defined. Call method - create mesh.")
        return self._num_points_msh
    
    @property
    def coord_points_msh(self):
        if self._coord_points_msh is None:
            raise AttributeError("Attribute not defined. Call method - create mesh.")
        return self._coord_points_msh