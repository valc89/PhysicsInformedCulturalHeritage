import h5py
import numpy as np
import pandas as pd

class AnalyzeXdmf:
    def __init__(self, h5_file):
        """
        Classe per analizzare file XDMF/HDF5 contenenti soluzioni numeriche su una mesh.

        Parametri:
        h5_file (str): Percorso al file HDF5 contenente i dati della simulazione.
        """
        self.h5_file = h5_file
        self.num_nodes = 0
        self.num_variables = 0
        self.solution = None
        self.time_steps = 0

        self._load_data()

    def _load_data(self):
        """Carica i dati dal file HDF5 e memorizza la soluzione."""
        with h5py.File(self.h5_file, 'r') as f:
            # Determinare il numero di step temporali
            visualization_group = f["VisualisationVector"]
            time_steps = [int(k) for k in visualization_group.keys()]
            self.time_steps = len(time_steps)

            # Determinare il numero di nodi dalla geometria del primo step
            first_mesh = f["Mesh/0/mesh/geometry"]
            self.num_nodes = first_mesh.shape[0]

            # Determinare il numero di variabili (assumiamo che ogni step abbia lo stesso formato)
            first_step_data = visualization_group[str(time_steps[0])][()].astype(np.float64)
            self.num_variables = 1 if first_step_data.ndim == 1 else first_step_data.shape[1]

            # Alloca un array per la soluzione (time_steps, num_nodes)
            self.solution = np.zeros((self.time_steps, self.num_nodes), dtype=np.float64)

            # Legge i dati nei vari step temporali
            for i, step in enumerate(time_steps):
                step_data = visualization_group[str(step)][()]
                self.solution[i, :] = np.reshape(step_data, (1, step_data.shape[0]))  # Supponiamo che contenga una sola variabile

    def get_summary(self):
        """Restituisce un riepilogo dei dati estratti."""
        return {
            "Numero di nodi": self.num_nodes,
            "Numero di variabili": self.num_variables,
            "Numero di step temporali": self.time_steps,
            "Forma della soluzione": self.solution.shape
        }
    
    def export_solution(self, filename):
        df = pd.DataFrame(self.solution)
        df.to_csv(filename, sep=";", index=None)
