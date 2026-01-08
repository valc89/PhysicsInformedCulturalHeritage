# Physics-Informed Cultural Heritage

This repository provides a **research-oriented framework for physics-informed inference on 3D models** in the field of cultural heritage,
integrating **IoT data** (simulated in the examples of the repository),
**Physics-Informed Neural Networks (PINNs)**, and **Reduced Order Models (ROMs)**.  
The goal is to enable **data- and physics-driven analysis directly on realistic digital replicas**, supporting scalable and physically consistent Digital Twin applications.

## Architecture

<a name="framework-architecture"></a>

![Framework architecture](docs/images/arch.png)

**Figure 1.** Multi-layer architecture of the proposed framework. The architecture is organized into four functional layers—**Acquisition**, **Knowledge-Base**, **Inference Engine**, and **Application**—and enables inference and simulation directly on complex 3D models. The 3D Model Module provides geometry-aware preprocessing for FEM, ROM (POD), and PINN-based workflows, while simulated IoT data support both direct and inverse physics-informed analyses.

## Repository Structure

- `direct/` – PINN-based **direct problems** (TP3–TP4)  
- `inverse/` – PINN-based **inverse problems** and parameter identification (TP1–TP2)  
- `model_acquisition/` – 3D Model Module for Blender-based geometry preprocessing  
- `models/` – 3D Blender assets  
- `fem_problems/` – High-fidelity FEM solvers and POD-based ROM workflows  
- `paths.py` – Centralized path and resource management  

## Scope

The repository implements the **core inference backbone** of the architecture, demonstrating how **geometry-aware preprocessing, physics-based solvers, learning-based inference, and model reduction** can be coherently combined to operate directly on complex 3D domains.

### Architecture–Repository Mapping

The following table maps the **architectural layers and modules (Figure 1)** to their concrete implementation in the repository, highlighting how each conceptual component is instantiated in code.

| Architecture Layer / Module | Role in the Framework | Repository Components | Main Technologies |
|-----------------------------|-----------------------|-----------------------|------------------|
| **Acquisition - 3D Model Module** | Geometry preprocessing and domain definition for inference | `model_acquisition/` | Blender API (`bpy`), Mesh processing (`gmsh`), PINA API (`pina`) |
| **Inference Engine – FEM Submodule** | Full-Order solution of PDEs | `fem_problems/` | FEniCS, FEM |
| **Inference Engine – ROM Saving** | Model order reduction and efficient surrogate modeling | `fem_problems/`, `inverse/tp*/01_OfflineStage` | POD |
| **Inference Engine – PINN Module (Inverse)** | Parameter identification from simulated IoT data | `inverse/tp*/02_SolveInverse` | PINNs, SciML |
| **Inference Engine – PINN Module (Direct)** | Physics-constrained field reconstruction | `direct/tp3/`, `direct/tp4/` | PINNs, SciML |
| **Inference Engine – Online Solver Submodule** | Fast reduced-order simulation and error evaluation | `inverse/tp*/03_OnlineStage` | ROM, POD |
| **Application Layer - Simulation** | Simulation inspection and visualization | `.xdmf` outputs | ParaView |


## Reference

If you use this repository, please cite:

## Installation

Clone the repository:

```bash
git clone https://github.com/valc89/PhysicsInformedCulturalHertage.git
cd PhysicsInformedCulturalHeritage
```

Create the *conda* environment from the provided *environment.yml* file:

```bash
conda env create -f environment.yml
```

Active the environment

```bash
conda activate pich
```

> **Notes**
>
> - The provided *conda* environment ensures reproducibility of all experiments presented in the reference paper.
> - All dependencies are specified in `environment.yml`.

## Reproducible Experiments and Notebooks

The repository includes a set of **Jupyter notebooks** that illustrate the full inference pipeline on 3D models, from geometry acquisition to physics-informed learning and reduced-order simulation.  
The notebooks are organized according to the **test problems described in the paper** and are located in the following folders:

- `inverse/tp1/`
- `inverse/tp2/`
- `direct/tp3/`
- `direct/tp4/`

Each folder provides a **self-contained and reproducible workflow** operating on a specific 3D geometry and governing PDE.

---

### Summary of Test Problems

| Test Problem | Folder          | PDE / Physical Model                              | Geometry     | Inference Type | Methods Involved            |
|-------------|------------------|---------------------------------------------------|--------------|----------------|-----------------------------|
| **TP1**     | `inverse/tp1`    | Parametrized Poisson equation (scalar)             | 3D Rock      | Inverse        | FEM, POD-ROM, PINN          |
| **TP2**     | `inverse/tp2`    | Parametrized parabolic PDE system                  | 3D Column    | Inverse        | FEM, POD-ROM, PINN          |
| **TP3**     | `direct/tp3`     | Heat equation                                      | 3D Rock      | Direct         | PINN                        |
| **TP4**     | `direct/tp4`     | Poisson system                                     | 3D Column    | Direct         | PINN (with / without data)  |

---

### Inverse Problems (`inverse/tp1`, `inverse/tp2`)

The folders `inverse/tp1` and `inverse/tp2` address **inverse problems**, where unknown physical parameters are identified from simulated IoT-like data using PINNs and ROMs.  
Both test cases follow the same structured pipeline:

1. **`00_ProblemSettings`**  
   - Definition of the physical problem and parameters  
   - Generation of **simulated sensor data**  
   - Mesh generation from the Blender 3D model  
   - Creation of the `.xdmf` file to enable visualization of subsequent simulations  

2. **`01_OfflineStage`**  
   - High-fidelity FEM simulations  
   - Construction of the **POD-based Reduced Order Model (ROM)**  
   - Error analysis and assessment of the reduced basis  

3. **`02_SolveInverse`**  
   - Solution of the **inverse problem** using Physics-Informed Neural Networks  
   - Identification and approximation of the unknown physical parameters  

4. **`03_OnlineStage`**  
   - Online POD stage using the parameters inferred by the PINN  
   - Comparison with reference solutions computed using exact parameters  
   - Generation of `.xdmf` files for simulation results and error visualization  

**Test problems:**
- **TP1** – Parametrized Poisson equation on a 3D rock geometry (scalar problem)  
- **TP2** – Parametrized system of parabolic PDEs on a 3D column  

---

### Direct Problems (`direct/tp3`, `direct/tp4`)

The folders `direct/tp3` and `direct/tp4` focus on **direct problems**, where the governing physics is known and PINNs are used to reconstruct the solution field directly on the 3D domain.

#### TP3 – Direct PINN solution
- **`00_Problem_Settings`**  
  - Generation of simulated data  
  - Mesh creation and `.xdmf` preparation  
- **Single notebook**  
  - Direct solution of the heat equation using PINNs  
  - Generation of `.xdmf` files for simulation and visualization  

**Test problem:**
- **TP3** – Heat equation on a 3D rock geometry  

#### TP4 – Direct PINN solution with and without data
- **`00_Problem_Settings`**  
  - Generation of simulated data  
  - Mesh creation and `.xdmf` preparation  
- **Two notebooks**  
  - Direct PINN solution **with integrated simulated data**  
  - Direct PINN solution **without data**, relying only on physical constraints  

**Test problem:**
- **TP4** – Poisson system on a 3D column geometry  

---

Overall, these notebooks provide a **guided and progressive entry point** to the framework, demonstrating how **3D geometry acquisition, simulated IoT data, physics-informed learning, and reduced-order modeling** are combined to perform inference and simulation on complex digital assets.