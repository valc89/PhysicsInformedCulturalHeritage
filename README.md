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