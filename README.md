# Physics-Informed Cultural Heritage

## Description

**Physics-Informed Cultural Heritage** is a research framework for cultural heritage conservation that integrates  
**Physics-Informed Neural Networks (PINNs)**, **Reduced Order Models (ROMs)**, **Finite Element Methods (FEM)**, and **Internet of Things (IoT)** data within a unified **Digital Twin** architecture.

The repository supports:

- physics-aware simulations on complex 3D cultural heritage geometries,
- hybrid data-driven and physics-based modeling,
- direct and inverse problem solving governed by partial differential equations,
- efficient offline–online workflows for predictive maintenance and monitoring.

The framework, shown in [Figure 1](#framework-architecture), is designed to assist conservation experts by combining physical knowledge, sensor data, and artificial intelligence in a scalable and reproducible workflow.

<a name="framework-architecture"></a>

![Framework architecture](docs/images/arch.png)

**Figure 1 –** Overview of the Physics-Informed Cultural Heritage framework, illustrating the integration of IoT, physics-based models, and AI components.

## Reference

If you use this repository, please cite:

## Installation

Clone the repository:

```bash
git clone https://github.com/valc89/PhysicsInformedCulturalHertage.git
cd PhysicsInformedCulturalHeritage
```

Create the *conda* environment from the provided environment.yml file:

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