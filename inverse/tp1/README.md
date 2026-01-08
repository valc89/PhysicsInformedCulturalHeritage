# TP1 – Inverse Parametrized Poisson Problem on a 3D Rock

This folder implements **Test Problem 1 (TP1)**, which addresses an **inverse parametrized Poisson problem** defined on a **3D rock geometry**.  
The objective is to infer unknown physical parameters from **simulated IoT-like boundary measurements** by combining **Physics-Informed Neural Networks (PINNs)** and **Reduced Order Models (ROMs)** within an offline–online framework.

---

## Physical Problem Description

Let \( \Omega \subset \mathbb{R}^3 \) denote the domain associated with the rock geometry.  
The physical phenomenon is modeled by the following **parametrized Poisson equation**:

\[
\begin{cases}
\Delta u(x, y, z) = -(\alpha^2 + \beta^2)\pi^2 \lambda x \cos(\alpha \pi y)\sin(\beta \pi z), & \text{in } \Omega, \\
u(x, y, z) = \lambda x \cos(\alpha \pi y)\sin(\beta \pi z), & \text{on } \partial \Omega,
\end{cases}
\]

where:
- \( u \) represents the scalar field of interest (e.g., temperature),
- \( \lambda \) is the amplitude of the forcing term,
- \( \alpha \) and \( \beta \) control the spatial oscillations along the \( y \) and \( z \) directions.
---

## Inverse Problem Objective

The inverse problem consists of **identifying the unknown parameters**
\[
\mu = (\lambda, \alpha, \beta)
\]
from **simulated sensor measurements**, which emulate IoT data collected on the boundary of the 3D rock.

The inference task is carried out by:
- enforcing the governing PDE through a **Physics-Informed Neural Network**, and
- assimilating simulated boundary data as observational constraints.

---