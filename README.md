# Emergent Quantum Statistics in Stochastic Pilot-Wave Dynamics

[![DOI](https://zenodo.org/badge/1129929103.svg)](https://doi.org/10.5281/zenodo.20260635)

**Christian Revoire**  
Independent Researcher — January 2026


----------

## 🔬 Overview

This repository contains high-performance Python/Numba simulations of a local hidden-variable
model inspired by walking droplets (Couder–Fort–Bush).

The goal is to investigate whether key quantum statistical properties, such as Born’s rule
and fermionic exclusion, can **emerge dynamically** from local wave–particle dynamics with feedback,
without imposing quantum axioms by hand.


---

## 📊 Interpretation & Scripts & Key Results

### 1️⃣ Emergence of Born’s Rule (1D)

**Objective:** 

Demonstrate the statistical emergence of Born's Rule.
That particle guided by a local pilot-wave with feedback
dynamically relaxes toward the Born probability density ρ(x) ≈ |ψ(x)|².
By statistical convergence and Ergodicity, from singles particle dynamics

**Physical Model:**

This model couples a stochastic point particle to a complex scalar field (pilot wave):
- The wave evolves according to a complex Ginzburg-Landau equation (Schrödinger-like).
- The particle is guided by the local phase gradient of the field (Langevin dynamics).
- The particle acts as a moving source, continuously interacting and fueling its own pilot wave (feedback).
- The system is in a state of free expansion (diffusion).
    
**Key Result:**

- The simulation demonstrates that the particle's statistical distribution ρ(x)
  dynamically conforms to the shape of the spread wave packet |ψ|²,
  with a high correlation and a low error L1.

- In other words, the probability density of the particle's position converges towards
  the intensity |ψ|² of the field, validating the dynamical "quantum relaxation"
  towards Born's Rule without axiomatic postulates.

- It supports the emergence of Born's Rule from a purely deterministic, and realistic dynamics.

**📁 Script:** `src/simulation_1D_Born.py`  

**⏱ Runtime:** ~2–10 minutes (depending on params and CPU cores).

----------

### 2️⃣ Emergent Pauli-Like Exclusion (1D)

**Objective:** 

This simulation investigates whether fermionic-like exclusion effects
can emerge dynamically from a local pilot-wave dynamics, without
imposing antisymmetry, exchange rules, or quantum statistics by hand.

The model extends a previously validated single-particle framework
demonstrating dynamical convergence toward the Born rule (ρ ≈ |ψ|²).
Here, the focus is placed on two-particle correlations.
    
**Physical Model:**

- Each particle continuously emits and interacts with its own complex
  guiding field ψ₁(x,t), ψ₂(x,t). 
- Particles dynamics is governed by the phase gradient
  of an effective guiding field constructed
  from the sum of these individual fields.
    
**Key assumptions:**
  
- ❌ No antisymmetrization of trajectories
- ❌ No explicit exclusion principle
- ❌ No fermionic statistics imposed
- ✅ Only local field–particle coupling and stochastic diffusion are included.
    
**Key Result:**

Ensemble-averaged statistics reveal:
- A depletion of the pair correlation function at short distances:
  g(r) << 1
- A correlation hole analogous to the Fermi hole
- A suppression of joint configurations along x₁ = x₂
- Single-particle densities still obey ρ ≈ |ψ|²

These results suggest that Pauli-like exclusion may emerge as a consequence 
of local pilot-wave dynamics, rather than as a fundamental postulate.

**📁 Script:** `src/simulation_1D_Pauli.py`  

**⏱ Runtime:** ~10–50 minutes (depending on params and CPU cores).

----------

### 3️⃣ Born Rule & Polarization in 2D

**Objective:**  

Extend the pilot-wave model to 2D and analyze statistical convergence and 
to reveals local vectorial structures of the field (vortices).

**Key Result:**

- The particle correctly samples the 2D field,
  with empirical density ρ(x,y) ≈ |(x,y)|²
- Although ψ is a scalar field, analysis of the probability current
  reveals local vectorial structures (vortices),
  suggesting a possible connection with intrinsic angular momentum (spin)


**📁 Script:**
  
Two implementations are available:

- **CPU Version:** `src/simulation_2D_Born_&_polarization.py`

  Reference implementation (slow, not recommended for high N_runs with high correlation)

  **⏱ Runtime:** ~1-5 hours (depending on params and CPU cores)
  
- **GPU Version (Recommended):** `notebooks/simulation_2D_Born_&_polarization_GPU_Colab.ipynb`
  
  High-performance implementation using Taichi Lang. (Coded for Google Colab GPU)
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zobeewan/Born_Rule_emergence-Pilot_Wave/blob/main/notebooks/simulation_2D_Born_&_polarization_GPU_Colab.ipynb)

  **⏱ Runtime:** ~20–100 minutes (depending on params)

----------

## 🛠️ Installation

This code relies on Numba for JIT compilation and Joblib for parallel execution.

**Install dependencies:**
```bash
pip install numpy matplotlib scipy numba joblib tqdm
``` 
or

```bash    
pip install -r requirements.txt
```

----------

## ⚙️ Model Parameters

Key physical parameters (diffusion Dψ​, coupling α, memory γ, Dispersive frequency ω, Amplitude source, stochastic noise) are defined inside `src/config.py`.
The default values are tuned to reach the quantum relaxation regime.

----------

## ▶️ Usage

**Run 1st simulation:**
```bash
src/simulation_1D_Born.py
```

----------

## ⚠️ Scientific Disclaimer

This repository presents exploratory numerical simulations in the context of quantum foundations.

The results reported here do **not** claim to reproduce quantum mechanics in full generality.

All conclusions are model-dependent and subject to further validation, falsification, or reinterpretation.
