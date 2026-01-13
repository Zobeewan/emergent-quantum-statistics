Emergent Quantum Statistics in Local Pilot-Wave Dynamics

Author: Christian Revoire (Independent Researcher)

Date: Jan 2026

This repository contains the high-performance Python/Numba implementation of a local hidden variable model inspired by walking droplets (Couder/Fort/Bush). The simulation demonstrates that Born's Rule and Pauli Exclusion can emerge from a purely local, deterministic wave-particle dynamics with feedback.


----------
📁 Key Results & Scripts

1. Born Rule Convergence (1D)

    Script: src/simulation_1d_born.py

    Physics: Statistical convergence, Ergodicity, Single particle dynamics.

    Result: Empirical density ρ(x) converges to ∣ψ∣2 with high correlation and low error L1.

    Performance: ~2-10 minutes (depending on CPU cores).

2. Pauli Exclusion Principle (1D)

    Script: src/simulation_1d_pauli.py

    Physics: Interaction between two identical particles.

    Result: Spontaneous anti-correlation and emergence of the Fermi hole (g(2)(r)<<1).

    Performance: ~10-50 minutes (depending on CPU cores).

3. Born rule and Polarization & Vectorial Flow (2D)

    Script: src/simulation_2d_born_&_polarization.py

    Physics: 2D extension showing that the probability current J generates local vortex structures (emergent spin) from a scalar field.

    Two implementations are available:

    * **CPU Version:** `src/simulation_2d_cpu.py` (Reference implementation, slow).

      Performance: ~1-5 hours depending on CPU cores.
      
    * **GPU Version (Recommended):** High-performance implementation using Taichi Lang.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zobeewan/Born_Rule_emergence-Pilot_Wave/tree/main/notebooks/Simulation_2D_Born_&_polarization_GPU_Colab.ipynb)   Performance: ~30-150 minutes     
    


----------
🛠️ Installation & Usage

This code relies on Numba for JIT compilation and Joblib for parallel execution.

1. Install dependencies:
    
    pip install numpy matplotlib scipy numba joblib tqdm
   
   or
   
    pip install -r requirements.txt

3. Run 1st simulation:

    python src/simulation_1d_born.py


----------
⚙️ Model Parameters

Key physical parameters (diffusion Dψ​, coupling α, memory γ, Dispersive frequency ω, Amplitude source, stochastic noise) are defined inside src/config.py 
The default values are tuned for the "quantum regime" convergence.
