Emergent Quantum Statistics in Local Pilot-Wave Dynamics

Author: Christian Revoire (Independent Researcher)

Date: Jan 2026

This repository contains the high-performance Python/Numba implementation of a local hidden variable model inspired by walking droplets (Couder/Fort/Bush). The simulation demonstrates that Born's Rule and Pauli Exclusion can emerge from a purely local, deterministic wave-particle dynamics with feedback.


----------
📁 Key Results & Scripts & Interpretation

1. Born Rule Convergence (1D)

    - Principle
      
       This script simulates a pilot-wave dynamic with feedback (walking droplet type)
       in 1D to demonstrate the statistical emergence of Born's Rule.
       By statistical convergence and Ergodicity, from singles particle dynamics
        
    - Physical Model
      
       This model couples a stochastic point particle to a complex scalar field (pilot wave):
       1. The wave evolves according to a complex Ginzburg-Landau equation (Schrödinger-like).
       2. The particle is guided by the local phase gradient of the field (Langevin dynamics).
       3. The particle acts as a moving source, continuously interecting and fueling its own pilot wave (feedback).
       4. The system is in a state of free expansion (diffusion).
        
    - Key Result

       The simulation demonstrates that the particle's statistical distribution ρ(x)
       dynamically conforms to the shape of the spread wave packet |ψ|²,
       with an high correlation and a low error L1.
      
       In other words, the probability density of the particle's position converges towards 
       the intensity |ψ|² of the field, validating the dynamical "quantum relaxation"
       towards Born's Rule without axiomatic postulates.
      
       It validates the emergence of Born's Rule from a purely deterministic, local, and realistic dynamics.

    - Script: src/simulation_1d_born.py

    - Performance: ~2-10 minutes (depending on CPU cores).

1. Pauli Exclusion Principle (1D)

    - Principle

        This simulation investigates whether fermionic-like exclusion effects
        can emerge dynamically from a purely local pilot-wave model, without
        imposing antisymmetry, exchange rules, or quantum statistics by hand.
        
        The model extends a previously validated single-particle framework
        demonstrating dynamical convergence toward the Born rule (ρ ≈ |ψ|²).
        Here, the focus is placed on two-particle correlations.
        
    - Physical Model

        Each particle continuously emits and interacts with its own complex
        guiding field ψ₁(x,t), ψ₂(x,t). Particle dynamics is governed by the
        phase gradient of an effective guiding field constructed from the sum 
        of these individual fields.
        
    - Key assumptions

        • No antisymmetrization of trajectories
        • No explicit exclusion principle
        • No fermionic statistics imposed
        
        Only local field–particle coupling and stochastic diffusion are included.
        
    - Key Result

        Despite the absence of any imposed exclusion rule, ensemble-averaged
        statistics reveal:

        • A depletion of the pair correlation function at short distances:
          g(r) << 1
        • A correlation hole analogous to the Fermi hole
        • A suppression of joint configurations along x₁ = x₂
        • Single particles statistical distribution ρ(x) dynamically conforms to the shape of the spread wave packet |ψ|².
        
        These results suggest that Pauli-like exclusion may emerge as a consequence 
        of local pilot-wave dynamics, rather than as a fundamental postulate.
    
    - Script: src/simulation_1d_pauli.py

    - Performance: ~10-50 minutes (depending on CPU cores).

2. Born rule and Polarization & Vectorial Flow (2D)

    Physics: 2D extension showing that the probability current J generates local vortex structures (emergent spin) from a scalar field.

    Two implementations are available:

    * **CPU Version:** `src/simulation_2d_born_&_polarization.py` (Reference implementation, slow).

      Performance: ~1-5 hours depending on CPU cores.
      
    * **GPU Version (Recommended):** `notebooks/Simulation_2D_Born_&_polarization_GPU_Colab.ipynb` High-performance implementation using Taichi Lang.
      (Coded for Google Colab or a powerful graphics card)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zobeewan/Born_Rule_emergence-Pilot_Wave/blob/main/notebooks/Simulation_2D_Born_&_polarization_GPU_Colab.ipynb)   Performance: ~30-150 minutes     
    


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
