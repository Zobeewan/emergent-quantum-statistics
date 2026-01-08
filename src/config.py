class SimConfig:
    # --- Space & Time ---
    Nx: int = 300                 # Grid size (x)
    Ny: int = 300                 # Grid size (y), Only use on 2D simulation
    
    x_min: float = -50.0          # Spatial extent (x) (left) 
    x_max: float = 50.0           # Spatial extent (x) (right)
    
    y_min: float = -50.0          # Spatial extent (y) (left), Only use on 2D simulation
    y_max: float = 50.0           # Spatial extent (y) (right), Only use on 2D simulation
    
    dt: float = 0.01              # Time step
    
    # --- Simulation Control ---
    N_steps: int = 40000          # Steps per particle
    thermalization: int = 10000   # Steps to ignore (warmup)
    N_runs: int = 1200            # Number of independent simulated particles
    N_CORES: int = -1             # 0 = use all available cores, -1 = keep 1 cores free (or more)
    SUBSAMPLE: int = 1            # 1 = keep all points, 10 = keep 1 point out of 10 (avoid use if possible)
    
    # --- Wave Physics (Pilot Field) ---
    # Equation: ∂t ψ = (Dψ + iω)∇²ψ - γψ + Source
    c: float = 1.0                # Propagation speed (c = 1 by choice of units, ideally should match the discretization dx/dt)    
    gamma: float = 0.02           # Dissipation (system memory)
    D_psi: float = 0.9            # Spatial diffusion
    emit_amp: float = 0.57        # Source emission amplitude
    sigma_emit: float = 1.0       # Spatial width of the source (will be scaled by dx)
    omega: float = 2.0            # Dispersive frequency (analog to ℏ/2m)
    
    # --- Particle Physics ---
    # Equation: dx = (α ⋅ ∇φ) dt + noise
    alpha: float = 4.0            # Coupling strength (Inertial factor, analog k/m, equal to k*2ω)
    D_x: float = 0.28             # Stochastic diffusion (Brownian noise)
    epsilon: float = 1e-3         # Regularization factor for guidance
    
# Global instance for easy access (can be overridden)
CFG = SimConfig()
