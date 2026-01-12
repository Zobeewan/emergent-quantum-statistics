import numpy as np

# ===============================
# CONFIG GLOBAL
# ===============================
class Base_Config:                # (Opti for Born 1D)
    # --- Space & Time ---
    Nx: int = 300                 # Grid size (x)
    Ny: int = 300                 # Grid size (y), Only use on 2D simulation
        
    x_min: float = -120.0         # Spatial extent (x) (left) 
    x_max: float = 120.0          # Spatial extent (x) (right)
    
    y_min: float = -120.0         # Spatial extent (y) (left), Only use on 2D simulation
    y_max: float = 120.0          # Spatial extent (y) (right), Only use on 2D simulation
    
    dt: float = 0.01              # Time step
    
    # --- Simulation Control ---
    N_steps: int = 40000          # Steps per particle
    thermalization: int = 10000   # Steps to ignore (warmup)
    N_runs: int = 3000            # Number of independent simulated particles
    N_CORES: int = -1             # 0 = use all available cores, -1 = keep 1 cores free (or more)
    SUBSAMPLE: int = 1            # 1 = keep all points, 10 = keep 1 point out of 10 (avoid use if possible)
    
    # --- Wave Physics (Pilot Field) ---
    # Equation: ∂t ψ = (Dψ + iω)∇²ψ - γψ + Source
    c: float = 1.0                # Propagation speed (c = 1 by choice of units, ideally should match the discretization dx/dt)    
    gamma: float = 0.045          # Dissipation (system memory)
    D_psi: float = 0.9            # Spatial diffusion
    emit_amp: float = 0.57        # Source emission amplitude
    sigma_emit: float = 1.0       # Spatial width of the source (will be scaled by dx)
    omega: float = 2.0            # Dispersive frequency (analog to ℏ/2m)
    
    # --- Particle Physics ---
    # Equation: dx = (α ⋅ ∇φ) dt + noise
    alpha: float = 4.0            # Coupling strength (Inertial factor, analog k/m, equal to k*2ω)
    D_x: float = 0.28             # Stochastic diffusion (Brownian noise)
    epsilon: float = 1e-3         # Regularization factor for guidance
    
    def __init__(self):
        # Derived parameters
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)        # Grid initialization
        self.y = np.linspace(self.y_min, self.y_max, self.Ny)
        self.dx = self.x[1] - self.x[0]                              # Space step
        self.dy = self.y[1] - self.y[0]
        # Adjust source width relative to grid
        self.sigma_emit_scaled = self.dx * 3.0

# ===============================
# CONFIG PAULI 1D
# ===============================
class Pauli_Config(Base_Config):   # (Opti for pauli 1D)
    # --- Space & Time ---
    x_min: float = -150.0          # Spatial extent (x) (left) 
    x_max: float = 150.0           # Spatial extent (x) (right)
    
    # --- Simulation Control ---
    N_runs: int = 1200             # Number of independent simulated particles

    
    # --- Initial configuration of particle positions --- 
    SIDE = "norm"                     
    # "norm" = Particle 1 on the left, Particle 2 on the right
    # "rand" = Randomized left/right assignment                   
    # any other value : inverted configuration
    
    if SIDE == "norm":
        start_area_p1: float = (-15.0, -5.0)   # Particle 1 starts on the left
        start_area_p2: float = (5.0, 15.0)     # Particle 2 starts on the right
    elif SIDE == "rand":
        start_area_p1: float = None            # Random starts
        start_area_p2: float = None            # Random starts
    else:  # "inverted"
        start_area_p1: float = (5.0, 15.0)     # Particle 1 starts on the right
        start_area_p2: float = (-15.0, -5.0)   # Particle 2 starts on the left 

    
    # --- Coupling ---
    coupling_type = "sum"
    # "sum"  : ψ_guide = ψ₁ + ψ₂
    #          Leads to effective repulsion and fermion-like correlations
    # "diff" : ψ_guide = ψ₁ − ψ₂
    #          Leads to effective attraction and boson-like correlations


# ===============================
# CONFIG BRON 2D
# ===============================
class Born_2D_Config(Base_Config): # (Opti for Born 2D)
    # --- Space & Time ---
    x_min: float = -125.0          # Spatial extent (x) (left) 
    x_max: float = 125.0           # Spatial extent (x) (right)
    
    # --- Simulation Control ---
    N_runs: int = 600              # Number of independent simulated particles

    # ---  Potentials Control ---
    V0 = 0.1                       # Harmonic Potential
    V0_coulomb = 15.0              # Coulomb Potential
    softening = 1.0



