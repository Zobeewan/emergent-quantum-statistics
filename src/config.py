# --- Space & Time ---
Nx: int = 300                 # Grid size
Ny: int = 300                 # Only use on 2D simulation
x_min: float = -50.0          # Spatial extent (left)
x_max: float = 50.0           # Spatial extent (right)
dt: float = 0.01              # Time step

# --- Simulation Control ---
N_steps: int = 40000          # Steps per particle
thermalization: int = 10000   # Steps to ignore (warmup)
N_runs: int = 1200            # Number of independent particles
N_CORES: int = -1             # 0 = use all available cores, -1 = keep 1 cores free (or more)
SUBSAMPLE: int = 1            # 1 = keep all points, 10 = keep 1/10 (avoid use if possible)

# --- Wave Physics (Pilot Field) ---
# Equation: ∂t ψ = (Dψ + iω)∇²ψ - γψ + Source
c: float = 1.0                # Propagation speed (normalized)
gamma: float = 0.02           # Dissipation (memory limit)
D_psi: float = 0.9            # Spatial diffusion
emit_amp: float = 0.57        # Source amplitude
sigma_emit: float = 1.0       # Source width (will be scaled by dx)
omega: float = 2.0            # Dispersion frequency (analog to ℏ/2m)

# --- Particle Physics ---
# Equation: dx = (α ⋅ ∇φ) dt + noise
alpha: float = 4.0            # Coupling strength (Inertial factor, analog k/m, equal to 2ω mean k=1)
D_x: float = 0.28             # Stochastic diffusion (Brownian noise)
epsilon: float = 1e-3         # Regularization factor for guidance

def __init__(self):
    # Derived parameters
    self.x = np.linspace(self.x_min, self.x_max, self.Nx)        # Grid initialization
    self.dx = self.x[1] - self.x[0]                              # Space step
    # Adjust source width relative to grid
    self.sigma_emit_scaled = self.dx * 3.0
    
# Global instance for easy access (can be overridden)
CFG = SimConfig()
