# Quantum ODE Solver Implementation Report

## Overview

This report presents a PyTorch-based numerical ODE solver framework designed for quantum system simulations. The implementation features Verner's 7th-order Runge-Kutta method with adaptive step size control, dense output interpolation, and GPU acceleration capabilities.

---

## Architecture

### Module Structure

```
solver/
├── __init__.py          # Package exports
├── solver.py            # Base Integrator class
├── explicit_RK.py       # Core RK algorithm
├── rk7_coeff.py         # Verner 7 coefficients
├── sesolver.py          # Schrödinger equation solver
├── lvnesolver.py        # Lindblad von Neumann equation solver
└── upsolver.py          # Unitary propagator solver
```

### Class Hierarchy

```
Integrator (Base Class)
    ├── SESolver      (State vector: |ψ⟩)
    ├── LVNESolver    (Density matrix: ρ)
    └── UPSolver      (Unitary propagator: U)
```

---

## Core Components

### 1. Base Integrator Class (`solver.py`)

The `Integrator` class provides a common interface for all ODE solvers:

**Key Methods:**
- `set_state(t, state0)` - Initialize solver state
- `integrate(t)` - Evolve system to time t
- `get_state()` - Retrieve current (t, state, expectation values)
- `run(tlist)` - Generator for time series integration
- `mcstep(t)` - Monte Carlo step with interpolation support

**Features:**
- Automatic option management with defaults
- State reset and argument update support
- Integration with observable computation

### 2. Explicit Runge-Kutta Engine (`explicit_RK.py`)

The `ExplicitRungeKutta` class implements the numerical integration core:

**Key Features:**

| Feature | Description |
|---------|-------------|
| Adaptive Step Size | Error-controlled stepping with WRMSE norm |
| Dense Output | 7th-order polynomial interpolation |
| Memory Pooling | Pre-allocated buffers to minimize allocations |
| Event Handling | Adaptive stepping near pulse intervals |

**Status Codes:**
```python
NOT_INITIATED = -5    # Solver not initialized
OUTSIDE_RANGE = -4    # Time outside valid range
TOO_MUCH_WORK = -3    # Exceeded max steps
DT_UNDERFLOW  = -2    # Step size too small
NORMAL        = 0     # Normal operation
AT_FRONT      = 1     # At integration front
INTERPOLATED  = 2     # Using interpolation
```

### 3. Verner 7 Coefficients (`rk7_coeff.py`)

Implementation of Verner's "most efficient" 7th-order Runge-Kutta method.

**Specifications:**
- Order: 7
- Stages: 10 (main) + 6 (dense output) = 16 total
- Dense output order: 7
- Error estimation: Embedded 6th-order method

**Reference:** [https://www.sfu.ca/~jverner/](https://www.sfu.ca/~jverner/)

---

## Solver Types

### SESolver - Schrödinger Equation

Solves the time-dependent Schrödinger equation:
$$i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = H(t)|\psi(t)\rangle$$

**Observable Computation:**
```python
⟨O⟩ = ⟨ψ|O|ψ⟩ = ψ†·O·ψ
```

### LVNESolver - Lindblad von Neumann Equation

Solves the master equation for density matrices:
$$\frac{\partial \rho}{\partial t} = -\frac{i}{\hbar}[H, \rho] + \mathcal{L}[\rho]$$

**Observable Computation:**
```python
⟨O⟩ = Tr(ρ·O)
```

### UPSolver - Unitary Propagator

Evolves the unitary propagator matrix:
$$i\hbar \frac{\partial U}{\partial t} = H(t)U(t)$$

**Observable Computation:**
```python
⟨O⟩(t) = U†(t)·O·U(t)
```

---

## Key Algorithms

### Adaptive Step Size Control

The solver uses weighted root mean square error norm:

$$\text{error} = \sqrt{\frac{1}{N}\sum_i \left(\frac{|\Delta y_i|}{\text{atol} + \text{rtol} \cdot |y_i|}\right)^2}$$

**Step Size Adjustment:**
```python
factor = 0.9 * error^(-1/(order+1))
factor = clamp(factor, 0.2, 10.0)
dt_new = dt * factor
```

### Dense Output Interpolation

Uses 7th-order polynomial interpolation coefficients for efficient state retrieval at arbitrary times within a computed step:

$$y(\tau) = y_n + h\sum_{i=0}^{15} b_i(\tau) \cdot k_i$$

where $\tau = (t - t_n)/h$ and $b_i(\tau)$ are polynomial functions.

### Event-Adaptive Stepping

For pulse sequences, the solver automatically reduces step size during critical intervals:

```python
if interval_start <= t <= interval_end:
    max_step = interval_dt  # Small step during pulse
else:
    max_step = store_max_step  # Normal step otherwise
```

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `atol` | 1e-6 | Absolute tolerance |
| `rtol` | 1e-4 | Relative tolerance |
| `first_step` | 0 | Initial step size (0 = automatic) |
| `min_step` | 0 | Minimum step size |
| `max_step` | 0 | Maximum step size (0 = unlimited) |
| `interpolate` | True | Enable dense output |
| `interval_start` | None | Pulse interval start times |
| `interval_end` | None | Pulse interval end times |
| `interval_dt` | None | Step size during pulses |

---

## Usage Example

```python
import torch
from solver import SESolver, LVNESolver

# Define system dynamics
def hamiltonian(t, state):
    H = ...  # Time-dependent Hamiltonian
    return -1j * H @ state

# Create solver
solver = SESolver(
    system=hamiltonian,
    observable={'energy': H0, 'population': projector},
    options={'atol': 1e-8, 'rtol': 1e-6, 'max_step': 0.1},
    device=torch.device('cuda'),
    dtype=torch.complex128
)

# Initialize and evolve
psi0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
solver.set_state(t=0.0, state0=psi0)

# Time evolution
tlist = torch.linspace(0, 10, 100)
for t, state, expvals in solver.run(tlist):
    print(f"t={t:.2f}, ⟨E⟩={expvals['energy']:.4f}")
```

---

## Performance Optimizations

### 1. Memory Pooling
Pre-allocated buffers (`_weighted_sum_buffer`, `_interp_buffer`) avoid repeated tensor allocations during integration loops.

### 2. Vectorized Operations
Uses `torch.einsum` for efficient tensor contractions:
```python
torch.einsum('i,i...->...', factors[:size], k[:size])
```

### 3. Interval Caching
Maintains `_current_interval_idx` to skip already-passed pulse intervals, avoiding O(n) searches.

### 4. GPU Support
All computations use PyTorch tensors, enabling automatic GPU acceleration via CUDA.

---

## Error Handling

The solver provides detailed error status codes and messages:

```python
status_messages = {
    -5: "Integrator not initialized",
    -4: "Integration outside available range",
    -3: "Too much work done. Try increasing nsteps or tolerance.",
    -2: "Step size becomes too small. Try increasing tolerance.",
}
```

---

## Future Considerations

1. **Sparse Tensor Support** - For large, sparse Hamiltonians
2. **Automatic Differentiation** - For gradient-based optimization
3. **Multi-trajectory Support** - Parallel Monte Carlo simulations
4. **Higher-order Methods** - RK8, RK9 for extreme precision needs

---

## Summary

This solver implementation provides:

- **High Accuracy**: 7th-order Verner method with error control
- **Flexibility**: Supports state vectors, density matrices, and propagators
- **Performance**: GPU acceleration and memory optimization
- **Robustness**: Adaptive stepping with comprehensive error handling
- **Usability**: Clean API with configurable options

The modular design allows easy extension for new equation types while maintaining a consistent interface across all solvers.
