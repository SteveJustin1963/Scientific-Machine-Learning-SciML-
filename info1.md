

```markdown
# Symplectic Neural ODEs for Noisy Hamiltonian Systems

A research-level SciML project combining Neural ODEs, Hamiltonian mechanics, and symplectic integration. The central question: **when and why does hard architectural enforcement of conservation laws outperform soft penalty-based approaches as training noise increases?**

## Table of Contents

1. [Project Proposal](#1-project-proposal)
2. [Mathematical Foundation: Hamiltonian Mechanics](#2-mathematical-foundation-hamiltonian-mechanics)
3. [The Leapfrog Integrator: Code and Theory](#3-the-leapfrog-integrator-code-and-theory)
4. [Symplectic Geometry and Its Discrete Analog](#4-symplectic-geometry-and-its-discrete-analog)
5. [Production Code](#5-production-code)
6. [Experimental Design](#6-experimental-design)
7. [Symplectic Neural ODEs for Robotics — Flowchart](#7-symplectic-neural-odes-for-robotics--flowchart)
8. [Robotics Implementation & Deployment](#8-robotics-implementation--deployment)
9. [Math Inventory](#9-math-inventory)

---

# 1. Project Proposal

## Leading-Edge SciML Project: Symplectic Neural ODEs for Noisy Hamiltonian Systems

This is a project at the absolute frontier of SciML (circa 2024–2026), where geometric mechanics meets deep learning. It sits at the intersection of **Neural ODEs**, **Hamiltonian mechanics**, and **symplectic integration**. 

### The Core Idea

Traditional Neural ODEs learn dynamics by integrating $dx/dt = f_\theta(x,t)$ using a generic ODE solver (e.g., Runge-Kutta). The problem: for physical systems like pendulums, planetary orbits, or molecular dynamics, **energy should be conserved**, but standard integrators drift over time (simulations "create" or "destroy" energy).

You will build a network that **cannot violate the laws of physics by construction**, not by penalty. This is the difference between:
- **Physics-Regularized Neural ODEs:** Add $\lambda \|\frac{dH}{dt}\|^2$ to the loss $\rightarrow$ soft constraint. *(Note: True Physics-Informed Neural Networks (PINNs) usually map $t \mapsto x(t)$ directly using the network's own derivatives, rather than relying on an external ODE solver. For our baselines, we compare against physics-regularized Neural ODEs).*
- **Structure-based (this project):** Architect the network so that energy conservation is mathematically guaranteed $\rightarrow$ hard constraint.

### The Novel Insight You Could Make (The "Research Question")

**The open question is:**

> *How does the geometric structure of symplectic Neural ODEs interact with noisy training data, and when does hard architectural enforcement outperform soft penalty methods?*

Some hypotheses you could test:
1. **Hypothesis A:** Symplectic NNs degrade gracefully under noise because the conservation law acts as a powerful regularizer.
2. **Hypothesis B:** Penalty-based methods win on noisy data because they can "give up" conservation when data is unreliable, while strict symplectic nets are rigid.
3. **Hypothesis C:** There is a **phase transition**—a critical noise level beyond which structure-based methods collapse or win decisively.

---

# 2. Mathematical Foundation: Hamiltonian Mechanics

## 1. Where We Start: The Lagrangian
In classical mechanics, a system with $n$ generalized coordinates $q = (q_1, \ldots, q_n)$ is described by a **Lagrangian**:
$$L(q, \dot{q}, t) = T(q, \dot{q}) - V(q)$$

## 2. The Legendre Transform: The Key Step
We promote velocities to **independent variables** called **canonical momenta**:
$$p_i \equiv \frac{\partial L}{\partial \dot{q}_i}$$

The **Hamiltonian** is defined as:
$$H(q, p, t) \equiv \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)$$

## 3. Deriving Hamilton's Equations
Taking differentials gives **Hamilton's canonical equations**—a **first-order** system in $2n$ variables $(q, p)$:
$$\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}$$

## 4. The Symplectic Structure (The Magic)
Notice the **antisymmetry**. We can write them compactly as:
$$\dot{x} = J\,\nabla H(x), \quad x = \begin{pmatrix} q \\ p \end{pmatrix}, \quad J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$$
The matrix $J$ is the **standard symplectic matrix**, the algebraic shadow of the **symplectic 2-form** $\omega = \sum_i dp_i \wedge dq_i$. Hamiltonian flows **preserve this volume exactly**.

---

# 3. The Leapfrog Integrator: Code and Theory

Generic numerical ODE solvers (Euler, RK4) do **not** preserve geometric structure. Symplectic integrators do. 

## Deriving Leapfrog
The leapfrog scheme can be derived as the exact flow of a modified Lagrangian, resulting in a staggered update:
$$\begin{aligned} p_{k+1/2} &= p_k - \frac{h}{2} \frac{\partial H}{\partial q}(q_k) \\ q_{k+1} &= q_k + h \frac{\partial H}{\partial p}(p_{k+1/2}) \\ p_{k+1} &= p_{k+1/2} - \frac{h}{2} \frac{\partial H}{\partial q}(q_{k+1}) \end{aligned}$$

## Implementation (Optimized via `torch.func`)
*Note: Previous naive implementations looping `torch.autograd` caused severe memory leaks. We utilize modern functional autograd.*

```python
import torch
from torch.func import grad, vmap

def leapfrog(H_single, q0, p0, h, n_steps, return_trajectory=False):
    """
    Symplectic leapfrog integration.
    H_single(q, p) -> scalar  (must evaluate a single unbatched sample).
    """
    # Vectorize exact gradients across the batch dimension
    dHdq_fn = vmap(grad(H_single, argnums=0))
    dHdp_fn = vmap(grad(H_single, argnums=1))

    qs, ps = [q0], [p0]
    q, p = q0, p0

    for _ in range(n_steps):
        p = p - (h / 2) * dHdq_fn(q, p)
        q = q + h * dHdp_fn(q, p)
        p = p - (h / 2) * dHdq_fn(q, p)
        
        if return_trajectory:
            qs.append(q)
            ps.append(p)

    if return_trajectory:
        return torch.stack(qs), torch.stack(ps)
    return q, p

```

---

# 4. Symplectic Geometry and Its Discrete Analog

## Backward Error Analysis (The "Why")

A symplectic integrator with step size $h$ does not exactly preserve the true $H$, but it exactly preserves a **modified Hamiltonian** $\tilde{H}$ via **Backward Error Analysis** (Hairer, Lubich, and Wanner):


$$\tilde{H}(q, p) = H(q, p) + h^2 H_2(q, p) + h^4 H_4(q, p) + \cdots$$

**Crucial consequence:** $\tilde{H}$ is *conserved exactly* (to machine precision) by the discrete map. Energy $H$ oscillates around the true value within $O(h^2)$ but does **not drift**. Long-time simulations stay on the energy manifold.

---

# 5. Production Code

Building on the `torch.func` framework, a robust implementation for benchmarking involves higher-order compositions like Yoshida4.

## The 4th-Order Yoshida Integrator

A 4th-order symplectic step is built from three leapfrog steps:


$$\Phi_h^{(4)} = \Phi_{w_1 h} \circ \Phi_{w_0 h} \circ \Phi_{w_1 h}$$


where $w_1 = \frac{1}{2 - 2^{1/3}}$ and $w_0 = 1 - 2w_1$.

```python
def yoshida4(H_single, q, p, h):
    cbrt2 = 2 ** (1/3)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = 1.0 - 2.0 * w1

    q, p = leapfrog(H_single, q, p, w1 * h, 1, False)
    q, p = leapfrog(H_single, q, p, w0 * h, 1, False)
    q, p = leapfrog(H_single, q, p, w1 * h, 1, False)
    return q, p

```

---

# 6. Experimental Design

This outlines the rigorous, publishable framework to evaluate these models.

## 1. Data Generation & Noise Injection (Partial Observability)

Adding independent noise to both $q$ and $p$ is physically unnatural. We simulate partial observability by injecting noise *only* into the configuration space $q$, and computing the noisy momenta $p$ via finite differences:

$$q_i^{\text{noisy}} = q_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2 I)$$

$$p_i^{\text{noisy}} = M \left( \frac{q_i^{\text{noisy}} - q_{i-1}^{\text{noisy}}}{h} \right)$$

This introduces realistic correlated noise.

## 2. Statistical Significance & Seed Count

* **Seeds:** **10 to 30 seeds** per (model, system, SNR) configuration. Variance in chaotic systems (like Hénon-Heiles) is too extreme for 5 seeds to yield reliable confidence intervals. You must have sufficient statistical power to claim a phase transition.

## 3. Computational Considerations: Gradient Checkpointing

Rolling out a symplectic integrator for 50+ steps inside a training loop creates a massive computation graph, leading to memory leaks. **Gradient checkpointing is mandatory** for long backpropagation through time.

```python
from torch.utils.checkpoint import checkpoint

# Inside your training loop, chunk the rollouts:
q_pred, p_pred = checkpoint(
    leapfrog, 
    model.H_single, q, p, h, n_steps=10, 
    use_reentrant=False
)

```

## 4. The Three Main Figures

1. **Energy Drift vs. Rollout Length:** Proves strict conservation (flat lines for symplectic, upward drift for standard).
2. **The Phase Transition Plot:** Y-axis is Rollout MSE; X-axis is SNR (Noise). The crossing point marks the transition where hard constraints out-compete flexible penalties.
3. **Phase Space Reconstruction:** Qualitative visual of orbits.

---

# 7. Symplectic Neural ODEs for Robotics — Flowchart

```
┌──────────────────────────────────────────────────────────────┐
│                    ROBOT PHYSICS                             │
│   Real robot has clean physics (Newton-Euler) + messy real   │
│   world stuff (friction, contact, wear, flex).               │
└──────────────────────────────────────────────────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │     WHAT IS A SYMPLECTIC NN?          │
        │   A neural network that LEARNS how    │
        │   much energy the robot has at any    │
        │   (position, momentum) state.         │
        └───────────────────────────────────────┘
                            ▼
   ┌──────────────────────────────────────┐
   │  PART 1: Known Analytical Physics    │
   │  H_analytic(q, p) (FROZEN)           │
   └──────────────────────────────────────┘
              +
   ┌──────────────────────────────────────┐
   │  PART 2: Learned Neural Residual     │
   │  Learns friction, contact, etc.      │
   └──────────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────┐
   │  OUTPUT: Total energy H(q, p)        │
   │  H = H_analytic + H_neural           │
   └──────────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────┐
   │  STEP 3: Auto-diff via torch.func    │
   │  Computes exact equations of motion  │
   └──────────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────┐
   │  STEP 4: Symplectic Integrator       │
   │  Preserves energy exactly over time  │
   └──────────────────────────────────────┘

```

---

# 8. Robotics Implementation & Deployment

Robotic systems are *inherently* Hamiltonian. A vanilla learned model that drifts 1% energy per second produces a trajectory your MPC will chase into a wall.

## Residual Symplectic NN (Recommended)

Don't learn $H$ from scratch — **learn the residual** between your analytical model and reality.

```python
import torch
import torch.nn as nn

class ResidualHamiltonianNN(nn.Module):
    def __init__(self, n_dof, hidden=128):
        super().__init__()
        self.n_dof = n_dof
        
        # Learned correction
        self.correction_net = nn.Sequential(
            nn.Linear(2 * n_dof, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def H_total_single(self, q, p):
        # True analytical Hamiltonian (call user's fast evaluator)
        H_analytic = self.H_analytic_fast(q, p)
        # Learned residual
        x = torch.cat([q, p], dim=-1)
        delta_H = self.correction_net(x).squeeze(-1)
        return H_analytic + delta_H

```

## Where it Installs in the Stack

Every method installs in the **DYNAMICS MODEL + INTEGRATOR** layer. State estimation and MPC control logic remain entirely untouched.

```python
# robot_controller/dynamics/symplectic_integrator.py
from torch.func import grad, vmap

class SymplecticRobotIntegrator:
    def __init__(self, model):
        self.model = model # Your ResidualHamiltonianNN
    
    def step(self, q, p, tau, h):
        """ONE forced leapfrog step. Drop-in for RK4."""
        # Note: model.H_total_single evaluates an unbatched tensor
        dHdq_fn = vmap(grad(self.model.H_total_single, argnums=0))
        dHdp_fn = vmap(grad(self.model.H_total_single, argnums=1))
        
        # Half-kick with control
        dHdq = dHdq_fn(q, p)
        p_half = p - (h/2) * dHdq + (h/2) * tau
        
        # Full drift
        q_new = q + h * dHdp_fn(q, p_half)
        
        # Half-kick at new position
        dHdq_new = dHdq_fn(q_new, p_half)
        p_new = p_half - (h/2) * dHdq_new + (h/2) * tau
        
        return q_new, p_new

```

---

# 9. Math Inventory

| Purpose | Key Equations |
| --- | --- |
| **System definition** | Lagrangian, Legendre transform, $H(q,p)$ |
| **Continuous dynamics** | $\dot{x}=J\nabla H$, Poisson brackets |
| **Geometric invariant** | $\omega = \sum dp_i \wedge dq_i$, $J^T M^T JM=J$, Liouville |
| **Discrete dynamics** | Leapfrog, Yoshida, Forest-Ruth |
| **Long-term accuracy** | Backward Error Analysis: Modified Hamiltonian $\tilde{H} = H + O(h^2)$ |
| **Learning** | $H_\theta$, residual form, Lagrangian form |
| **Validation** | Energy drift, symplecticity violation, rollout MSE |
| **Robotics Control** | Linearized Dynamics (iLQR Core), $\delta x_{t+1} = A_t \delta x_t + B_t \delta u_t$ |

## Why It's Called Scientific Machine Learning

Regular ML is a blank slate. SciML injects known physics as a structural constraint. It produces scientifically interpretable models where weights map directly to physical quantities, enabling discovery and bounded, predictable hardware deployment.

```

```
