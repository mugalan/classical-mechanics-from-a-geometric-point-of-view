# rigid-body-sim (sims)

This is a compilation of a set of interactive notes and supplementary material for teaching and learning Mechanics.

They have been the source for the undergraduate courses CE1010, ME211, ME320, ME301, ME327, ME518 offered by the Department of Mechanical Engineering at the University of Peradeniya.

**D. H. S. Maithripala, PhD.**  
smaithri@eng.pdn.ac.lk  
https://eng.pdn.ac.lk/ME/People/FacultyProfiles.php?id=6  
https://orcid.org/0000-0002-8200-2696

It also contains lightweight Python utilities for rigid‑body simulations and simple dynamical system demos — with Plotly animations and quaternion/rotation helpers.

> Package import: `import sims` → `sims.RigidBodySim`

For theory and simulation of intrinsic PID control and DEKF for rigid body motion can be found in the repo:

https://github.com/mugalan/intrinsic-rigid-body-control-estimation

---

## ✨ Features

- Skew (hat) matrix: cross-product as a matrix operator.
- Quaternion ⇄ rotation matrix conversions.
- Euler-angle extraction (ZYX convention in this implementation).
- Align $e_3$ to a target direction $R e_3 = \gamma$ via axis–angle.
- Rotate + translate arbitrary vertices.
- Plotly helpers:
  - Orthonormal frame rendering.
  - 3D particle path animation.
  - 2D animated scatter.
  - Flat-shaded animated cube.
- ODE helpers:
  - Linear system model $\dfrac{dX}{dt} = A X$.
  - `simulate_dy_system` wrapper around `scipy.integrate.odeint`.
  - Custom integrators: Euler and RK4 pipeline for a rigid body state.
- Convenience geometry: cube-vertex generator and “simulate a cube” pipeline.

---

## 📦 Installation

### From GitHub (recommended while iterating)
```bash
pip install "git+https://github.com/<you>/<repo>.git#egg=rigid-body-sim"
```

### From a local checkout
```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -e .
```

Python ≥ 3.9. Core deps: `numpy`, `scipy`, `sympy`, `plotly`, `pandas`, `matplotlib` (plus `simdkalman` pinned per your spec).

---

## 🔎 Quickstart

```python
import numpy as np
import sims

rb = sims.RigidBodySim()

# 1) Hat (skew) matrix
w = np.array([1.0, 2.0, 3.0])
print(rb.hat_matrix(w))
# [[ 0. -3.  2.]
#  [ 3.  0. -1.]
#  [-2.  1.  0.]]

# 2) Quaternion from axis–angle (90° about z)
q = rb.q_from_axis_angles(np.pi/2, [0, 0, 1])
R = rb.r_from_quaternions(q)
print(R.round(4))
# [[ 0. -1.  0.]
#  [ 1.  0.  0.]
#  [ 0.  0.  1.]]

# 3) Euler angles from rotation matrix (φ, θ, ψ) assuming ZYX logic in this method
phi, theta, psi = rb.rotation_matrix_2_euler_angles(R)
print(phi, theta, psi)
```

---

## 🎯 Aligning $e_3$ to a target direction $\gamma$

Given a unit directionr $\gamma$, build $R\in SO(3)$ such that $R e_3 = \gamma$:

```python
gamma = np.array([1/np.sqrt(2), 0.0, 1/np.sqrt(2)])
R = rb.re3_equals_gamma(gamma)
assert np.allclose(R @ np.array([0,0,1.0]), gamma)
```

---

## 🧊 Rotate + translate vertices (cube example)

```python
# Define a cube and center offsets (dimensions and offsets in your class API)
cube = {'l': 2.0, 'w': 2.0, 'h': 2.0, 'xp': 1.0, 'yp': 1.0, 'zp': 1.0}
X = rb.cube_vertices(cube)      # [X, Y, Z], each with 8 coords
R = np.eye(3)                   # no rotation
o = np.array([1.0, 2.0, 0.5])   # translation
XT = rb.rotate_and_translate(np.array(X), R, o)  # 3x8 transformed
```

---

## 📈 Visualizations

### 1) Orthonormal frame (Plotly)

```python
import plotly.graph_objects as go
fig = go.Figure()
rb.add_orth_norm_frame(fig, o=[0,0,0], R=np.eye(3),
                       axis_range=[(-2,2),(-2,2),(-2,2)],
                       axis_color="blue")
fig.show()
```

### 2) Animate a particle trajectory in 3D

```python
t = np.linspace(0, 6*np.pi, 200)
xx = list(zip(np.cos(t), np.sin(t), t/(2*np.pi)))   # helix
fig = rb.animate_particle_motion(xx,
                                 axis_range=[(-1.5,1.5),(-1.5,1.5),(-0.5,4)],
                                 fig_title="Helical Motion")
# fig.show() is already called inside animate_particle_motion
```

### 3) 2D animated scatter

```python
x = np.linspace(0, 2*np.pi, 100)
YY = np.array([np.sin(x + phase) for phase in np.linspace(0, 2*np.pi, 60)])
fig = rb.animate_2D_scatter_plot(x, YY, "x", "sin(x+phase)", "Sine Wave Animation")
fig.show()
```

### 4) Animated cube (flat shading)

```python
# Create a slow rotation about z
angles = np.linspace(0, 2*np.pi, 60)
cube_geom = {'l': 2.0, 'w': 2.0, 'h': 2.0, 'xp': 1.0, 'yp': 1.0, 'zp': 1.0}
base = np.array(rb.cube_vertices(cube_geom))

cube_frames = []
for th in angles:
    qz = rb.q_from_axis_angles(th, [0,0,1])
    Rz = rb.r_from_quaternions(qz)
    XT = rb.rotate_and_translate(base, Rz, [0,0,0])
    cube_frames.append([XT.tolist()])

rb.animated_cube_flat_shading(cube_frames, "Rotating Cube")
```

---

## 🧮 ODEs and system simulation

### Linear model `dX/dt = A X`

```python
A = np.array([[0, 1],
              [-1, -0.5]])
x0 = np.array([1.0, 0.0])
t, sol, fig = rb.simulate_dy_system(
    dynamic_system_model=rb.LinearSystemModel,
    t_max=10.0,
    dt=0.05,
    x0=x0,
    sys_para=A,
    fig_title="Second-Order System",
    x_label="time (s)",
    y_label="states"
)
```
> Internally uses `scipy.integrate.odeint` and returns `(t, sol, fig)`.

---

## 🌀 Rigid-body integration (Euler & RK4 pipeline)

Register your physics hooks first:

- `set_external_force_model(fn)` where `fn` is **either**
  - `fn(parameters, X) -> (tau_e, f_e)` **or**
  - `fn(self, parameters, X) -> (tau_e, f_e)`
- `set_actuator(fn)` where `fn` is **either**
  - `fn(parameters, t, X, tau_e, f_e) -> (tau_a, f_a)` **or**
  - `fn(self, parameters, t, X, tau_e, f_e) -> (tau_a, f_a)`

All returned vectors must be length-3. If a hook is not set, the integrators default to zeros for that hook.

The integrators expect a composite state:
```
X = [[R, o], omega, p, Xc]
  where R ∈ R^{3x3}, o ∈ R^3, omega ∈ R^3, p ∈ R^3, and Xc = controller/extra state
```

### Minimal demo (no forces, no control → constant linear/angular momenta)
```python
import numpy as np
import sims

rb = sims.RigidBodySim()

# Zero models (both signatures without `self`; bound methods also supported)
def ext_zero(parameters, X):
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

def act_zero(parameters, t, X, tau_e, f_e):
    return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

rb.set_external_force_model(ext_zero)
rb.set_actuator(act_zero)

# Parameters
M  = 1.0
II = np.diag([0.1, 0.2, 0.3])
params = {"M": M, "II": II}

# Initial state
R0     = np.eye(3)
o0     = np.zeros(3)
omega0 = np.array([0.0, 0.0, 5.0])   # initial angular velocity
doto0  = np.array([0.2, 0.0, 0.0])   # initial linear velocity (p/M)
Xc0    = np.zeros(3)

ICs = [[R0, o0], omega0, doto0, Xc0]

# Integrate (Euler or RK4)
dt, Tmax = 0.02, 1.0
# euler_states = rb.eulers_method(dt, Tmax, params, ICs)
rk_states   = rb.runga_kutta_method(dt, Tmax, params, ICs)

# Turn states into cube frames and animate
cube = {'l': 1.0, 'w': 1.0, 'h': 1.0, 'xp': 0.5, 'yp': 0.5, 'zp': 0.5}
base = np.array(rb.cube_vertices(cube))

frames = []
for X in rk_states:
    R, o = X[0][0], X[0][1]
    XT = rb.rotate_and_translate(base, R, o)
    frames.append([XT.tolist()])

rb.animated_cube_flat_shading(frames, "Rigid Body RK4 Demo")
```

> **Important:** Ensure `eulers_method` calls `self.r_from_quaternions(...)` (remove the `ns`).

---


## 📐 Math refs

- Hat matrix (skew) for $x = [x, y, z]^T$:
  $$
  \begin{align}
    \hat{x} = \begin{bmatrix}
    0 & -z & y \\
    z & 0 & -x \\
    -y & x & 0
    \end{bmatrix}
  \end{align}
  $$

- Quaternion from axis–angle (unit axis $n$, angle $\theta$):
  $q = [\cos\left(\tfrac{\theta}{2}\right),\ \sin\left(\tfrac{\theta}{2}\right)\,n]$

- Rotation from quaternion (scalar–vector split $q = [q_0, w]$):
  $R = I + 2 q_0\,\hat{w} + 2\,\hat{w}^2$

## 🔖 License

MIT © DHS Maithripala
