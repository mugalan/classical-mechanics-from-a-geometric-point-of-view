# rigid-body-sim (sims)

This is a compilation of a set of interactive notes and supplementary material for teaching and learning Mechanics.

They have been the source for the undergraduate courses CE1010, ME211, ME320, ME301, ME327, ME518 offered by the Department of Mechanical Engineering at the University of Peradeniya.

**D. H. S. Maithripala, PhD.**  
smaithri@eng.pdn.ac.lk  
https://eng.pdn.ac.lk/ME/People/FacultyProfiles.php?id=6  
https://orcid.org/0000-0002-8200-2696

It also contains lightweight Python utilities for rigid‑body simulations and simple dynamical system demos — with Plotly animations and quaternion/rotation helpers.

> Package import: `import sims` → `sims.RigidBodySim`

---

## ✨ Features

- Skew (hat) matrix: cross-product as a matrix operator.
- Quaternion ⇄ rotation matrix conversions.
- Euler-angle extraction (ZYX convention in this implementation).
- Align `e3` to a target vector (`R e3 = γ`) via axis–angle.
- Rotate + translate arbitrary vertices.
- Plotly helpers:
  - Orthonormal frame rendering.
  - 3D particle path animation.
  - 2D animated scatter.
  - Flat-shaded animated cube.
- ODE helpers:
  - Linear system model `dX/dt = A X`.
  - `simulate_dy_system` wrapper around `scipy.integrate.odeint`.
  - Custom integrators: Euler and RK4 pipeline for a rigid body state.
- Convenience geometry: cube-vertex generator and “simulate a cube” pipeline.

> Heads-up: the original class had a duplicate `add_orth_norm_frame` and a likely typo `r_from_quaternionsns` → `r_from_quaternions`. The examples below assume those are fixed.

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

## 🎯 Aligning e₃ to a target direction γ

Given a unit vector `γ`, build `R` such that `R e3 = γ`:

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

## 🧭 Discrete Intrinsic EKF (DEKF) on SO(3)

This package includes a lightweight **discrete, right-invariant EKF** for attitude using two stacked direction measurements (e.g., magnetometer + gravity). It runs directly on **SO(3)** and updates the estimate by **right-multiplying** a small exponential.

### Model & Notation (SO(3))

- State: attitude \(R_k \in \mathrm{SO}(3)\)
- Propagation (body rate \(\Omega_{k-1}\) in rad/s):
  \[
  \widetilde R_k^- \,=\, \widetilde R_{k-1}\,\exp\!\big(\Delta T\,\widehat{\Omega}_{k-1}\big)
  \]
- Measurement (two known inertial directions \(e_1,e_3\in\mathbb S^2\)):
  \[
  y_k \,=\, 
  \begin{bmatrix}
  R_k^{\!\top} e_1\\[2pt]
  R_k^{\!\top} e_3
  \end{bmatrix}\in\mathbb R^{6},\qquad
  \hat y_k^- \,=\,
  \begin{bmatrix}
  \widetilde R_k^{-\top} e_1\\[2pt]
  \widetilde R_k^{-\top} e_3
  \end{bmatrix}
  \]
- Right-invariant residual (in tangent):
  \[
  r_k \,\approx\,
  \begin{bmatrix}
  \hat y_{1,k}^- \times y_{1,k}\\
  \hat y_{3,k}^- \times y_{3,k}
  \end{bmatrix}\in\mathbb R^{6}
  \]

### Linearizations

Let \(\operatorname{ad}_\omega(\cdot)=\widehat\omega(\cdot)=\omega\times (\cdot)\). For a small step \(\Delta T\):

\[
A_{k-1}=\exp(-\Delta T\,\widehat{\Omega}_{k-1}),\quad
G_{k-1}=\sqrt{\Delta T}\,I_3\ \text{(or } \sqrt{\Delta T}\,\psi(\operatorname{ad}_{\Delta T\Omega})\text{ for higher fidelity)},
\]
\[
H_k =
\begin{bmatrix}
-\widehat{\widetilde R_k^{-\top} e_1}\\
-\widehat{\widetilde R_k^{-\top} e_3}
\end{bmatrix}\in\mathbb R^{6\times 3}.
\]

### Filter Recursions

- **Predict**
  \[
  P_k^- = A_{k-1} P_{k-1} A_{k-1}^{\!\top} + G_{k-1} \,\Sigma_q\, G_{k-1}^{\!\top}
  \]
- **Gain / Update**
  \[
  K_k = P_k^- H_k^{\!\top}\big(H_k P_k^- H_k^{\!\top}+\Sigma_m\big)^{-1},\quad
  \widetilde R_k=\widetilde R_k^-\,\exp\!\big(\Delta T\,K_k\,r_k\big),
  \]
  \[
  P_k = (I-K_k H_k)P_k^- (I-K_k H_k)^{\!\top} + K_k \Sigma_m K_k^{\!\top}\quad\text{(Joseph form)}
  \]

### Typical Noise Settings (tune to hardware)

- Gyro white noise \(\sigma_\omega\)[rad/s]: **1e-4 … 2e-2** (good MEMS ~1e-3 … 5e-3)  
- Direction component noise \(\sigma_{\text{dir}}\) [-]: **5e-3 … 1e-1** (≈0.3°…5.7° per axis)  
- Initial attitude 1-σ (deg): **1° … 20°**

Recommended covariances:
```python
sigma_omega = 3e-3                      # rad/s
sigma_dir   = 2e-2                      # unit-vector component stdev
Sigma_q = (sigma_omega**2) * np.eye(3)  # process
Sigma_m = (sigma_dir**2)   * np.eye(6)  # measurement (stacked two vectors)
Sigma_p0 = (np.deg2rad(10.0)**2) * np.eye(3)  # prior attitude variance
```

### Quick Online Example

```python
import numpy as np
import sims

rb = sims.RigidBodySim()

def sensor(R, omega_body, sigma_omega=3e-3, sigma_dir=2e-2, rng=None):
    rng = rng or np.random.default_rng(0)
    e1 = np.array([1.,0.,0.]); e3 = np.array([0.,0.,1.])
    n3 = lambda s: rng.normal(0.0, s, size=3) if np.isscalar(s) else rng.normal(0.0, s, size=3)
    Omega_meas = omega_body + n3(sigma_omega)
    A_n_meas   = R.T @ e1 + n3(sigma_dir); A_n_meas /= max(np.linalg.norm(A_n_meas), 1e-12)
    A_g_meas   = R.T @ e3 + n3(sigma_dir); A_g_meas /= max(np.linalg.norm(A_g_meas), 1e-12)
    return Omega_meas, A_n_meas, A_g_meas

def innovation(R_minus, A_n, A_g):
    y1 = R_minus.T @ np.array([1.,0.,0.])
    y3 = R_minus.T @ np.array([0.,0.,1.])
    r1 = np.cross(y1, A_n); r3 = np.cross(y3, A_g)
    return np.hstack([r1, r3]).reshape(-1,1)

rb.set_sensor(sensor)
rb.set_KF_innovation(innovation)

sigma_omega, sigma_dir = 3e-3, 2e-2
Sigma_q = (sigma_omega**2) * np.eye(3)
Sigma_m = (sigma_dir**2)   * np.eye(6)
Sigma_p0 = (np.deg2rad(10.0)**2) * np.eye(3)

dt = 0.01
steps = 1000
omega_body_true = np.array([0.25, -0.05, 0.15])
R_true = np.eye(3)
R_hat  = np.eye(3)
P_hat  = Sigma_p0.copy()

for k in range(steps):
    R_true = R_true @ rb.exp_map(dt * omega_body_true)
    Omega_meas, A_n_meas, A_g_meas = rb.sensor(R_true, omega_body_true, sigma_omega, sigma_dir)
    R_hat, P_hat, K, H, S = rb.predict_update_attitude(
        DeltaT=dt, Omega_km1=Omega_meas,
        R_previous=R_hat, P_previous=P_hat,
        Sigma_q=Sigma_q, Sigma_m=Sigma_m,
        A_n_meas=A_n_meas, A_g_meas=A_g_meas
    )
```

### Offline Example (pre-generated trajectory + covariance plot)

```python
import numpy as np, plotly.graph_objects as go

def angle_from_R(R):
    c = float(np.clip((np.trace(R)-1.0)*0.5, -1.0, 1.0))
    return np.arccos(c)

# trajectory: list of states from rb.runga_kutta_method
# Xk = [[R_k, o_k], omega_spatial_k, p_k, Xc_k]; DeltaT is the integrator step

sigma_omega, sigma_dir = 1e-3, 2e-2
Sigma_q = (sigma_omega**2) * np.eye(3)
Sigma_m = (sigma_dir**2)   * np.eye(6)
Sigma_p0 = (np.deg2rad(10.0)**2) * np.eye(3)

mr.set_sensor(sensor)
mr.set_KF_innovation(innovation)

R_hat = trajectory[0][0][0]
P_hat = Sigma_p0.copy()

N = len(trajectory)
t = np.arange(N) * DeltaT
err_deg, trace_err, lam_max, ang_1sig = [], [], [], []

for k in range(N):
    R_true        = trajectory[k][0][0]
    omega_spatial = trajectory[k][1]
    omega_body    = R_true.T @ omega_spatial
    Omega_meas, A_n_meas, A_g_meas = mr.sensor(R_true, omega_body, sigma_omega, sigma_dir)
    R_hat, P_hat, K, H, S = mr.predict_update_attitude(
        DeltaT=DeltaT, Omega_km1=Omega_meas,
        R_previous=R_hat, P_previous=P_hat,
        Sigma_q=Sigma_q, Sigma_m=Sigma_m,
        A_n_meas=A_n_meas, A_g_meas=A_g_meas
    )
    R_err = R_hat.T @ R_true
    err_deg.append(np.degrees(angle_from_R(R_err)))
    trace_err.append(3.0 - np.trace(R_err))
    w = np.linalg.eigvalsh(0.5*(P_hat+P_hat.T))
    lam = float(w[-1]); lam_max.append(lam)
    ang_1sig.append(np.degrees(np.sqrt(max(lam,0.0))))

print(f"Final attitude error: {err_deg[-1]:.3f} deg,  √λ_max(P): {ang_1sig[-1]:.3f} deg")

fig_cov = go.Figure()
fig_cov.add_trace(go.Scatter(x=t, y=lam_max, mode="lines", name="λ_max(P) [rad²]"))
fig_cov.add_trace(go.Scatter(x=t, y=ang_1sig, mode="lines", name="√λ_max(P) [deg]", yaxis="y2"))
fig_cov.update_layout(
    title="DEKF covariance magnitude (offline)",
    xaxis_title="Time (s)",
    yaxis=dict(title="Largest eigenvalue λ_max(P) [rad²]"),
    yaxis2=dict(title="1σ angle √λ_max(P) [deg]", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
)
fig_cov.show()
```

**Tips**
- Ensure both the generator and DEKF use **body** rate for \(\dot R = R\,\widehat{\Omega}_{\text{body}}\). If your trajectory stores **spatial** \(\omega\), convert via \(\Omega_{\text{body}} = R^\top \omega_{\text{spatial}}\).
- If “error drifts while P shrinks,” check: (i) error matrix `R_err = R_hat.T @ R_true`, (ii) RK4 stage timing/weights, and (iii) signs in `H` and residual.

---

## 📐 Math refs

- Hat matrix (skew) for `x = [x, y, z]^T`:
  \[
    \hat{x} = \begin{bmatrix}
    0 & -z & y \\
    z & 0 & -x \\
    -y & x & 0
    \end{bmatrix}
  \]

- Quaternion from axis–angle (unit axis `n`, angle `θ`):
  \[
    q = [\cos(\tfrac{\theta}{2}),\ \sin(\tfrac{\theta}{2})\,n]
  \]

- Rotation from quaternion (scalar–vector split `q = [q_0, w]`):
  \[
    R = I + 2 q_0\,\hat{w} + 2\,\hat{w}^2
  \]

## 🔖 License

MIT © DHS Maithripala
