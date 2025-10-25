import numpy as np
import math
from numpy import linalg


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class KFShapeError(ValueError):
    pass

class KFValueError(ValueError):
    pass

class LinearKF:
    """
    Linear-Gaussian Kalman Filter utilities with explicit predict (A_k, G_k, Q_k-1)
    and measurement update (H_k, R_k).

    Process model (time-varying allowed):
        x_k = A_k x_{k-1} + G_k w_{k-1},       w_{k-1} ~ N(0, Q_{k-1})

    Measurement model:
        y_k = H_k x_k + z_k,                   z_k ~ N(0, R_k)

    Notation in code:
        m_prev   : E[x_{k-1}]            (n_prev,)
        P_prev   : Cov[x_{k-1}]          (n_prev, n_prev)
        A        : A_k                    (n, n_prev)
        G        : G_k (optional)         (n, r)      [default: Identity on state, i.e. G=I_n]
        Q        : Q_{k-1}                (r, r) if G given; else (n, n) if G is None
        m_pred   : E[x_k | y_{1:k-1}]     (n,)
        P_pred   : Cov[x_k | y_{1:k-1}]   (n, n)

        H        : H_k                    (p, n)
        R        : R_k                    (p, p)
        y        : y_k                    (p,)
        v        : innovation             (p,)
        S        : innovation covariance  (p, p)
        K        : Kalman gain            (n, p)

    Covariance updates:
        Predict:  P^- = A P A^T + (G Q G^T)   [or Q directly if G is None]
        Update:   K  = P^- H^T (H P^- H^T + R)^{-1}
                  m  = m^- + K (y - H m^-)
                  P  = (I - K H) P^- (I - K H)^T + K R K^T   (Joseph form if enabled)
    """

    def __init__(self, *, use_joseph: bool = True, symmetrize: bool = True, atol_sym: float = 1e-10):
        """
        Parameters
        ----------
        use_joseph : bool
            Use Joseph-stabilized covariance update (default True).
        symmetrize : bool
            Force symmetry of covariances at the end via (P + P^T)/2 (default True).
        atol_sym : float
            Tolerance for symmetry checks.
        """
        self.use_joseph = use_joseph
        self.symmetrize = symmetrize
        self.atol_sym = atol_sym

    # ---------- public API ----------

    def predict(self, m_prev, P_prev, A, Q, G=None):
        """
        One-step prediction using x_k = A x_{k-1} + G w_{k-1},  w ~ N(0, Q).

        Parameters
        ----------
        m_prev : (n_prev,)
        P_prev : (n_prev, n_prev)
        A      : (n, n_prev)
        Q      : if G is None: (n, n)
                 else:          (r, r) where G is (n, r)
        G      : optional (n, r). If None, prediction uses P^- = A P A^T + Q (treats Q in state space)

        Returns
        -------
        m_pred : (n,)
        P_pred : (n, n)
        """
        m_prev = self._as_1d(m_prev, "m_prev")
        P_prev = self._as_2d(P_prev, "P_prev")
        A      = self._as_2d(A, "A")
        Q      = self._as_2d(Q, "Q")
        n_prev = m_prev.shape[0]

        # shape checks
        if P_prev.shape != (n_prev, n_prev):
            raise KFShapeError(f"`P_prev` must be {(n_prev, n_prev)}; got {P_prev.shape}.")
        if A.shape[1] != n_prev:
            raise KFShapeError(f"`A` must have second dim {n_prev}; got {A.shape}.")

        n = A.shape[0]
        self._assert_symmetric(P_prev, "P_prev", atol=self.atol_sym)
        self._assert_symmetric(Q, "Q", atol=self.atol_sym)
        _ = self._assert_spd(Q, "Q")  # Q should be SPD (or at least PSD; we enforce SPD for numerical safety)

        if G is None:
            # Q is in state coordinates (n x n)
            if Q.shape != (n, n):
                raise KFShapeError(f"With G=None, `Q` must be shape {(n, n)}; got {Q.shape}.")
            m_pred = A @ m_prev
            P_pred = A @ P_prev @ A.T + Q
        else:
            G = self._as_2d(G, "G")
            if G.shape[0] != n:
                raise KFShapeError(f"`G` must have {n} rows to match A's output dim n; got {G.shape}.")
            r = G.shape[1]
            if Q.shape != (r, r):
                raise KFShapeError(f"With G of shape {(n, r)}, `Q` must be {(r, r)}; got {Q.shape}.")
            m_pred = A @ m_prev
            P_pred = A @ P_prev @ A.T + G @ Q @ G.T

        if self.symmetrize:
            P_pred = 0.5 * (P_pred + P_pred.T)

        return m_pred, P_pred

    def measurement_update(self, m_pred, P_pred, H, y, R):
        """
        Compute K, m, P given m^- (=m_pred), P^- (=P_pred), H, y, R.

        Returns
        -------
        m_upd : (n,)
        P_upd : (n,n)
        K     : (n,p)
        S     : (p,p)   innovation covariance = H P^- H^T + R
        """
        m_pred, P_pred, H, y, R, n, p = self._validate_shapes(m_pred, P_pred, H, y, R)

        # Symmetry checks and SPD requirements
        self._assert_symmetric(P_pred, "P_pred", atol=self.atol_sym)
        self._assert_symmetric(R, "R", atol=self.atol_sym)
        _ = self._assert_spd(R, "R")  # R must be SPD

        # Innovation covariance S = H P^- H^T + R
        HP = H @ P_pred               # (p,n)
        S = HP @ H.T + R              # (p,p)
        try:
            Ls = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as e:
            raise KFValueError(
                f"Innovation covariance `S` is not SPD. "
                f"Check H, P_pred, and R. Cholesky failed: {e}"
            )

        # Gain K = P^- H^T S^{-1} without explicit inverse
        U = np.linalg.solve(Ls, HP)   # (p,n)
        X = np.linalg.solve(Ls.T, U)  # (p,n)
        K = X.T                       # (n,p)

        # Residual
        v = y - H @ m_pred            # (p,)

        # Updated mean
        m_upd = m_pred + K @ v        # (n,)

        # Updated covariance
        I = np.eye(n)
        if self.use_joseph:
            I_KH = (I - K @ H)
            P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        else:
            P_upd = (I - K @ H) @ P_pred

        if self.symmetrize:
            P_upd = 0.5 * (P_upd + P_upd.T)

        return m_upd, P_upd, K, S

    def innovation(self, m_pred, P_pred, H, R, y=None):
        """
        Compute innovation covariance S = H P^- H^T + R and (optionally) residual v.

        Returns
        -------
        (v, S) if y is given; otherwise just S.
        """
        # Validate without using y unless provided
        m_pred = self._as_1d(m_pred, "m_pred")
        P_pred = self._as_2d(P_pred, "P_pred")
        H = self._as_2d(H, "H")
        R = self._as_2d(R, "R")

        n = m_pred.shape[0]
        if P_pred.shape != (n, n):
            raise KFShapeError(f"`P_pred` must have shape {(n, n)}; got {P_pred.shape}.")
        if H.shape[1] != n:
            raise KFShapeError(f"`H` second dim must be {n}; got {H.shape}.")

        p = H.shape[0]
        if R.shape != (p, p):
            raise KFShapeError(f"`R` must have shape {(p, p)}; got {R.shape}.")

        self._assert_symmetric(P_pred, "P_pred", atol=self.atol_sym)
        self._assert_symmetric(R, "R", atol=self.atol_sym)
        _ = self._assert_spd(R, "R")

        S = H @ P_pred @ H.T + R
        if y is None:
            return S
        y = self._as_1d(y, "y")
        if y.shape[0] != p:
            raise KFShapeError(f"`y` length must be {p}; got {y.shape}.")
        v = y - H @ m_pred
        return v, S

    def step(self, m_prev, P_prev, A, Q, H, y, R, G=None):
        """
        Convenience: predict → update in one call.

        Parameters
        ----------
        m_prev, P_prev : prior mean/cov at k-1
        A, Q, G        : process model at k
        H, y, R        : measurement model at k

        Returns
        -------
        m_upd, P_upd, K, S, (m_pred, P_pred)
        """
        m_pred, P_pred = self.predict(m_prev, P_prev, A, Q, G)
        m_upd, P_upd, K, S = self.measurement_update(m_pred, P_pred, H, y, R)
        return m_upd, P_upd, K, S, (m_pred, P_pred)

    def set_options(self, *, use_joseph=None, symmetrize=None, atol_sym=None):
        """Update instance options."""
        if use_joseph is not None:
            self.use_joseph = bool(use_joseph)
        if symmetrize is not None:
            self.symmetrize = bool(symmetrize)
        if atol_sym is not None:
            self.atol_sym = float(atol_sym)

    # ---------- helpers (static) ----------

    @staticmethod
    def _as_1d(x, name):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise KFShapeError(f"`{name}` must be a 1D vector of shape (n,) or (n,1); got {x.shape}.")
        return x

    @staticmethod
    def _as_2d(x, name):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise KFShapeError(f"`{name}` must be a 2D array; got {x.ndim}D with shape {x.shape}.")
        return x

    @staticmethod
    def _assert_symmetric(M, name, atol=1e-10):
        if M.shape[0] != M.shape[1]:
            raise KFShapeError(f"`{name}` must be square; got {M.shape}.")
        if not np.allclose(M, M.T, atol=atol):
            raise KFValueError(f"`{name}` must be symmetric within atol={atol}.")

    @staticmethod
    def _assert_spd(M, name):
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError as e:
            raise KFValueError(f"`{name}` must be symmetric positive definite (SPD). Cholesky failed: {e}")

    # ---------- internal ----------

    def _validate_shapes(self, m_pred, P_pred, H, y, R):
        m_pred = self._as_1d(m_pred, "m_pred")
        y = self._as_1d(y, "y")
        P_pred = self._as_2d(P_pred, "P_pred")
        H = self._as_2d(H, "H")
        R = self._as_2d(R, "R")

        n = m_pred.shape[0]
        if P_pred.shape != (n, n):
            raise KFShapeError(f"`P_pred` must have shape {(n, n)} to match `m_pred`; got {P_pred.shape}.")
        if H.shape[1] != n:
            raise KFShapeError(f"Second dim of `H` must equal len(m_pred)={n}; got H.shape={H.shape}.")
        p = H.shape[0]
        if y.shape[0] != p:
            raise KFShapeError(f"`y` length must match number of rows in `H` (p={p}); got y.shape={y.shape}.")
        if R.shape != (p, p):
            raise KFShapeError(f"`R` must have shape {(p, p)}; got {R.shape}.")
        return m_pred, P_pred, H, y, R, n, p


class LDSShapeError(ValueError): ...
class LDSValueError(ValueError): ...

class LinearGaussianSystemSyms:
    """
    Discrete-time linear Gaussian system (time-invariant here):
        x_k = A x_{k-1} + G w_{k-1},   w_{k-1} ~ N(0, Σ_p)
        y_k = H x_k       + z_k,       z_k     ~ N(0, Σ_m)

    Parameters
    ----------
    A : (n,n)
    H : (p,n)
    Sigma_p :  (n,n) if G is None   OR   (r,r) if G is provided (G has shape (n,r))
    Sigma_m : (p,p)
    G : None or (n,r), optional
    x0 : (n,), optional
    rng : np.random.Generator or seed
    """

    def __init__(self, A, H, Sigma_p, Sigma_m, x0=None, rng=None, G=None):
        self.A = self._as_2d(A, "A")
        self.H = self._as_2d(H, "H")
        self.G = None if G is None else self._as_2d(G, "G")
        self.Sigma_p = self._as_2d(Sigma_p, "Sigma_p")
        self.Sigma_m = self._as_2d(Sigma_m, "Sigma_m")

        n = self.A.shape[0]
        p = self.H.shape[0]

        # Shape checks
        if self.A.shape != (n, n):
            raise LDSShapeError(f"`A` must be (n,n); got {self.A.shape}.")
        if self.H.shape[1] != n:
            raise LDSShapeError(f"`H` must be (p,n) with n={n}; got {self.H.shape}.")

        if self.G is None:
            # Σ_p is already in state space
            if self.Sigma_p.shape != (n, n):
                raise LDSShapeError(f"With G=None, `Sigma_p` must be (n,n)={(n,n)}; got {self.Sigma_p.shape}.")
        else:
            # Σ_p is in w-space (r,r)
            r = self.G.shape[1]
            if self.Sigma_p.shape != (r, r):
                raise LDSShapeError(f"With G of shape {(n,r)}, `Sigma_p` must be (r,r)={(r,r)}; got {self.Sigma_p.shape}.")

        if self.Sigma_m.shape != (p, p):
            raise LDSShapeError(f"`Sigma_m` must be (p,p); got {self.Sigma_m.shape}.")

        # Symmetry + SPD (for sampling and numerical sanity)
        self._assert_symmetric(self.Sigma_p, "Sigma_p")
        self._assert_symmetric(self.Sigma_m, "Sigma_m")
        self.Lw = self._assert_spd(self.Sigma_p, "Sigma_p")  # Cholesky for w (or state if G=None)
        self.Lm = self._assert_spd(self.Sigma_m, "Sigma_m")

        self.n, self.p = n, p
        self.r = None if self.G is None else self.G.shape[1]
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.x = self._as_1d(x0, "x0") if x0 is not None else np.zeros(n)

    # ---------- convenience ----------

    def process_cov_state(self):
        """Return Q_state = Cov(noise in state space) = Σ_p (if G=None) else G Σ_p G^T."""
        if self.G is None:
            return self.Sigma_p
        return self.G @ self.Sigma_p @ self.G.T

    # ---------- stepping ----------

    def step_state(self):
        """x_k = A x_{k-1} + (G w_{k-1} or w_{k-1} if G=None). Returns x_k."""
        if self.G is None:
            # w is in state space (n,)
            w = self.Lw @ self.rng.standard_normal(self.n)
            self.x = self.A @ self.x + w
        else:
            # w is in w-space (r,)
            w = self.Lw @ self.rng.standard_normal(self.r)
            self.x = self.A @ self.x + self.G @ w
        return self.x

    def step_measurement(self, x=None):
        """y_k = H x_k + z_k  (returns y_k)"""
        xk = self._as_1d(x, "x") if x is not None else self.x
        z = self.Lm @ self.rng.standard_normal(self.p)
        return self.H @ xk + z

    def step(self):
        """Advance one step and return (x_k, y_k)."""
        xk = self.step_state()
        yk = self.step_measurement(xk)
        return xk, yk

    def simulate(self, T):
        """Run T steps; returns X (T,n) and Y (T,p)."""
        X = np.zeros((T, self.n))
        Y = np.zeros((T, self.p))
        for k in range(T):
            X[k], Y[k] = self.step()
        return X, Y

    # ---------- helpers ----------
    @staticmethod
    def _as_2d(M, name):
        M = np.asarray(M, dtype=float)
        if M.ndim != 2:
            raise LDSShapeError(f"`{name}` must be 2D; got shape {M.shape}.")
        return M

    @staticmethod
    def _as_1d(v, name):
        v = np.asarray(v, dtype=float)
        if v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        if v.ndim != 1:
            raise LDSShapeError(f"`{name}` must be 1D; got shape {v.shape}.")
        return v

    @staticmethod
    def _assert_symmetric(M, name, atol=1e-12):
        if M.shape[0] != M.shape[1]:
            raise LDSShapeError(f"`{name}` must be square; got {M.shape}.")
        if not np.allclose(M, M.T, atol=atol):
            raise LDSValueError(f"`{name}` must be symmetric within atol={atol}.")

    @staticmethod
    def _assert_spd(M, name):
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError as e:
            raise LDSValueError(f"`{name}` must be symmetric positive definite (SPD). Cholesky failed: {e}")

    # ---------- visuals / demos (updated to use G when present) ----------

    def animate_measurement_gaussians_scalar(
        self,
        *,
        T: int | None = None,
        Y: np.ndarray | None = None,
        m0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        kf=None,
        frame_ms: int = 120,
        auto_play: bool = False,
        save_html_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
        component_label: str = "y"
    ):
        if self.p != 1:
            raise ValueError(f"This animation requires scalar measurements (p=1); got p={self.p}.")

        A = np.asarray(self.A, float)
        H = np.asarray(self.H, float).reshape(1, -1)    # (1,n)
        Qs = np.asarray(self.process_cov_state(), float)
        Rm = np.asarray(self.Sigma_m, float)
        R_scalar = float(np.atleast_2d(Rm)[0, 0])

        n = self.n

        if Y is None:
            if T is None:
                raise ValueError("Provide either Y or T.")
            _, Y_sim = self.simulate(T)
            Y = np.asarray(Y_sim, float).reshape(-1)
        else:
            Y = np.asarray(Y, float).reshape(-1)
            T = Y.shape[0]

        m = np.zeros(n) if m0 is None else np.asarray(m0, float).reshape(-1)
        if m.shape[0] != n:
            raise ValueError(f"`m0` must have length n={n}; got {m.shape}.")
        P = (1e2 * np.eye(n)) if P0 is None else np.asarray(P0, float)
        if P.shape != (n, n):
            raise ValueError(f"`P0` must be (n,n)=({n},{n}); got {P.shape}.")

        if kf is None:
            kf = LinearKF(use_joseph=True, symmetrize=True)

        mu_pred = np.zeros(T)
        sig_pred = np.zeros(T)
        mu_post = np.zeros(T)
        sig_post = np.zeros(T)

        for k in range(T):
            # Predict
            m_pred = A @ m
            P_pred = A @ P @ A.T + Qs

            # Predictive y ~ N(H m^-, H P^- H^T + R)
            mu_pred[k] = (H @ m_pred).item()
            S_pred = (H @ P_pred @ H.T).item() + R_scalar
            sig_pred[k] = float(np.sqrt(S_pred)) if S_pred > 0 else 0.0

            # Update with robust KF
            m, P, _, _ = kf.measurement_update(m_pred, P_pred, H, np.array([Y[k]]), np.array([[R_scalar]]))

            # Posterior-predictive y ~ N(H m, H P H^T + R)
            mu_post[k] = (H @ m).item()
            S_post = (H @ P @ H.T).item() + R_scalar
            sig_post[k] = float(np.sqrt(S_post)) if S_post > 0 else 0.0

        # ---- plotting (unchanged logic) ----
        def _gauss_pdf(x, mu, sigma):
            if sigma <= 0:
                return np.zeros_like(x)
            return (1.0 / (np.sqrt(2.0*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

        pred_lo = np.min(mu_pred - 4.0 * sig_pred)
        pred_hi = np.max(mu_pred + 4.0 * sig_pred)
        post_lo = np.min(mu_post - 4.0 * sig_post)
        post_hi = np.max(mu_post + 4.0 * sig_post)
        y_lo = float(np.min(Y)) - 1e-6
        y_hi = float(np.max(Y)) + 1e-6

        x_min = float(np.min([pred_lo, post_lo, y_lo]))
        x_max = float(np.max([pred_hi, post_hi, y_hi]))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = -3.0, 3.0

        xgrid = np.linspace(x_min, x_max, 700)

        frames = []
        for k in range(T):
            y_pred_pdf = _gauss_pdf(xgrid, mu_pred[k], sig_pred[k])
            y_post_pdf = _gauss_pdf(xgrid, mu_post[k], sig_post[k])
            ymax = float(max(y_pred_pdf.max(initial=0.0), y_post_pdf.max(initial=0.0)))

            frames.append(
                go.Frame(
                    name=str(k),
                    data=[
                        go.Scatter(x=xgrid, y=y_pred_pdf, mode="lines",
                                   name="Predictive p(y_k|Y_{k-1})", showlegend=False),
                        go.Scatter(x=xgrid, y=y_post_pdf, mode="lines",
                                   name="Posterior p(y_k|Y_k)", showlegend=False),
                        go.Scatter(x=[Y[k]], y=[ymax*0.9], mode="markers",
                                   name=f"observed {component_label}_k", showlegend=False, marker=dict(size=8)),
                    ],
                    layout=go.Layout(
                        annotations=[
                            dict(
                                x=0.98, y=0.95, xref="paper", yref="paper",
                                xanchor="right", yanchor="top",
                                text=(f"k={k} | μ⁻={mu_pred[k]:.3f}, σ⁻={sig_pred[k]:.3f} "
                                      f"| μ={mu_post[k]:.3f}, σ={sig_post[k]:.3f}"),
                                showarrow=False, font=dict(size=12)
                            )
                        ]
                    )
                )
            )

        y_pred_pdf0 = _gauss_pdf(xgrid, mu_pred[0], sig_pred[0])
        y_post_pdf0 = _gauss_pdf(xgrid, mu_post[0], sig_post[0])
        ymax0 = float(max(y_pred_pdf0.max(initial=0.0), y_post_pdf0.max(initial=0.0)))

        fig = go.Figure(
            data=[
                go.Scatter(x=xgrid, y=y_pred_pdf0, mode="lines", name="Predictive p(y_k|Y_{k-1})"),
                go.Scatter(x=xgrid, y=y_post_pdf0, mode="lines", name="Posterior p(y_k|Y_k)"),
                go.Scatter(x=[Y[0]], y=[ymax0*0.9], mode="markers",
                           name=f"observed {component_label}_k", marker=dict(size=8)),
            ],
            layout=go.Layout(
                title="Scalar KF: Predictive vs Posterior-Predictive Measurement Gaussians",
                xaxis_title=component_label,
                yaxis_title="density",
                template="plotly_white",
                legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
                margin=dict(r=160, l=60, t=60, b=80),
                updatemenus=[
                    dict(
                        type="buttons", direction="left",
                        x=0.0, y=1.15, xanchor="left", yanchor="top",
                        buttons=[
                            dict(label="Play", method="animate",
                                 args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                              "transition": {"duration": 0},
                                              "fromcurrent": True}]),
                            dict(label="Pause", method="animate",
                                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}]),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        active=0, x=0.05, y=-0.12, xanchor="left", yanchor="top",
                        len=0.9,
                        currentvalue=dict(prefix="k = ", visible=True, xanchor="right"),
                        steps=[dict(method="animate",
                                    args=[[str(k)], {"mode": "immediate",
                                                     "frame": {"duration": 0, "redraw": True},
                                                     "transition": {"duration": 0}}],
                                    label=str(k)) for k in range(T)]
                    )
                ],
            ),
            frames=frames
        )

        if save_html_path:
            fig.write_html(save_html_path, include_plotlyjs="cdn", auto_play=auto_play)
        if show:
            fig.show()
        if return_fig:
            return fig

    def plot_y(self, Y, *, nbins=40, component_labels=None,
               curve_samples=500, show=True, return_fig=False):
        """
        Plot all measurement components y[:, j] together in one figure as scatter traces.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise LDSShapeError(f"`Y` must be 2D with shape (T, p); got {Y.shape}.")
        T, p = Y.shape
        if p != self.p:
            raise LDSShapeError(f"`Y` second dim (p={p}) must match system p={self.p}.")

        labels = component_labels or [f"y[{j}]" for j in range(p)]
        t = np.arange(T)

        fig = go.Figure()
        for j in range(p):
            yj = Y[:, j]
            mask = np.isfinite(yj)
            if not np.any(mask):
                continue
            fig.add_trace(
                go.Scatter(
                    x=t[mask], y=yj[mask],
                    mode="lines+markers",
                    name=labels[j],
                    marker=dict(size=5),
                    line=dict(width=1),
                    hovertemplate="k=%{x}<br>y=%{y:.4g}<extra>" + labels[j] + "</extra>",
                )
            )
        fig.update_layout(
            title="Measurement Traces (all y components)",
            template="plotly_white",
            height=420 if p <= 3 else 480,
            legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
            margin=dict(r=160),
        )
        fig.update_xaxes(title_text="time step (k)")
        fig.update_yaxes(title_text="y components")

        if show:
            fig.show()
        if return_fig:
            return fig

    def filter_with_kf_and_plot(
        self,
        *,
        T: int | None = None,
        Y: np.ndarray | None = None,
        m0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        kf=None,  # instance of LinearKF; if None, a default LinearKF() is used
        component_labels: list[str] | None = None,
        nbins: int = 40,            # kept for signature compatibility (unused)
        curve_samples: int = 500,
        show: bool = True,
        return_fig: bool = False,
    ):
        """
        Run a Kalman filter over provided (or simulated) measurements and plot:
          (1) y_k (noisy) vs filtered estimate H m_k  [time series]
          (2) residual Gaussians ONLY (no histograms)
        """
        A = np.asarray(self.A, float)
        H = np.asarray(self.H, float)
        Q = np.asarray(self.Sigma_p, float)   # w-space if G is set; state-space otherwise
        R = np.asarray(self.Sigma_m, float)
        n, p = self.n, self.p

        if Y is None:
            if T is None:
                raise ValueError("Provide either Y or T.")
            _, Y = self.simulate(T)
        else:
            Y = np.asarray(Y, float)
            if Y.ndim != 2 or Y.shape[1] != p:
                raise ValueError(f"`Y` must have shape (T, p={p}); got {Y.shape}.")
            T = Y.shape[0]

        m = np.zeros(n) if m0 is None else np.asarray(m0, float).reshape(-1)
        if m.shape[0] != n:
            raise ValueError(f"`m0` must have length n={n}; got {m.shape}.")
        P = (1e2 * np.eye(n)) if P0 is None else np.asarray(P0, float)
        if P.shape != (n, n):
            raise ValueError(f"`P0` must be shape (n,n)=({n},{n}); got {P.shape}.")

        if kf is None:
            kf = LinearKF(use_joseph=True, symmetrize=True)

        # ----- filtering loop -----
        M = np.zeros((T, n))
        Yhat = np.zeros((T, p))

        for k in range(T):
            # Predict (uses G if present; otherwise treats Q in state space)
            m_pred, P_pred = kf.predict(m, P, A, Q, G=self.G)

            # Update
            m, P, _, _ = kf.measurement_update(m_pred, P_pred, H, Y[k], R)

            # Store
            M[k] = m
            Yhat[k] = H @ m

        # Residuals (posterior): e_k = y_k - H m_k
        E = Y - Yhat  # shape (T, p)

        # ----- plot (1): time series -----
        labels = component_labels or [f"y[{j}]" for j in range(p)]
        t = np.arange(T)
        fig_ts = make_subplots(rows=p, cols=1, shared_xaxes=True,
                               subplot_titles=[f"{lab}: noisy vs filtered Hm" for lab in labels])

        for j in range(p):
            fig_ts.add_trace(
                go.Scatter(x=t, y=Y[:, j], mode="markers",
                           name=f"{labels[j]} noisy", marker=dict(size=5)),
                row=j+1, col=1
            )
            fig_ts.add_trace(
                go.Scatter(x=t, y=Yhat[:, j], mode="lines",
                           name=f"{labels[j]} filtered (H m)"),
                row=j+1, col=1
            )
            fig_ts.update_yaxes(title_text=labels[j], row=j+1, col=1)

        fig_ts.update_xaxes(title_text="time step", row=p, col=1)
        fig_ts.update_layout(
            title="Measurements vs. KF Filtered Estimates (H m_k)",
            template="plotly_white",
            height=max(300, 260 * p),
            legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
            margin=dict(r=140),
        )

        # ----- plot (2): residual Gaussians only -----
        def _gauss_pdf(x, mu, sigma):
            if sigma <= 0:
                return np.zeros_like(x)
            return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        fig_gauss = make_subplots(
            rows=p, cols=1, shared_xaxes=False,
            subplot_titles=[f"{lab} residual Gaussian fits" for lab in labels]
        )

        for j in range(p):
            ej_raw = E[:, j]
            ej = ej_raw[np.isfinite(ej_raw)]
            mu_hat = float(np.mean(ej)) if ej.size else 0.0
            sigma_hat = float(np.std(ej, ddof=0)) if ej.size else 0.0
            sigma_model = float(np.sqrt(max(R[j, j], 0.0)))

            span = max(1e-6, 4.0 * sigma_hat, 4.0 * sigma_model, np.ptp(ej) if ej.size > 1 else 0.0)
            center = mu_hat
            lo, hi = center - 0.5 * span, center + 0.5 * span
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = -1.0, 1.0

            xgrid = np.linspace(lo, hi, curve_samples)
            pdf_hat = _gauss_pdf(xgrid, mu_hat, sigma_hat)
            pdf_model = _gauss_pdf(xgrid, 0.0, sigma_model)

            fig_gauss.add_trace(
                go.Scatter(x=xgrid, y=pdf_hat, mode="lines",
                           name=f"{labels[j]} Gaussian Approx (μ̂={mu_hat:.3g}, σ̂={sigma_hat:.3g})",
                           hovertemplate="x=%{x:.4g}<br>pdf=%{y:.4g}<extra></extra>"),
                row=j+1, col=1
            )
            fig_gauss.add_trace(
                go.Scatter(x=xgrid, y=pdf_model, mode="lines",
                           name=f"{labels[j]} Modeled N(0, R_jj) (σ={sigma_model:.3g})",
                           hovertemplate="x=%{x:.4g}<br>pdf=%{y:.4g}<extra></extra>"),
                row=j+1, col=1
            )

            fig_gauss.update_xaxes(title_text=f"{labels[j]} residual e = y - Hm", row=j+1, col=1)
            fig_gauss.update_yaxes(title_text="density", row=j+1, col=1)

        fig_gauss.update_layout(
            title="Residual Gaussian Fits (Sample MLE vs. Modeled Noise)",
            template="plotly_white",
            height=max(300, 260 * p),
            legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
            margin=dict(r=240),
        )

        if show:
            fig_ts.show()
            fig_gauss.show()

        return (M, Yhat, fig_ts, fig_gauss) if return_fig else (M, Yhat)