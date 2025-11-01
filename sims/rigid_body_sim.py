import numpy as np
import scipy as sp
from numpy import linalg as LA
from scipy.integrate import odeint
import math
from numpy import linalg
import sympy
from copy import deepcopy

from sympy import symbols
from sympy import *
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
init_vprinting(pretty_print=True)

import plotly.graph_objects as go
from typing import Callable, Any


class RigidBodySim:

    def __init__(self):
        self.state =[]
        self.trajectory =[]
        pass


    def hat_matrix(self, x):
        """
        Computes the skew-symmetric matrix (hat matrix) corresponding to a 3 by 1 matrix.

        The hat matrix is a representation of the cross product operation as a matrix.
        It is commonly used in rigid body dynamics, robotics, and control theory to
        transform a vector into its corresponding cross-product operation.

        Args:
            x (numpy.ndarray): A 3-element numpy array representing a 3 by 1 matrix.

        Returns:
            numpy.ndarray: A 3x3 skew-symmetric matrix corresponding to the input 3 by 1 matrix.

        Example:
            Input: x = [1, 2, 3]
            Output:
            [[  0., -3.,  2.],
            [  3.,  0., -1.],
            [ -2.,  1.,  0.]]
        """
        return np.array([
            [0., -x[2], x[1]],
            [x[2],  0., -x[0]],
            [-x[1], x[0],  0.]
        ])

    def _Phi_SO3(self, omega: np.ndarray) -> np.ndarray:
        """
        Left Jacobian Φ(omega) on SO(3).
        Maps perturbations in the body frame.
        Implements Φ(omega) = I + (1 - cosθ)/θ² * hat(omega)
                            + (θ - sinθ)/θ³ * hat(omega)²
        """
        omega = np.asarray(omega, dtype=float).reshape(3,)
        I3 = np.eye(3)
        theta = np.linalg.norm(omega)

        if theta < 1e-8:
            # Small-angle Taylor expansion (accurate up to O(theta^3))
            w_hat = self.hat_matrix(omega)
            return I3 + 0.5 * w_hat + (1.0 / 12.0) * (w_hat @ w_hat)
        else:
            w_hat = self.hat_matrix(omega)
            A = (1 - np.cos(theta)) / (theta**2)
            B = (theta - np.sin(theta)) / (theta**3)
            return I3 + A * w_hat + B * (w_hat @ w_hat)

    def q_from_axis_angles(self, theta, unit_axis):
        """
        Computes a quaternion from a given rotation angle and unit axis.

        A quaternion is a compact representation of rotations in 3D space.
        It is composed of a scalar part and a 3 by 1 matrix part, where the scalar represents the
        rotation magnitude, and the 3 by 1 matrix encodes the rotation axis.

        Args:
            theta (float): The rotation angle in radians.
            unit_axis (list or numpy.ndarray): A 3-element unit 3 by 1 matrix representing the axis of rotation.

        Returns:
            numpy.ndarray: A 4-element array representing the quaternion [q0, q1, q2, q3],
                          where q0 is the scalar part, and [q1, q2, q3] is the 3 by 1 matrix part.

        Formula:
            Quaternion (q) = [cos(theta / 2), sin(theta / 2) * unit_axis]

        Example:
            Input: theta = π/2, unit_axis = [0, 0, 1] (z-axis)
            Output: [0.7071, 0.0, 0.0, 0.7071] (represents a 90-degree rotation about the z-axis)

        Notes:
            - Ensure the input `unit_axis` is normalized to avoid incorrect results.
            - Commonly used in 3D transformations and rigid body dynamics.

        """
        # Compute the scalar part of the quaternion (cosine of half the rotation angle)
        scalar_part = np.cos(theta / 2)

        # Compute the vector part of the quaternion (sine of half the rotation angle times the unit axis)
        three_by_one_matrix_part = np.sin(theta / 2) * np.array(unit_axis)

        # Combine scalar and vector parts into a single quaternion
        return np.concatenate(([scalar_part], three_by_one_matrix_part))

    def r_from_quaternions(self, q):
        """
        Computes a rotation matrix from a given quaternion.

        The rotation matrix is a \(3 \times 3\) orthogonal matrix that represents a rotation
        in 3D space. This method utilizes a quaternion to derive the corresponding rotation matrix.

        Args:
            q (numpy.ndarray): A 4-element numpy array representing the quaternion [q0, q1, q2, q3],
                              where q0 is the scalar part, and [q1, q2, q3] forms a \(3 \times 1\) matrix.

        Returns:
            numpy.ndarray: A \(3 \times 3\) rotation matrix corresponding to the input quaternion.

        Formula:
            Rotation Matrix (R) = I + 2 * q0 * hat(w) + 2 * hat(w) @ hat(w)
            - \(I\): Identity matrix (\(3 \times 3\)).
            - \(q0\): Scalar part of the quaternion.
            - \(w\): \(3 \times 1\) matrix (vector part of the quaternion).
            - \(hat(w)\): Skew-symmetric (hat) matrix of \(w\).

        Example:
            Input: q = [0.7071, 0.7071, 0, 0] (90-degree rotation about x-axis)
            Output:
            [[ 1.   0.   0.  ]
            [ 0.   0.  -1.  ]
            [ 0.   1.   0.  ]]

        Notes:
            - Input quaternion \(q\) should be normalized to ensure a valid rotation matrix.
            - The method converts the vector part of the quaternion into a \(3 \times 1\) matrix internally.
        """
        # Extract the scalar part of the quaternion (q0)
        q0 = q[0]

        # Extract the direction part and reshape it into a 3x1 matrix
        w = q[1:] #.reshape((3, 1))

        # Compute the rotation matrix using the quaternion formula
        return (
            np.identity(3) +
            2 * q0 * self.hat_matrix(w) +
            2 * self.hat_matrix(w) @ self.hat_matrix(w)
        )

    def rotation_matrix_2_euler_angles(self, R):
        """
        Converts a \(3 \times 3\) rotation matrix into its corresponding Euler angles.

        Euler angles provide a representation of rotations using three consecutive rotations about
        specified axes. This method extracts the angles (φ, θ, ψ) from the rotation matrix \(R\).

        Args:
            R (numpy.ndarray): A \(3 \times 3\) rotation matrix representing the rotation in 3D space.

        Returns:
            tuple: A tuple of three angles (φ, θ, ψ) in radians:
                - φ (phi): Rotation about the z-axis.
                - θ (theta): Rotation about the y-axis.
                - ψ (psi): Rotation about the x-axis.

        Formula:
            Depending on the structure of the rotation matrix \(R\):
            - Handle standard cases when \(R[2, 2] > -1\) and \(R[2, 2] < 1\).
            - Handle gimbal lock cases when \(R[2, 2] = ±1\).

        Example:
            Input:
            R = [[ 0.866, -0.5,  0. ],
                [ 0.5,    0.866, 0. ],
                [ 0. ,    0. ,   1. ]]
            Output: (φ=0.5236, θ=0.0, ψ=0.0) (30° rotation about z-axis)

        Notes:
            - Euler angles are dependent on the chosen axis order (assumes ZYX order here).
            - Handles edge cases like gimbal lock (singularities in rotation representation).
        """
        # Check the third row, third column element to determine the rotation scenario
        if R[2, 2] < 1:  # General case: not gimbal lock
            if R[2, 2] > -1:  # Unique solution
                phi = np.pi - math.atan2(R[0, 2], R[1, 2])  # Rotation about z-axis
                theta = math.acos(R[2, 2])  # Rotation about y-axis
                psi = np.pi - math.atan2(R[2, 0], -R[2, 1])  # Rotation about x-axis
                return phi, theta, psi
            # Gimbal lock: theta = π
            phi = -math.atan2(R[0, 1], -R[0, 0])  # Rotation about z-axis
            return phi, np.pi, 0
        # Gimbal lock: theta = 0
        phi = math.atan2(R[0, 1], R[0, 0])  # Rotation about z-axis
        return phi, 0, 0

    def exp_map(self, phi):
        """
        Exponential map from so(3) → SO(3) using Euler–Rodrigues (unit-quaternion) formula.

        Interprets `phi ∈ ℝ³` as a rotation **vector** (axis–angle in vector form):
        θ = ‖phi‖ is the rotation angle (radians), and n = phi / θ is the unit axis.
        The corresponding unit quaternion is
            q = [cos(θ/2),  sin(θ/2) * n]
        and the rotation matrix is obtained from this quaternion.

        For very small angles (θ < 1e-8), a first-order quaternion approximation
        q ≈ [1 − θ²/8,  0.5*phi] is used for numerical stability. The quaternion is
        normalized before conversion to ensure a proper rotation matrix.

        Parameters
        ----------
        phi : array_like, shape (3,)
            Rotation vector (axis–angle). Its norm is the rotation angle in radians;
            its direction is the rotation axis.

        Returns
        -------
        R : ndarray, shape (3, 3)
            Proper orthonormal rotation matrix in SO(3) corresponding to exp(̂phi).

        Notes
        -----
        - This is equivalent to the matrix exponential `exp(hat(phi))`, where
        `hat(v)` is the 3×3 skew-symmetric (cross-product) matrix of `v`.
        - The returned matrix is orthonormal up to floating-point roundoff.
        - Small-angle branch is accurate to O(θ³) and avoids division by very small θ.
        - The mapping is consistent with *right-multiplication* usage in updates like
        `R_new = R @ exp_map(delta)`.

        Examples
        --------
        >>> rb = RigidBodySim()
        >>> Rz90 = rb.exp_map([0, 0, np.pi/2])  # 90° about z
        >>> np.allclose(Rz90 @ [1,0,0], [0,1,0], atol=1e-12)
        True

        See Also
        --------
        hat_matrix : Build the skew-symmetric matrix of a 3-vector.
        r_from_quaternions : Convert a unit quaternion to a rotation matrix.
        """

        phi = np.asarray(phi, float).reshape(3,)
        th = np.linalg.norm(phi)
        if th < 1e-8:
            # small-angle quaternion approx: [1 - th^2/8, 0.5*phi]
            q0 = 1.0 - (th*th)/8.0
            qv = 0.5 * phi
        else:
            n  = phi / th
            q0 = np.cos(th/2.0)
            qv = np.sin(th/2.0) * n

        q = np.concatenate(([q0], qv))
        q /= np.linalg.norm(q)
        return self.r_from_quaternions(q)

    def re3_equals_gamma(self, gamma):
        """
        Computes a rotation matrix \( R \) such that \( R \cdot e_3 = \gamma \),
        where \( e_3 \) is the unit axis along the z-axis.

        This method constructs a rotation matrix that aligns the z-axis (\( e_3 \))
        with a given 3D vector (\( \gamma \)). It uses the axis-angle representation
        to calculate the required rotation.

        Args:
            gamma (numpy.ndarray): A 3-element numpy array (or \(3 \times 1\) matrix)
                                  representing the target axis to align \( e_3 \) with.
                                  \( \gamma \) should be normalized.

        Returns:
            numpy.ndarray: A \(3 \times 3\) rotation matrix \( R \) such that \( R \cdot e_3 = \gamma \).

        Formula:
            1. Compute the angle \( \theta = \arccos(\gamma[2]) \).
            2. Determine the rotation axis \( n \):
              \( n = [-\gamma[1]/\sin(\theta), \gamma[0]/\sin(\theta), 0] \).
            3. Construct the quaternion using \( \theta \) and \( n \).
            4. Convert the quaternion to a rotation matrix \( R \).

        Example:
            Input:
            gamma = [0, 0, 1]  # Already aligned with e_3
            Output:
            [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]  # Identity matrix

            Input:
            gamma = [1/√2, 0, 1/√2]  # 45-degree tilt in the xz-plane
            Output:
            [[ 0.7071, 0.0,  0.7071],
            [ 0.0,    1.0,  0.0   ],
            [-0.7071, 0.0,  0.7071]]

        Notes:
            - \( \gamma \) must be normalized before passing to the function.
            - If \( \sin(\theta) \) is close to zero, the method handles the edge case
              where \( \gamma \) is aligned with \( e_3 \) directly.
        """
        # Compute the rotation angle (theta) between e3 and gamma
        theta = math.acos(gamma[2])

        # Handle edge case where gamma is already aligned with e3
        if np.isclose(np.sin(theta), 0):
            return np.identity(3)  # No rotation needed

        # Compute the rotation axis (n) as a 3x1 matrix
        n = np.array([[-gamma[1] / np.sin(theta)],
                      [gamma[0] / np.sin(theta)],
                      [0]])

        # Construct the quaternion and compute the rotation matrix
        return self.r_from_quaternions(self.q_from_axis_angles(theta, n))

    def rotate_and_translate(self, object_vertices, R, o):
        """
        Applies a rotation and translation to a set of object vertices.

        This method transforms an object's vertices by first applying a rotation
        using the provided rotation matrix \( R \), and then translating the result
        by the translation  \( o \).

        Args:
            object_vertices (numpy.ndarray): A \(3 \times N\) matrix where each column represents
                                            the coordinates of a vertex in the object.
            R (numpy.ndarray): A \(3 \times 3\) rotation matrix representing the orientation.
            o (numpy.ndarray): A \(3 \times 1\) matrix (or list/array of 3 elements)
                              representing the translation.

        Returns:
            numpy.ndarray: A \(3 \times N\) matrix where each column represents the transformed
                          coordinates of the vertices after applying the rotation and translation.

        Formula:
            Transformed Vertices = Translation + Rotation @ Original Vertices
            \[
            \text{Transformed Vertices} = o + R \cdot \text{object\_vertices}
            \]

        Example:
            Input:
            object_vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 3 vertices
            R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90-degree rotation about z-axis
            o = [1, 1, 0]  # Translation]

            Output:
            [[1, 1, 1], [0, 2, 1], [0, 0, 1]]  # Transformed vertices

        Notes:
            - The object vertices should be provided as a \(3 \times N\) matrix.
            - The translation  \( o \) should be a \(3 \times 1\) matrix or convertible to one.
            - Ensures seamless transformation of 3D objects in simulations and graphics.
        """
        # Reshape the translation into a 3x1 matrix if not already in that form
        o = np.array([[o[0], o[1], o[2]]]).T

        # Apply the rotation and translation to the object vertices
        return o + R @ object_vertices

    def add_orth_norm_frame(self, fig, o, R, axis_range, axis_color):
        """
        Adds an orthonormal frame to a 3D Plotly figure, representing the axes of a rotated frame.

        This method visualizes a rotated orthonormal frame in a 3D plot. The frame is defined
        by its origin and a rotation matrix, and it is represented by three arrows indicating
        the rotated \( x \)-axis, \( y \)-axis, and \( z \)-axis.

        Args:
            fig (plotly.graph_objects.Figure): The 3D figure to which the frame will be added.
            o (numpy.ndarray): A \(3 \times 1\) matrix (or list/array of 3 elements) representing
                              the origin of the frame in 3D space.
            R (numpy.ndarray): A \(3 \times 3\) rotation matrix defining the orientation of the frame.
            axis_range (list of tuples): Specifies the range for each axis in the format
                                        [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
            axis_color (str): Color for the axis lines in the frame (e.g., "red", "blue", "green").

        Returns:
            plotly.graph_objects.Figure: The input figure with the added orthonormal frame.

        Example:
            Input:
            fig = go.Figure()  # Empty 3D figure
            o = [0, 0, 0]  # Origin at (0, 0, 0)
            R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90-degree rotation about z-axis
            axis_range = [(-1, 1), (-1, 1), (-1, 1)]
            axis_color = "blue"

            Output:
            A 3D figure with the rotated frame visualized.

        Notes:
            - Each axis of the frame is scaled based on the rotation matrix \( R \).
            - The figure layout is updated to match the specified axis range and ensure aspect ratio consistency.
        """
        # Define the e-frame axis for the x, y, and z axes
        e = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # Apply the rotation matrix R to each e-frame axis to compute the rotated b-frame axes
        b = [R @ ei for ei in e]

        # Add each axis as a line starting at origin `o` and extending in the rotated direction
        for bi in b:
            fig.add_trace(go.Scatter3d(
                x=[o[0], o[0] + bi[0]],  # Line along the rotated axis
                y=[o[1], o[1] + bi[1]],
                z=[o[2], o[2] + bi[2]],
                hoverinfo='x+y+z',  # Tooltip displays the 3D coordinates
                mode='lines',  # Display as lines
                line=dict(width=8, color=axis_color)  # Line styling
            ))

        # Update the layout to fix the axis ranges and maintain aspect ratio
        fig.update_layout(
            showlegend=False,  # Hide legend
            scene=dict(
                xaxis=dict(range=axis_range[0], autorange=False),
                yaxis=dict(range=axis_range[1], autorange=False),
                zaxis=dict(range=axis_range[2], autorange=False),
                aspectratio=dict(x=1, y=1, z=1)  # Keep the aspect ratio uniform
            )
        )
        return fig

    def animate_particle_motion(self, xx, axis_range, fig_title):
        """
        Creates a 3D animated visualization of a particle's motion over time.

        This method visualizes the trajectory of a particle in 3D space, showing both
        the particle's current position as a marker and the entire path it follows as a line.

        Args:
            xx (list of tuples): A list of 3D coordinates representing the particle's position
                                at each time step, formatted as [(x1, y1, z1), (x2, y2, z2), ...].
            axis_range (list of tuples): Specifies the range for each axis in the format
                                        [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
            fig_title (str): Title for the animated 3D plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the animated 3D scatter plot.

        Example:
            Input:
            xx = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 2, 1)]  # Particle's path
            axis_range = [(-5, 5), (-5, 5), (-5, 5)]  # Axis limits
            fig_title = "Particle Motion Animation"

            Output:
            A Plotly animated 3D scatter plot showing the particle's motion over time.

        Notes:
            - The particle's current position is shown as a red marker.
            - The complete trajectory is shown as a blue line.
            - The animation can be controlled using play/pause buttons in the Plotly interface.

        """
        # Unpack the particle's trajectory into separate x, y, z coordinates
        x_vals, y_vals, z_vals = zip(*xx)

        # Define the initial position of the particle as a red marker
        trace_particle = go.Scatter3d(
            x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],  # Start with the first position
            mode="markers",  # Show as a marker
            marker=dict(color="red", size=10)  # Red marker with size 10
        )

        # Define the complete trajectory as a blue line
        trace_path = go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,  # All trajectory points
            mode="lines",  # Show as a line
            line=dict(color="blue", width=2),  # Blue line with width 2
            name='Path'
        )

        # Define the layout of the plot
        layout = go.Layout(
            title_text=fig_title,  # Title of the figure
            hovermode="closest",  # Tooltip shows closest data point
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="Play",  # Play button for the animation
                    method="animate",
                    args=[None]
                )]
            )],
            scene=dict(  # 3D scene settings
                xaxis=dict(range=axis_range[0], autorange=False),  # Fixed x-axis range
                yaxis=dict(range=axis_range[1], autorange=False),  # Fixed y-axis range
                zaxis=dict(range=axis_range[2], autorange=False),  # Fixed z-axis range
                aspectratio=dict(x=1, y=1, z=1)  # Uniform aspect ratio
            )
        )

        # Create animation frames for each point in the trajectory
        frames = [go.Frame(
            data=[go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],  # Current position of the particle
                mode="markers",  # Show as a marker
                marker=dict(color="red", size=10),  # Red marker with size 10
                name='Particle'
            )]) for point in xx]

        # Create the Plotly figure with initial data, layout, and frames
        fig = go.Figure(data=[trace_particle, trace_path], layout=layout, frames=frames)

        # Display the figure
        fig.show()

        # Return the figure object
        return fig

    def set_external_force_model(self, fn: Callable[..., tuple]) -> None:
        """
        Register the external force/torque model used by the rigid-body integrators.

        The function you provide is called each integration step to produce the **external**
        generalized inputs: torque (τₑ) and force (fₑ). Both must be 3-vectors.

        Accepted call signatures
        ------------------------
        - fn(parameters, X) -> (tau_e, f_e)
        - fn(self, parameters, X) -> (tau_e, f_e)

        where:
        parameters : dict
            Simulation parameters (e.g., {'M': mass, 'II': inertia, ...}).
        X : list
            Current state in the class' format:
                X = [[R, o], omega, p, Xc]
                with R∈ℝ^{3×3}, o∈ℝ³, omega∈ℝ³, p∈ℝ³, Xc=aux/controller state.

        Returns
        -------
        None

        Requirements
        ------------
        - tau_e and f_e must be array-like 3-vectors (shape (3,) after `np.asarray`).
        - Units and frames must be consistent with your dynamics (same frame as used in
        `rigid_body_system` and the integrators).

        Notes
        -----
        If no external model is registered, the integrators fall back to zeros for (τₑ, fₑ).
        The method supports either a bound method (expects `self`) or a free function.

        Examples
        --------
        >>> def gravity_only(parameters, X):
        ...     M = parameters['M']
        ...     g = 9.81
        ...     tau_e = [0.0, 0.0, 0.0]
        ...     f_e = [0.0, 0.0, -M*g]
        ...     return tau_e, f_e
        >>> sim.set_external_force_model(gravity_only)
        """
        if not callable(fn):
            raise TypeError("externalForceModel must be callable")
        self.externalForceModel = fn

    def set_actuator(self, fn: Callable[..., tuple]) -> None:
        """
        Register the actuator model (control inputs) used by the rigid-body integrators.

        The function you provide is called each integration step to produce the **actuated**
        generalized inputs: torque (τₐ) and force (fₐ). Both must be 3-vectors. The actuator
        can depend on time, current state, and the external inputs (τₑ, fₑ).

        Accepted call signatures
        ------------------------
        - fn(parameters, t, X, tau_e, f_e) -> (tau_a, f_a)
        - fn(self, parameters, t, X, tau_e, f_e) -> (tau_a, f_a)

        where:
        parameters : dict
            Simulation parameters (e.g., {'M': mass, 'II': inertia, ...}).
        t : float
            Current simulation time.
        X : list
            Current state in the class' format:
                X = [[R, o], omega, p, Xc]
        tau_e, f_e : array-like
            External torque/force passed through from the external model.

        Returns
        -------
        None

        Requirements
        ------------
        - tau_a and f_a must be array-like 3-vectors (shape (3,) after `np.asarray`).
        - Keep frames/units consistent with your dynamics implementation.

        Notes
        -----
        If no actuator is registered, the integrators fall back to zeros for (τₐ, fₐ).
        The method supports either a bound method (expects `self`) or a free function.

        Examples
        --------
        >>> def pd_attitude(parameters, t, X, tau_e, f_e):
        ...     # toy example: no forces, small stabilizing torque
        ...     Kp = 0.5; Kd = 0.1
        ...     R, o = X[0]
        ...     omega = X[1]
        ...     tau_a = (-Kd) * omega  # simple damping
        ...     f_a = [0.0, 0.0, 0.0]
        ...     return tau_a, f_a
        >>> sim.set_actuator(pd_attitude)
        """
        if not callable(fn):
            raise TypeError("actuator must be callable")
        self.actuator = fn

    def rigid_body_system(self, parameters, t, X):
        """
        Models the dynamics of a rigid body system.

        This method computes the time derivatives and other intermediate quantities
        for a rigid body's state, based on the provided parameters and external forces/torques.

        Args:
            parameters (dict): A dictionary containing parameters for the rigid body system:
                - 'CM' (numpy.ndarray): Center of mass position as a \(3 \times 1\) matrix.
                - 'M' (float): Mass of the rigid body.
            t (float): Current time (useful for time-dependent forces/torques).
            X (list): State of the system, containing:
                - \( X[0][0] \) (numpy.ndarray): Rotation matrix \( R \) (\(3 \times 3\)).
                - \( X[1] \) (numpy.ndarray): Spatial angular velocity \( \omega \) (\(3 \times 1\)).
                - \( X[2] \) (numpy.ndarray): Linear momentum r \( p \) (\(3 \times 1\)).

        Returns:
            list: A list of quantities for the rigid body dynamics:
                - \( \thetaomega \) (float): Magnitude of spatial angular velocity \( \omega \).
                - \( \nomega \) (numpy.ndarray): Normalized spatial angular velocity axis(\(3 \times 1\)).
                - \( \doto \) (numpy.ndarray): Time derivative of the position (\(3 \times 1\)).
                - \( dp \) (numpy.ndarray): Time derivative of linear momentum (\(3 \times 1\)).
                - \( dspi \) (numpy.ndarray): Time derivative of spatial angular momentum (\(3 \times 1\)).
                - \( dXc \) (numpy.ndarray): Time derivative of other external dynamics (\(3 \times 1\)).

        Formula:
            The equations are modeled as:
            - \( \dot{o} = \frac{p}{M} \)
            - \( \dot{p} = f_e + f_a \)
            - \( \dot{\text{spi}} = \tau_e + \tau_a \)
            - \( \omega = \text{angular velocity} \)

        Example:
            Input:
            parameters = {'CM': [0, 0, 0], 'M': 10}
            t = 0
            X = [[[np.eye(3)], [1, 0, 0], [0, 1, 0]]]

            Output:
            - \( \thetaomega \): 1.0
            - \( \nomega \): [1, 0, 0]
            - \( \doto \): [0, 0.1, 0.1]
            - \( dp \): ...
            - ...

        Notes:
            - Requires user-defined `externalForceModel` and `actuator` functions to be registered.
              external and actuator forces/torques.
            - Spatial angular velocity \( \omega \) is normalized if its magnitude exceeds a small threshold.

        """
        taue = np.zeros(3); fe = np.zeros(3)
        taua = np.zeros(3); fa  = np.zeros(3)

        ext = getattr(self, "externalForceModel", None)
        act = getattr(self, "actuator", None)

        # Helper to coerce 3-vectors
        def _v3(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != 3:
                raise ValueError("Expected a 3-vector")
            return x

        # Call externalForceModel if provided
        if callable(ext):
            try:
                # bound method or function not expecting self explicitly
                _taue, _fe = ext(parameters, X)
            except TypeError:
                # free function expecting self as first arg
                _taue, _fe = ext(self, parameters, X)
            taue, fe = _v3(_taue), _v3(_fe)

        # Call actuator if provided (uses taue/fe even if zeros)
        if callable(act):
            try:
                _taua, _fa = act(parameters, t, X, taue, fe)
            except TypeError:
                _taua, _fa = act(self, parameters, t, X, taue, fe)
            taua, fa = _v3(_taua), _v3(_fa)


        # Extract parameters
        barX, M = parameters['CM'], parameters['M']

        # Extract state variables
        R = X[0][0]  # Rotation matrix
        omega = X[1]  # Spatial angular velocity
        p = X[2]  # Linear momentum

        # Compute external and actuator forces and torques
        if not (callable(ext) and callable(act)):
            taue = taua = fe = fa = np.zeros(3)
        else:
            taue, fe = ext(self, parameters, X)  # External forces and torques
            taua, fa = act(self, parameters, t, X, taue, fe)  # Actuator forces and torques

        # Compute time derivatives of position, momentum, and spin
        doto = p / M  # Time derivative of position
        dp = fe + fa  # Time derivative of linear momentum
        dspi = taue + taua  # Time derivative of spin

        # External dynamics (can be expanded with a controller model)
        dXc = np.array([0., 0., 0.])

        # Compute angular velocity properties
        if np.linalg.norm(omega) >= 0.0001:  # Avoid division by zero
            nomega = omega / np.linalg.norm(omega)  # Normalized angular velocity
            thetaomega = np.linalg.norm(omega)  # Magnitude of angular velocity
        else:  # Handle case of negligible angular velocity
            nomega = np.array([0, 0, 0])
            thetaomega = 0

        # Return computed quantities
        return [thetaomega, nomega, doto, dp, dspi, dXc]

    def animate_2D_scatter_plot(self, x, YY, xlabel, ylabel, title):
        """
        Creates an animated 2D scatter plot using Plotly.

        This method visualizes a series of 2D data points as an animation, where the y-values
        evolve over time for a fixed set of x-values. It is useful for illustrating dynamic
        changes in data over time.

        Args:
            x (numpy.ndarray): A 1D array representing the x-axis values.
            YY (numpy.ndarray): A 2D array where each row represents the y-values at a specific
                                time step, and columns correspond to the x-values.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title for the animated plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the animated scatter plot.

        Example:
            Input:
            x = np.linspace(0, 10, 100)  # x-values
            YY = np.array([np.sin(x + t) for t in np.linspace(0, 2 * np.pi, 50)])  # y-values evolve over time
            xlabel = "X-axis"
            ylabel = "Y-axis"
            title = "Animated 2D Scatter Plot"

            Output:
            A Plotly animated scatter plot showing the evolution of the sine wave over time.

        Notes:
            - The animation buttons are included by default in the plot layout.
            - The range of the y-axis is automatically determined based on the data in `YY`.

        """
        # Define the layout of the plot
        layout = go.Layout(
            xaxis={'title': xlabel},  # Label for the x-axis
            yaxis={'title': ylabel, 'range': [1.1 * YY.min(), 1.1 * YY.max()]},  # Label and range for the y-axis
            title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},  # Title styling
            scene=dict(aspectratio=dict(x=1, y=1)),  # Maintain aspect ratio
            hovermode="closest",  # Display closest data point information
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",  # Animation play button
                        method="animate",
                        args=[None]
                    )
                ]
            )]
        )

        # Create frames for each time step in YY
        frames = [go.Frame(data=[go.Scatter(x=x, y=y)]) for y in YY]

        # Initialize the figure with the first frame's data and the layout
        fig = go.Figure(data=[go.Scatter(x=x, y=YY[0, :])], layout=layout, frames=frames)

        # Return the figure object (can be displayed using fig.show())
        return fig

    def simulate_dy_system(self, dynamic_system_model, t_max, dt, x0, sys_para, fig_title, x_label, y_label):
        """
        Simulates the dynamics of a system using a numerical solver and visualizes the results.

        This method solves a system of differential equations defined by a user-provided
        dynamic system model, using initial conditions and system parameters. The solution
        is plotted over time for visualization.

        Args:
            dynamic_system_model (function): The function defining the system's dynamics.
                                            It should return \( \frac{dx}{dt} \) given
                                            the current state, time, and parameters.
            t_max (float): The total simulation time.
            dt (float): The time step for the simulation.
            x0 (numpy.ndarray): Initial state of the system (\( n \times 1 \) matrix or array).
            sys_para (any): Parameters for the dynamic system model, passed as an argument to the function.
            fig_title (str): Title for the generated plot.
            x_label (str): Label for the x-axis of the plot (typically time).
            y_label (str): Label for the y-axis of the plot (state variables).

        Returns:
            tuple: A tuple containing:
                - t (numpy.ndarray): The array of time points.
                - sol (numpy.ndarray): The solution matrix where each row corresponds to
                                      the state vector at a specific time.
                - fig (plotly.graph_objects.Figure): The Plotly figure object showing the simulation results.

        Example:
            Input:
            dynamic_system_model = LinearSystemModel  # A predefined dynamic model
            t_max = 10  # Simulate for 10 seconds
            dt = 0.1  # Time step of 0.1 seconds
            x0 = [1, 0]  # Initial state
            sys_para = [[0, 1], [-1, -2]]  # System matrix A
            fig_title = "System Dynamics"
            x_label = "Time (s)"
            y_label = "State Variables"

            Output:
            - Time array \( t \)
            - Solution matrix \( sol \)
            - Plotly figure showing the state variables over time

        Notes:
            - The `dynamic_system_model` must accept parameters in the form
              `(state_vector, time, system_parameters)`.
            - Uses `scipy.integrate.odeint` for numerical integration.
            - Visualization is done using Plotly, with each state variable plotted as a line.

        """
        # Generate time points for the simulation
        t = np.linspace(0, t_max, int(t_max / dt + 1))

        # Solve the system of differential equations
        sol = odeint(dynamic_system_model, x0, t, args=(sys_para,))

        # Create a Plotly figure for visualization
        fig = go.Figure()

        # Add a line plot for each state variable
        for sol_col in sol.T:
            fig.add_trace(go.Scatter(x=t, y=sol_col, mode='lines+markers'))

        # Update the figure layout with titles and axis labels
        fig.update_layout(
            title=fig_title,
            xaxis=dict(title=x_label),
            yaxis=dict(title=y_label)
        )

        # Display the figure
        fig.show()

        # Return the time points, solution matrix, and figure
        return t, sol, fig

    def LinearSystemModel(self, X, t, A):
        """
        Represents a linear dynamic system model.

        This function defines the evolution of a linear system over time using the
        state-space representation. It calculates the time derivative of the state
         \( X \) given the system matrix \( A \).

        Args:
            X (numpy.ndarray): A \( n \times 1 \) matrix (or list/array of \( n \) elements)
                              representing the state of the system.
            t (float): Time variable (not used in the calculation but required for compatibility
                      with ODE solvers).
            A (numpy.ndarray): A \( n \times n \) matrix representing the system matrix that defines
                              the linear dynamics.

        Returns:
            numpy.ndarray: The time derivative of the state \( dX/dt \), computed as \( A \cdot X \).

        Formula:
            The system is modeled as:
            \[
            \frac{dX}{dt} = A \cdot X
            \]

        Example:
            Input:
            X = [1, 2]
            t = 0  # Time (not used in this example)
            A = [[0, 1], [-1, -2]]  # System matrix

            Output:
            [-2, -5]  # Time derivative of the state vector

        Notes:
            - Ensure the dimensions of \( A \) and \( X \) are consistent (\( A \) should be square and
              \( X \) should have the appropriate size).
            - Commonly used in simulations of linear systems such as control systems and electrical circuits.
        """
        # Compute the time derivative of the state vector using the system matrix
        dXdt = A @ X
        return dXdt

    def cube_vertices(self, cube_dimensions):
        """
        Return the 8 axis-aligned vertices of a rectangular cuboid as three coordinate lists.

        The cuboid is defined in its local (object) frame by edge lengths `(l, w, h)`.
        A *pivot* (xp, yp, zp) is subtracted from each coordinate so that the point
        `(xp, yp, zp)` becomes the origin of the returned vertices. This is convenient
        when you want rotations to happen about a particular point (e.g., the cube
        center: set `xp=l/2, yp=w/2, zp=h/2`).

        Parameters
        ----------
        cube_dimensions : dict
            Dictionary with keys:
            - 'l', 'w', 'h' : float
                Lengths along the x-, y-, and z-axes, respectively (edge lengths).
            - 'xp', 'yp', 'zp' : float
                Pivot offset to subtract from each coordinate. After subtraction,
                the pivot lies at the origin (0, 0, 0) of the returned vertex set.

        Returns
        -------
        list[list[float]]
            A list `[X, Y, Z]` of three lists, each of length 8.
            These are the x-, y-, and z-coordinates of the 8 vertices in a fixed order:
            indices 0–3 are the lower (z = -zp) face, 4–7 are the upper (z = h - zp) face:
                0: (-xp,      -yp,      -zp)
                1: (-xp,       w-yp,    -zp)
                2: ( l-xp,     w-yp,    -zp)
                3: ( l-xp,    -yp,      -zp)
                4: (-xp,      -yp,       h-zp)
                5: (-xp,       w-yp,     h-zp)
                6: ( l-xp,     w-yp,     h-zp)
                7: ( l-xp,    -yp,       h-zp)

        Notes
        -----
        - The shape is 3×8 (as three 1×8 lists). This layout plays nicely with later
        rigid-body transforms, e.g., `R @ vertices + o`, where `R` is 3×3 and `o` is 3×1.
        - To center the cuboid at the origin before rotation, use
        `xp = l/2`, `yp = w/2`, `zp = h/2`.

        Examples
        --------
        >>> dims = {'l': 2.0, 'w': 2.0, 'h': 4.0, 'xp': 1.0, 'yp': 1.0, 'zp': 2.0}
        >>> X, Y, Z = RigidBodySim().cube_vertices(dims)
        >>> len(X), len(Y), len(Z)
        (8, 8, 8)

        See Also
        --------
        rotate_and_translate : Apply a rotation and translation to these vertices.
        """
        l, w, h = cube_dimensions.get('l',1.0), cube_dimensions.get('w',1.0), cube_dimensions.get('h',1.0)
        xp, yp, zp = cube_dimensions.get('xp',1.0), cube_dimensions.get('yp',1.0), cube_dimensions.get('zp',1.0)

        X = [-xp, -xp, l-xp, l-xp, -xp, -xp, l-xp, l-xp]
        Y = [-yp, w-yp, w-yp, -yp, -yp, w-yp, w-yp, -yp]
        Z = [-zp, -zp, -zp, -zp, h-zp, h-zp, h-zp, h-zp]

        return [X, Y, Z]

    def animated_cube_flat_shading(self, cubeVertices,figTitle):
        """
        Animate a rigid-body cube (triangulated) with flat shading in Plotly.

        This renders a sequence of cube poses as an animation using a single
        `Mesh3d` per frame. Each pose is provided as the 8 vertices of the cube
        after applying your rigid-body transform. The faces are drawn via fixed
        triangle indices (`i, j, k`) and displayed with flat shading.

        Parameters
        ----------
        cubeVertices : list
            A list of frames. **Each element must be a one-item list** whose item
            is `[X, Y, Z]`, where `X`, `Y`, `Z` are length-8 lists (or arrays) of
            the cube’s vertex coordinates at that time step:

                cubeVertices = [
                    [[X0, Y0, Z0]],   # frame 0 → each of X0/Y0/Z0 has 8 numbers
                    [[X1, Y1, Z1]],   # frame 1
                    ...
                ]

            This matches the output structure of `simulating_a_cube(...)`, which
            builds `rotatedVertices=[[XX0], [XX1], ...]` with `XXk=[Xk, Yk, Zk]`.

        figTitle : str
            Title for the Plotly figure.

        Returns
        -------
        plotly.graph_objects.Figure
            The animated Plotly figure. Use `fig.show()` to display (already
            called inside the method) or further customize the layout/traces.

        Notes
        -----
        - The cube faces are defined by the fixed triangle index lists `i, j, k`
        assuming vertex ordering consistent with `cube_vertices(...)`:
            indices 0–3 → lower face, 4–7 → upper face.
        - The scene axis ranges are set to `[-5, 5]` for all axes; adjust as needed.
        - Animation `frame.duration` is set to 10 (milliseconds).
        - If you want to use a different vertex container shape (e.g. a direct
        `(3,8)` array per frame), adapt the indexing where `x=xx[0][0]`, etc.

        Examples
        --------
        >>> rb = RigidBodySim()
        >>> frames = rb.simulating_a_cube(dt, Tmax, cube_dims, params, ICs)
        >>> fig = rb.animated_cube_flat_shading(frames, "Rigid Body RK4 Demo")
        >>> # fig is shown and also returned
        """

        fig = go.Figure(
            frames=[go.Frame(data=[
            go.Mesh3d(
                # 8 vertices of a cube
                x=xx[0][0],
                y=xx[0][1],
                z=xx[0][2],
                # i, j and k give the vertices of triangles
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                name='y',
                opacity=0.6,
                color='#DC143C',
                flatshading = True)]) for xx in cubeVertices])

        fig.add_trace(go.Mesh3d(
                # 8 vertices of a cube
                x=cubeVertices[0][0][0],
                y=cubeVertices[0][0][1],
                z=cubeVertices[0][0][2],
                # i, j and k give the vertices of triangles
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                name='y',
                opacity=0.6,
                color='#DC143C',
                flatshading = True)
            )

        duration=10;
        fig.update_layout(
            title=figTitle,width=600,height=600,
            scene=dict(xaxis=dict(range=[-5., 5.], autorange=False),yaxis=dict(range=[-5., 5.], autorange=False),zaxis=dict(range=[-5., 5.], autorange=False),aspectratio=dict(x=1, y=1, z=1),),
            updatemenus=[dict(type="buttons",
                                buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None, {"frame": {"duration": duration},"mode": "immediate","fromcurrent": True, "transition": {"duration": duration, "easing": "linear"},}]
                                                )])])
        len(fig.frames)
        fig.show()
        return fig

    def eulers_method(self, dt, Tmax, parameters, ICs):
        """
        Integrate rigid-body translation & rotation with **explicit (forward) Euler** time stepping.

        This stepper advances a free/flying rigid body subject to external and control
        wrenches provided by user hooks `externalForceModel(...)` and `actuator(...)`.
        The state uses your 4-block layout:
            X = [[R, o], omega, doto, Xc]
        where
            R     ∈ ℝ^{3×3} : rotation (body→inertial),
            o     ∈ ℝ^3     : position of the reference point,
            omega ∈ ℝ^3     : spatial angular velocity (in inertial frame),
            doto  ∈ ℝ^3     : linear velocity (in inertial frame),
            Xc    ∈ ℝ^m     : extra/controller state (passed through here).

        Dynamics (discrete Euler update)
        --------------------------------
        Let M be mass, II the body inertia in the body frame. Define
        spi = R II Rᵀ omega          # spatial angular momentum (in inertial frame)
        p   = M doto                  # linear momentum (in inertial frame)

        At each step:
        1) Query user hooks for wrenches:
            (tau_e, f_e) = externalForceModel(self, parameters, X)
            (tau_a, f_a) = actuator(self, parameters, t, X, tau_e, f_e)
            with tau_* ∈ ℝ^3 torques, f_* ∈ ℝ^3 forces (in inertial frame).

        2) Integrate momenta (explicit Euler):
            dspi = tau_e + tau_a
            dp   = f_e   + f_a
            spi ← spi + dt * dspi
            p   ← p   + dt * dp

        3) Integrate pose/velocity:
            o     ← o + dt * doto
            doto  ← p / M
            omega ← (R II Rᵀ)^{-1} spi = R (II^{-1}) Rᵀ spi
            R     ← exp( dt * omega ) @ R
            (Uses quaternion-based exp map for numerical stability; small-angle branch included.)

        Parameters
        ----------
        dt : float
            Time step (seconds).
        Tmax : float
            Total simulated time (seconds). The integrator runs over t = 0:dt:Tmax.
        parameters : dict
            Physical parameters. Recognized keys:
            'M'  : mass (default 1.0)
            'II' : 3×3 body inertia matrix (default I₃)
            Additional keys are passed to user hooks.
        ICs : list
            Initial state in the form [[R0, o0], omega0, doto0, Xc0].

        Returns
        -------
        Xout : list
            List of states `X` at each saved time step, including the initial one.
            Also stores the last state into `self.state` and the whole trajectory
            into `self.trajectory`.

        Notes
        -----
        - **Stability**: explicit Euler is only conditionally stable; for large angular
        rates/torques use a sufficiently small `dt` (or prefer the RK4 stepper).
        - **Hooks contract**: You must provide `externalForceModel(self, parameters, X)`
        and `actuator(self, parameters, t, X, tau_e, f_e)` that each return a tuple
        of 3-vectors `(tau, f)` in the **inertial frame**.
        - Inertia handling assumes `II` is defined in the body frame. The spatial
        inertia used here is `I_spatial = R II Rᵀ`.
        - Orientation update uses `self.r_from_quaternions(...)` via an exponential map
        with a small-angle branch to avoid numerical issues for ‖omega‖ dt ≪ 1.

        Example
        -------
        >>> params = {'M': 2.0, 'II': np.diag([0.2, 0.3, 0.4])}
        >>> R0 = np.eye(3); o0 = np.zeros(3); omega0 = np.zeros(3); doto0 = np.zeros(3); Xc0 = np.zeros(3)
        >>> ICs = [[R0, o0], omega0, doto0, Xc0]
        >>> traj = rb.eulers_method(dt=1e-3, Tmax=1.0, parameters=params, ICs=ICs)
        """
        M, II = parameters.get('M',1.0), parameters.get('II',np.eye(3))
        invII = np.linalg.inv(II)
        timeSteps = np.arange(0, Tmax+dt, dt)
        R, o, omega, doto, Xc = ICs[0][0], ICs[0][1], ICs[1], ICs[2], ICs[3]
        spi = R @ II @ R.T @ omega
        p = M * doto
        Xout = [ICs]
        self.state=ICs
        for t in timeSteps:
            taue, fe = externalForceModel(self, parameters, X)
            taua, fa = actuator(self, parameters, t, X, taue, fe)

            dspi = taue + taua
            dp = fe + fa

            if np.linalg.norm(omega) >= 0.0001:
                nomega = omega / np.linalg.norm(omega)
                thetaomegat = dt * np.linalg.norm(omega)
                qomegat = np.concatenate(([np.cos(thetaomegat/2)], np.sin(thetaomegat/2) * nomega))
                R = self.r_from_quaternions(qomegat) @ R

            o += dt * doto
            spi += dt * dspi
            p += dt * dp
            doto = p / M
            omega = R @ invII @ R.T @ spi
            X = [[R, o], omega, doto, Xc]
            Xout.append(X)
            self.state=X
        self.trajectory=Xout
        return Xout

    def runga_kutta_method(self, dt, Tmax, parameters, ICs):
        """
        Integrate a rigid body's pose and momenta with a **Lie-group RK4** stepper.

        This integrator advances the composite state
            X = [[R, o], omega, p, Xc]
        over t ∈ [0, Tmax] using a fourth-order Runge–Kutta scheme that:
        • updates attitude R ∈ SO(3) via a **left-increment** R ← exp(dt·ω̄) @ R
            (preserves orthonormality by construction),
        • updates translation, linear momentum, and auxiliary/controller state
            with classical RK4 stage averaging.

        ---------- State & Dynamics ----------
        State layout:
            R     : (3,3) rotation matrix (orthonormal, det=+1)
            o     : (3,)  position
            omega : (3,)  body angular velocity
            p     : (3,)  linear momentum
            Xc    : (...) auxiliary/controller state (any shape)

        Required hooks on `self`:
            rigid_body_system(parameters, t, X) -> tuple
                Returns (theta, n, doto, dp, dspi, dXc) at (t, X), where
                • theta = ||omega|| (scalar),
                • n     = omega / ||omega|| (unit axis, zero if tiny),
                • doto  = ȯ (linear velocity),
                • dp    = ṗ (external/actuator forces),
                • dspi  = spatial spin rate (time derivative of spatial angular momentum),
                • dXc   = time derivative of Xc.
                The product w := theta * n is treated as the **body angular rate** used
                to build RK4 stage predictors on SO(3).
            exp_map(v: (3,)) -> (3,3)
                Exponential map on SO(3): exp_map(v) = exp(hat(v)).
            r_from_quaternions / hat_matrix (optional elsewhere): not used here.

        Parameters
        ----------
        dt : float
            Fixed time step (seconds).
        Tmax : float
            Total simulated time horizon (seconds). The loop executes N=floor(Tmax/dt) steps.
            Use ceil if you prefer to land exactly on Tmax.
        parameters : dict
            Model parameters; recognized keys:
            • 'II' : (3,3) SPD body inertia matrix (default I3).
            • Any additional keys are forwarded to `rigid_body_system`.
        ICs : list/tuple
            Initial composite state in the same layout as X: [[R0, o0], omega0, p0, Xc0].

        Returns
        -------
        Xout : list
            Sequence of states over the grid (length N+1), including the initial state `ICs`.
            Each item has the layout [[R, o], omega, p, Xc].

        Algorithm (per step)
        --------------------
        1) Evaluate k1 at (t, X).
        2) Build stage predictor Y1 at t+dt/2 with exp(0.5·dt·w1) and Euler drifts; evaluate k2.
        3) Build stage predictor Y2 at t+dt/2 with exp(0.5·dt·w2); evaluate k3.
        4) Build stage predictor Y3 at t+dt with exp(dt·w3); evaluate k4.
        5) Form RK4 weighted averages:
            w̄   = (w1 + 2w2 + 2w3 + w4)/6,
            d•̄  = (k1 + 2k2 + 2k3 + k4)/6 for each translational/momentum/controller rate.
        6) Update:
            R_new   = exp(dt·w̄) @ R,
            o_new   = o   + dt·dō̄,
            p_new   = p   + dt·dp̄,
            spi_new = R II Rᵀ omega + dt·ds̄,
            omega_new = R_new II⁻¹ R_newᵀ spi_new,
            Xc_new  = Xc + dt·dXc̄.

        Numerical Properties & Notes
        ----------------------------
        • SO(3) update is group-consistent and maintains RᵀR=I, det R=+1 up to machine precision.
        • Stage times are (t, t+dt/2, t+dt/2, t+dt); weights (1,2,2,1)/6 (classical RK4).
        • The angular stage vector w_i := theta_i n_i must represent a **body-frame** rate to
        match the left-increment kinematics Ṙ = R·hat(omega_body).
        • A small-angle safeguard is applied implicitly by exp_map; you may add an explicit check
        if your exp_map assumes ||v|| > 0.

        Caveats
        -------
        • If your dynamics produce **spatial** quantities, convert them to/from the body frame
        consistently (this routine assumes body w_i and spatial momentum spin dynamics).
        • The loop uses floor(Tmax/dt); consider adjusting if exact endpoint alignment is required.

        Example
        -------
        >>> params = {'II': np.diag([0.1, 0.2, 0.3])}
        >>> R0 = np.eye(3); o0 = np.zeros(3); omega0 = np.array([0.1, 0.0, 0.0])
        >>> p0 = np.zeros(3); Xc0 = np.zeros(2)
        >>> traj = sim.runga_kutta_method(0.01, 1.0, params, [[R0, o0], omega0, p0, Xc0])
        >>> R1 = traj[-1][0][0]
        >>> np.allclose(R1.T @ R1, np.eye(3), atol=1e-12)
        True
        """
        II = parameters.get('II', np.eye(3))
        invII = np.linalg.inv(II)

        # steps: floor hits t in [0, N*dt); use ceil if you want to land on Tmax exactly
        N = int(np.floor(Tmax / dt + 1e-12))
        t0 = 0.0

        X = ICs
        Xout = [X]
        self.state = ICs

        for k in range(N):
            t = t0 + k*dt
            R, o, omega, p, Xc = X[0][0], X[0][1], X[1], X[2], X[3]

            # k1 at (t, X)
            th1, n1, do1, dp1, ds1, dXc1 = self.rigid_body_system(parameters, t, X)

            # k2 from Y1 at t+dt/2
            w1 = th1 * n1
            Y1 = [[ self.exp_map(0.5*dt * w1) @ R, o + 0.5*dt*do1 ],
                omega + 0.5*dt*w1, p + 0.5*dt*dp1, Xc + 0.5*dt*dXc1]
            th2, n2, do2, dp2, ds2, dXc2 = self.rigid_body_system(parameters, t + 0.5*dt, Y1)

            # k3 from Y2 at t+dt/2
            w2 = th2 * n2
            Y2 = [[ self.exp_map(0.5*dt * w2) @ R, o + 0.5*dt*do2 ],
                omega + 0.5*dt*w2, p + 0.5*dt*dp2, Xc + 0.5*dt*dXc2]
            th3, n3, do3, dp3, ds3, dXc3 = self.rigid_body_system(parameters, t + 0.5*dt, Y2)

            # k4 from Y3 at t+dt
            w3 = th3 * n3
            Y3 = [[ self.exp_map(dt * w3) @ R, o + dt*do3 ],
                omega + dt*w3, p + dt*dp3, Xc + dt*dXc3]
            th4, n4, do4, dp4, ds4, dXc4 = self.rigid_body_system(parameters, t + dt, Y3)

            # RK4 weights
            w_bar   = (th1*n1 + 2*th2*n2 + 2*th3*n3 + th4*n4) / 6.0
            do_bar  = (do1     + 2*do2    + 2*do3    + do4   ) / 6.0
            dp_bar  = (dp1     + 2*dp2    + 2*dp3    + dp4   ) / 6.0
            ds_bar  = (ds1     + 2*ds2    + 2*ds3    + ds4   ) / 6.0
            dXc_bar = (dXc1    + 2*dXc2   + 2*dXc3   + dXc4  ) / 6.0

            # group-consistent updates
            theta = np.linalg.norm(w_bar)
            if theta > 1e-12:
                R_new = self.exp_map(dt * w_bar) @ R
            else:
                R_new = R

            o_new   = o   + dt * do_bar
            p_new   = p   + dt * dp_bar
            spi_new = R @ II @ R.T @ omega + dt * ds_bar
            omega_new = R_new @ invII @ R_new.T @ spi_new
            Xc_new  = Xc + dt * dXc_bar

            X = [[R_new, o_new], omega_new, p_new, Xc_new]
            Xout.append(X)
            self.state = X

        self.trajectory = Xout
        return Xout

    def _rk4_function(self, dtk, X_base, tk, Xk, parameters, invII=None):
        """
        One RK substep: advance a state Xk forward by dtk using dynamics at (tk, Xk).
        Returns a new temporary state (R_next, o_next, omega_next, p_next, Xc_next)
        built RELATIVE TO Xk (not X_base).
        """
        II = parameters.get('II', np.eye(3))
        if invII is None:
            invII = np.linalg.inv(II)

        # dynamics at the stage point (tk, Xk)
        theta, n, doto, dp, dspi, dXc = self.rigid_body_system(parameters, tk, Xk)

        # unpack the STAGE BASE (use Xk as the reference)
        Rk, ok   = Xk[0]
        omegak   = Xk[1]
        pk       = Xk[2]
        Xck      = Xk[3]

        # incremental rotation from stage angular vector w = theta*n
        w = theta * n
        R_next = self.exp_map(dtk * w) @ Rk

        # advance translational/momentum/controller from STAGE BASE
        o_next   = ok  + dtk * doto
        p_next   = pk  + dtk * dp
        spi_next = Rk @ II @ Rk.T @ omegak + dtk * dspi    # spatial angular momentum
        omega_next = R_next @ invII @ R_next.T @ spi_next  # body rate at new frame
        Xc_next = Xck + dtk * dXc

        return [[R_next, o_next], omega_next, p_next, Xc_next]

    def simulating_a_cube(self, dt, Tmax, cubeDimensions, parameters,ICs):
        XX=self.cube_vertices(cubeDimensions);

        #Xs=self.eulers_method(dt,Tmax,parameters,ICs);
        Xs=self.runga_kutta_method(dt,Tmax,parameters,ICs);
        ICR=ICs[0][0];
        XX0=ICR @ XX;

        rotatedVertices=[[XX0]]
        for X in Xs:
        #print(X[0])
            R=X[0][0];
            o=X[0][1];
            XXi=self.rotate_and_translate(XX,R,o);
            XX0=XXi;
            rotatedVertices+=[[XX0]];
        return rotatedVertices

    def plot_momentum_energy_intersection(
        self,
        *,
        parameters: dict,
        ICOmega: np.ndarray,
        ICR: np.ndarray | None = None,
        grid_phi: int = 50,
        grid_theta: int = 100,
        tol: float = 1e-3,
        overlay_spatial: bool = False,
        sphere_opacity: float = 0.35,
        ellipsoid_opacity: float = 0.35,
        title: str | None = None,
        show: bool = True,
    ):
        """
        Plot |π| = const sphere and 2*KE = const ellipsoid in π-space and mark their intersection (polhode).

        Parameters
        ----------
        parameters : dict
            Must include 'II' (3x3 inertia in *body* frame). Symmetric, SPD expected.
        ICOmega : (3,) array-like
            Initial angular velocity in body frame (ω₀). Used to set |π| and KE via π₀ = I ω₀.
        ICR : (3,3) array-like, optional
            Rotation matrix from body→spatial frame at the instant of interest.
            If provided with overlay_spatial=True, a spatial-frame overlay is drawn (R·π).
        grid_phi : int
            Number of polar samples on the parameter grid (≥ 8 recommended).
        grid_theta : int
            Number of azimuth samples on the parameter grid (≥ 16 recommended).
        tol : float
            Relative tolerance for detecting the intersection curve on the sphere:
            mark points where |πᵀ I^{-1} π − 2KE| ≤ tol * (2KE).
        overlay_spatial : bool
            If True and `ICR` is provided, overlays the rotated surfaces/curve (spatial frame).
        sphere_opacity : float
            Opacity of the |π| sphere surface (0–1).
        ellipsoid_opacity : float
            Opacity of the 2KE ellipsoid surface (0–1).
        title : str, optional
            Figure title. If None, a default informative title is used.
        show : bool
            If True, calls fig.show().

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The constructed figure.
        info : dict
            Useful computed scalars: {'magPi', 'KE', 'pi0_body', 'pi0_spatial'(if ICR), 'eigvals', 'eigvecs'}.

        Notes
        -----
        - In π-space, the constant-energy surface is an ellipsoid: πᵀ I^{-1} π = 2 KE.
        In the principal-axes basis (I = diag(λ₁,λ₂,λ₃)), semi-axes are √(2 KE λᵢ).
        - The intersection with the |π| sphere is the (body-frame) polhode curve.
        """
        II = np.asarray(parameters["II"], dtype=float)
        if II.shape != (3, 3):
            raise ValueError("parameters['II'] must be a 3x3 matrix")
        if not np.allclose(II, II.T, atol=1e-10):
            raise ValueError("Inertia matrix II must be symmetric")
        # Basic SPD sanity (cholesky)
        LA.cholesky(II)

        ICOmega = np.asarray(ICOmega, dtype=float).reshape(3,)
        invII = LA.inv(II)

        # π0, |π|, KE
        pi0_body = II @ ICOmega
        magPi = LA.norm(pi0_body)
        KE = 0.5 * ICOmega @ (II @ ICOmega)  # == 0.5 * pi0^T invII pi0

        # Parametric grid on sphere (body frame)
        phi = np.linspace(0.0, np.pi, grid_phi)
        theta = np.linspace(0.0, 2.0 * np.pi, grid_theta)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")  # (P,T)

        # Sphere of radius |π|
        x1 = magPi * np.sin(PHI) * np.cos(THETA)
        y1 = magPi * np.sin(PHI) * np.sin(THETA)
        z1 = magPi * np.cos(PHI)

        # Ellipsoid for π^T invII π = 2KE → in principal axes: axes = sqrt(2 KE * λ)
        lam, Q = LA.eigh(II)  # II = Q diag(lam) Q^T, lam>0 ascending
        axes = np.sqrt(2.0 * KE * lam)  # (3,)

        # Build ellipsoid in principal frame then rotate to body frame
        xp = axes[0] * np.sin(PHI) * np.cos(THETA)
        yp = axes[1] * np.sin(PHI) * np.sin(THETA)
        zp = axes[2] * np.cos(PHI)
        Pp = np.stack([xp, yp, zp], axis=0).reshape(3, -1)        # (3, N)
        Pb = (Q @ Pp).reshape(3, *xp.shape)                       # (3, P, T)
        x2, y2, z2 = Pb[0], Pb[1], Pb[2]

        # Intersection on the sphere: evaluate f(π) = π^T invII π - 2KE
        Ps = np.stack([x1, y1, z1], axis=0).reshape(3, -1)        # sphere points (3,N)
        f = np.einsum("ij,jn,in->n", invII, Ps, Ps).reshape(x1.shape) - (2.0 * KE)
        mask = np.abs(f) <= (tol * 2.0 * KE)
        xi, yi, zi = x1[mask], y1[mask], z1[mask]                 # polhode samples (body frame)

        # Optional spatial overlay via ICR (R body→spatial)
        traces = []
        if overlay_spatial and ICR is not None:
            R = np.asarray(ICR, dtype=float).reshape(3, 3)
            # Rotate surfaces
            S_sphere = (R @ Ps).reshape(3, *x1.shape)
            S_ellip  = (R @ Pb.reshape(3, -1)).reshape(3, *x2.shape)
            x1s, y1s, z1s = S_sphere[0], S_sphere[1], S_sphere[2]
            x2s, y2s, z2s = S_ellip[0],  S_ellip[1],  S_ellip[2]
            # Rotate intersection points
            Pi = np.stack([xi, yi, zi], axis=0)  # (3, K)
            Pis = (R @ Pi)
            xis, yis, zis = Pis[0], Pis[1], Pis[2]
        else:
            x1s = y1s = z1s = x2s = y2s = z2s = xis = yis = zis = None

        # Build Plotly figure
        fig = go.Figure()

        # Body-frame surfaces
        fig.add_trace(go.Surface(
            x=x1, y=y1, z=z1, showscale=False, opacity=sphere_opacity,
            name="|π| = const (sphere, body)"
        ))
        fig.add_trace(go.Surface(
            x=x2, y=y2, z=z2, showscale=False, opacity=ellipsoid_opacity,
            name="2·KE = const (ellipsoid, body)"
        ))
        # Intersection (polhode) in body frame
        fig.add_trace(go.Scatter3d(
            x=xi, y=yi, z=zi, mode="markers",
            marker=dict(size=3),
            name="intersection (polhode, body)"
        ))

        # Optional spatial overlay
        if overlay_spatial and ICR is not None:
            fig.add_trace(go.Surface(
                x=x1s, y=y1s, z=z1s, showscale=False, opacity=0.2,
                name="|π| sphere (spatial)"
            ))
            fig.add_trace(go.Surface(
                x=x2s, y=y2s, z=z2s, showscale=False, opacity=0.2,
                name="2·KE ellipsoid (spatial)"
            ))
            fig.add_trace(go.Scatter3d(
                x=xis, y=yis, z=zis, mode="markers",
                marker=dict(size=3),
                name="intersection (spatial overlay)"
            ))

        # Axes range (auto) and layout
        max_extent = float(np.max(np.abs(np.concatenate([
            x1.ravel(), y1.ravel(), z1.ravel(),
            x2.ravel(), y2.ravel(), z2.ravel()
        ]))))
        pad = 0.1 * max_extent
        rng = [-max_extent - pad, max_extent + pad]

        fig.update_layout(
            title=title or "Intersection of angular-momentum sphere and constant-energy ellipsoid (π-space)",
            scene=dict(
                xaxis=dict(range=rng, autorange=False),
                yaxis=dict(range=rng, autorange=False),
                zaxis=dict(range=rng, autorange=False),
                aspectmode="data"
            ),
            showlegend=True
        )

        if show:
            fig.show()

        info = {
            "magPi": magPi,
            "KE": KE,
            "pi0_body": pi0_body,
            "pi0_spatial": (ICR @ pi0_body) if (overlay_spatial and ICR is not None) else None,
            "eigvals": lam,
            "eigvecs": Q,
        }
        return fig, info


    # --- Rigid Body EKF ---

    def set_sensor(self, fn: Callable[..., tuple]) -> None:
        """
        Register a **sensor callback** used to simulate/ingest IMU-like measurements.

        The callback will be invoked by your code (e.g., inside tests or higher-level
        loops) to produce:
        - a 3-vector gyro reading, and
        - two 3-vectors for the measured body-frame directions of two inertial axes
            (by convention here: e1 and e3, but your implementation may choose them).

        Expected callable signature
        ---------------------------
        fn(R: np.ndarray, omega_body: np.ndarray, *[, ...]) -> tuple
            Parameters
            ----------
            R : (3,3) ndarray
                Current attitude (body → inertial).
            omega_body : (3,) ndarray
                Body-frame angular velocity used to form the gyro measurement.
            ... : optional keyword args
                Noise stds, RNG handle, flags (e.g., renormalize), etc.

            Returns
            -------
            Omega_meas : (3,) ndarray
                Gyro measurement (usually `omega_body + noise`).
            A_n_meas : (3,) ndarray
                Noisy measurement of `Rᵀ e1` (direction #1 expressed in body frame).
            A_g_meas : (3,) ndarray
                Noisy measurement of `Rᵀ e3` (direction #2 expressed in body frame).

        Notes
        -----
        - Your EKF’s measurement covariance `Σ_m` should be compatible with the
        distribution of `A_n_meas` and `A_g_meas` produced by this function.
        - If you dynamically choose the inertial axes (not strictly e1/e3), keep the
        EKF’s H/S construction consistent with that choice.

        Examples
        --------
        >>> def my_sensor(R, omega_body, sigma_omega=5e-3, sigma_dir=2e-2):
        ...     # return (Omega_meas, A_n_meas, A_g_meas)
        ...     ...
        >>> rb.set_sensor(my_sensor)
        """
        self.sensor = fn

    def set_KF_innovation(self, fn: Callable[..., np.ndarray]) -> None:
        """
        Register the **innovation function** L = y - ŷ used by the EKF update.

        The callback computes the stacked residual for two direction measurements.
        It must return a column vector shaped (6,1) (or a 1-D array convertible to that),
        consistent with `H ∈ ℝ^{6×3}`.

        Expected callable signature
        ---------------------------
        fn(R_pred_minus: np.ndarray, A_n: np.ndarray, A_g: np.ndarray) -> np.ndarray
            Parameters
            ----------
            R_pred_minus : (3,3) ndarray
                Predicted-minus attitude used to form the predicted measurements.
            A_n : (3,) ndarray
                Measured body-frame direction of inertial axis #1 (e.g., `Rᵀ e1`).
            A_g : (3,) ndarray
                Measured body-frame direction of inertial axis #2 (e.g., `Rᵀ e3`).

            Returns
            -------
            L : (6,1) ndarray
                Innovation (residual) stacked as:
                    L = vec([A_n; A_g] - [R^-ᵀ e1; R^-ᵀ e3])

        Notes
        -----
        - If you change which inertial axes are used (not strictly e1/e3), this
        function must mirror that choice so that `ŷ` matches how H is built.
        - Returning a 1-D array of length 6 is acceptable; it will be reshaped to (6,1)
        by the EKF before multiplication.

        Examples
        --------
        >>> def my_innovation(Rm, A_n, A_g):
        ...     e1 = np.array([1.,0.,0.]); e3 = np.array([0.,0.,1.])
        ...     y      = np.vstack([A_n, A_g])
        ...     y_pred = np.vstack([Rm.T @ e1, Rm.T @ e3])
        ...     return (y - y_pred).reshape(-1, 1)
        >>> rb.set_KF_innovation(my_innovation)
        """
        self.kf_innovation = fn

    def _linearization_attitude_kinematics(self, DeltaT: float, Omega: np.ndarray, R_for_H: np.ndarray):
        """
        A_k-1 = I 
        G_k-1 = (DeltaT ** 0.5) * R @ Phi(-DeltaT * Omega)
        H_k-1 = [ -R^T hat(e1) R ; -R^T hat(e2) R ; -R^T hat(e3) R ]  (stacked 6x3), using predicted-minus attitude.
        Uses the identity R^T hat(e) R = hat(R^T e) for efficiency.
        """
        I3 = np.eye(3)
        Omega = np.asarray(Omega, float).reshape(3,)
        R = np.asarray(R_for_H, float).reshape(3, 3)

        # A and G
        A_km1 = np.eye(3) #self.exp_map(DeltaT * Omega) #
        G_km1 = (DeltaT) * R @  self._Phi_SO3(-DeltaT * Omega) # sqrt Brownian

        # H using e1 & e3
        e1 = np.array([1., 0., 0.])
        e2 = np.array([0., 1., 0.])
        e3 = np.array([0., 0., 1.])
        H1 = -self.hat_matrix(R.T @ e1) 
        H2 = -self.hat_matrix(R.T @ e2) 
        H3 = -self.hat_matrix(R.T @ e3)
        H_km1 = np.vstack([H1, H2, H3])    # (9,3)

        return A_km1, G_km1, H_km1

    # --- EKF predict/update (intrinsic, e1 & e3) ---

    def predict_update_attitude(
        self,
        DeltaT: float,
        Omega_km1: np.ndarray,       # (3,) body angular velocity
        R_previous: np.ndarray,      # (3,3) previous attitude (k-1)
        P_previous: np.ndarray,      # (3,3) previous covariance (k-1)
        Sigma_q: np.ndarray,         # (3,3) process noise cov (gyro PSD discretized)
        Sigma_m: np.ndarray,         # (6,6) meas noise cov for stacked [R^T e1; R^T e3]
        A_1_meas: np.ndarray,        # (3,) measured R^T e1
        A_2_meas: np.ndarray,        # (3,) measured R^T e2
        A_3_meas: np.ndarray,        # (3,) measured R^T e3
    ):
        """
        Intrinsic EKF on SO(3) with two direction measurements.

        Discretization (ΔT in the correction):
            R_k^- = R_{k-1} · exp(ΔT · Ω_{k-1})
            P_k^- = A P_{k-1} A^T + G Σ_q G^T,    A = I - ΔT·hat(Ω_{k-1}),  G = √ΔT·I
            H_k   = [ -hat(R_k^-T e1) ; -hat(R_k^-T e3) ]
            K_k   = P_k^- H_k^T (H_k P_k^- H_k^T + Σ_m)^{-1}
            R_k   = R_k^- · exp(ΔT · K_k (y_k - ŷ_k^-))
            P_k   = (I - K_k H_k) P_k^- (I - K_k H_k)^T + K_k Σ_m K_k^T

        Shapes:
            Ω, delta: (3,),  R: (3,3),  P, Σ_q: (3,3),  Σ_m, S: (9,9),
            H: (6,3),  K: (3,6),  innovation L: (6,1) or (6,)

        Notes:
            • Ω must be **body-frame** gyro rate to match A’s definition.
            • Uses Joseph update and light symmetrization for numerical SPD robustness.
        """
        # --- shape checks
        assert R_previous.shape == (3, 3)
        assert P_previous.shape == (3, 3)
        assert Sigma_q.shape == (3, 3)
        assert Sigma_m.shape == (9, 9)
        assert A_1_meas.shape == (3,)
        assert A_2_meas.shape == (3,)
        assert A_3_meas.shape == (3,)

        I3 = np.eye(3)

        # 1) State prediction
        R_pred_minus = R_previous @ self.exp_map(DeltaT * Omega_km1) 

        # 2) Linearize at predicted-minus attitude (H_k at R_k^-)
        A_km1, G_km1, H_km1 = self._linearization_attitude_kinematics(DeltaT, Omega_km1, R_pred_minus)

        # 3) Covariance prediction
        P_pred_minus = A_km1 @ P_previous @ A_km1.T + G_km1 @ Sigma_q @ G_km1.T

        # 4) Innovation 
        L = self.kf_innovation(R_pred_minus, A_1_meas, A_2_meas, A_3_meas)

        # 5) Kalman gain (stable solve; enforce symmetry + tiny jitter on S)
        S = H_km1 @ P_pred_minus @ H_km1.T + Sigma_m
        # Symmetrize for numerical stability
        S = 0.5 * (S + S.T)
        # Add a small jitter (size based on H_km1 output dimension)
        S += 1e-12 * np.eye(S.shape[0])

        # K = P^- H^T S^{-1}  via solve on S^T · K^T = (H P^-) → K = [(S^T)\(H P^-)]^T
        K = DeltaT * np.linalg.solve(S.T, (H_km1 @ P_pred_minus)).T

        # 6) correction with dt and clamp
        # delta = (DeltaT * (K @ L)).reshape(3,)
        delta = ((K @ L)).reshape(3,)
        n = np.linalg.norm(delta)
        if n > 0.2:       # ~11.5 deg cap
            delta *= 0.2 / n
        # R_pred = R_pred_minus @ self.exp_map(delta) #Left invariant outputs
        R_pred = self.exp_map(delta) @ R_pred_minus #Right invariant outputs


        # 7) Covariance update (Joseph + symmetrize)
        IKH = (I3 - K @ H_km1)
        P_pred = IKH @ P_pred_minus @ IKH.T + K @ Sigma_m @ K.T
        P_pred = 0.5 * (P_pred + P_pred.T)

        return R_pred, P_pred, K, H_km1, S

    def run_offline_EKF_analysis(self, trajectory, dt=0.01, 
                                Sigma_q_factor=1.0, Sigma_m_factor=1.0,
                                sigma_omega=5e-3, sigma_dir=2e-2,
                                sigma_init_deg=10.0):
        """
        Run an offline attitude EKF over a simulated trajectory
        and produce diagnostic plots.

        Parameters
        ----------
        trajectory : list
            List of simulation states [(X_k)], where each X_k contains
            [ [R, o], omega, p, Xc ], i.e. rotation, angular velocity, etc.
        dt : float
            Time step [s].
        Sigma_q : ndarray(3x3), optional
            Process noise covariance. If None, default uses (sigma_omega**2)*I3.
        Sigma_m : ndarray(9x9), optional
            Measurement noise covariance. If None, default uses (sigma_dir**2)*I9.
        sigma_omega, sigma_dir : float
            Sensor noise standard deviations.
        sigma_init_deg : float
            Initial attitude uncertainty [deg].
        """

        # --- defaults ---
        sigma_omega_discrete = sigma_omega /(dt**0.5) # Scaled to discrete time step
        Sigma_q = Sigma_q_factor * (sigma_omega ** 2) * np.eye(3)
        Sigma_m = Sigma_m_factor * (sigma_dir ** 2) * np.eye(9)

        deg_to_rad = np.pi / 180.0
        Sigma_p0 = (sigma_init_deg * deg_to_rad)**2 * np.eye(3)

        # --- initial filter state ---
        R_hat = trajectory[0][0][0]  # use true initial orientation
        P_hat = Sigma_p0.copy()

        # --- preallocate metrics ---
        N = len(trajectory)
        t = np.arange(N) * dt
        err_deg, trace_err, lambda_max, ang_1sigma_deg = [], [], [], []

        # --- helper: rotation angle from R ---
        def angle_from_R(R):
            c = (np.trace(R) - 1.0) * 0.5
            c = float(np.clip(c, -1.0, 1.0))
            return np.arccos(c)

        # --- main loop ---
        for k in range(N):
            Xk = trajectory[k]
            R_true = Xk[0][0]
            omega_spatial = Xk[1]
            omega_body = R_true.T @ omega_spatial

            # noisy sensor measurements
            Omega_meas, A_1_meas, A_2_meas, A_3_meas = self.sensor(
                dt, R_true, omega_body, sigma_omega, sigma_dir
            )

            # EKF step
            R_hat, P_hat, K, H, S = self.predict_update_attitude(
                DeltaT=dt,
                Omega_km1=Omega_meas,
                R_previous=R_hat,
                P_previous=P_hat,
                Sigma_q=Sigma_q,
                Sigma_m=Sigma_m,
                A_1_meas=A_1_meas,
                A_2_meas=A_2_meas,
                A_3_meas=A_3_meas,
            )

            # attitude error metrics
            R_err = R_hat.T @ R_true
            err_deg.append(np.degrees(angle_from_R(R_err)))
            trace_err.append(3.0 - np.trace(R_err))

            # covariance quality
            P_sym = 0.5 * (P_hat + P_hat.T)
            w = np.linalg.eigvalsh(P_sym)
            lam_max = float(w[-1])
            lambda_max.append(lam_max)
            ang_1sigma_deg.append(np.degrees(np.sqrt(max(lam_max, 0.0))))

        # --- summary printout ---
        print("==== Offline EKF Analysis ====")
        print(f"Final attitude error: {err_deg[-1]:.3f} deg")
        print(f"Median attitude error: {np.median(err_deg):.3f} deg")
        print(f"Final trace error: {trace_err[-1]:.6f}")
        print(f"Median trace error: {np.median(trace_err):.6f}")
        print(f"Final √λ_max(P): {ang_1sigma_deg[-1]:.3f} deg (1σ equivalent)")

        # --- plots ---
        # 1. Trace error vs time
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=t, y=trace_err, mode="lines",
                                    name="ε_tr = 3 - tr(R_trueᵀ R_hat)"))
        fig_line.update_layout(
            title="Trace misalignment vs time (offline EKF on simulated trajectory)",
            xaxis_title="Time (s)",
            yaxis_title="Trace error (0 = perfect alignment)",
        )

        # 2. Histogram of trace error
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=trace_err, nbinsx=50, name="trace error"))
        fig_hist.update_layout(
            title="Histogram of trace misalignment (offline EKF)",
            xaxis_title="Trace error",
            yaxis_title="Count",
            bargap=0.05,
        )

        # 3. Covariance eigenvalues
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Scatter(x=t, y=lambda_max, mode="lines", name="λ_max(P) [rad²]"))
        fig_cov.add_trace(go.Scatter(x=t, y=ang_1sigma_deg, mode="lines",
                                    name="√λ_max(P) [deg]", yaxis="y2"))
        fig_cov.update_layout(
            title="Covariance magnitude over time (offline EKF)",
            xaxis_title="Time (s)",
            yaxis=dict(title="Largest eigenvalue λ_max(P) [rad²]"),
            yaxis2=dict(title="1σ angle √λ_max(P) [deg]", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
        )

        # show
        fig_line.show()
        fig_hist.show()
        fig_cov.show()

        return dict(
            time=t,
            err_deg=np.array(err_deg),
            trace_err=np.array(trace_err),
            lambda_max=np.array(lambda_max),
            ang_1sigma_deg=np.array(ang_1sigma_deg),
            figs=(fig_line, fig_hist, fig_cov),
        )

    def auto_tune_EKF(self, trajectory, Sigma_p0,
                    sigma_omega=5e-3, sigma_dir=2e-2,
                    dt=0.01, steps=10000,
                    omega_body=None,
                    q_scales=[0.1, 1.0, 10.0, 100.0],
                    m_scales=[0.01, 0.1, 1.0, 10.0],
                    verbose=True):
        """
        Auto-tune EKF by sweeping over Sigma_q and Sigma_m scales.
        Evaluates performance based on final and median attitude RMSE.

        Parameters
        ----------
        trajectory : ndarray
            The trajectory.
        Sigma_p0 : ndarray(3x3)
            Initial covariance.
        sigma_omega, sigma_dir : float
            Sensor noise standard deviations (gyro, direction).
        dt : float
            Time step [s].
        omega_body : ndarray(3,), optional
            Constant body angular velocity [rad/s]; if None → [1,1,1].
        q_scales, m_scales : list of floats
            Multiplicative scales for Sigma_q and Sigma_m.
        verbose : bool
            Print intermediate results.

        Returns
        -------
        best_config : tuple (q_scale, m_scale)
        results : list of tuples (q_scale, m_scale, final_err, median_err)
        """
        sigma_omega_discrete = sigma_omega / (dt**0.5) #Scaled to discrete time step
        # --- search results ---
        best_config = None
        best_score = np.inf
        results = []

        for q_scale in q_scales:
            for m_scale in m_scales:
                # Covariances
                Sigma_q = q_scale * (sigma_omega ** 2) * np.eye(3)
                Sigma_m = m_scale * (sigma_dir ** 2) * np.eye(9)

                # Initialize filter
                R_hat = deepcopy(trajectory[0][0][0])  # initial attitude
                P_hat = Sigma_p0.copy()

                # Metrics
                err_deg = []

                for Xk in trajectory:
                    R_true = Xk[0][0]
                    omega_spatial = Xk[1]
                    omega_body_k = R_true.T @ omega_spatial

                    # generate noisy measurements
                    Omega_meas, A_1_meas, A_2_meas, A_3_meas = self.sensor(
                        dt, R_true, omega_body_k, sigma_omega, sigma_dir
                    )

                    # EKF predict + update
                    R_hat, P_hat, *_ = self.predict_update_attitude(
                        DeltaT=dt,
                        Omega_km1=Omega_meas,
                        R_previous=R_hat,
                        P_previous=P_hat,
                        Sigma_q=Sigma_q,
                        Sigma_m=Sigma_m,
                        A_1_meas=A_1_meas,
                        A_2_meas=A_2_meas,
                        A_3_meas=A_3_meas,
                    )

                    # compute attitude error
                    R_err = R_hat.T @ R_true
                    c = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
                    angle_rad = np.arccos(c)
                    err_deg.append(np.degrees(angle_rad))

                # summarize
                final_err = err_deg[-1]
                median_err = np.median(err_deg)
                score = final_err + median_err

                results.append((q_scale, m_scale, final_err, median_err))

                if verbose:
                    print(f"[q={q_scale:.2e}, m={m_scale:.2e}] Final: {final_err:.3f}°, Median: {median_err:.3f}°")

                if score < best_score:
                    best_score = score
                    best_config = (q_scale, m_scale)

        print("\n✅ Best configuration found:")
        print(f"Sigma_q = {best_config[0]:.2e} × base (σ_ω² I₃)")
        print(f"Sigma_m = {best_config[1]:.2e} × base (σ_dir² I₉)")

        return best_config, results

##### Dont Delete


    # def runga_kutta_method(self, dt, Tmax, parameters, ICs):
    #     """
    #     Integrate rigid-body pose and momenta with a Lie-group RK4-like stepper.

    #     This routine advances the composite state
    #         X = [[R, o], omega, p, Xc]
    #     over t ∈ [0, Tmax] with step size `dt`, where:
    #     - R ∈ ℝ^{3×3} is the attitude (rotation matrix, orthonormal),
    #     - o ∈ ℝ^3 is the position,
    #     - omega ∈ ℝ^3 is the body angular velocity,
    #     - p ∈ ℝ^3 is the linear momentum,
    #     - Xc is an optional controller/auxiliary state (shape free).

    #     Dynamics/hooks
    #     --------------
    #     The method relies on two helpers you provide elsewhere in the class:
    #     • `rigid_body_system(parameters, t, X)` → tuple
    #         (theta_omega, n_omega, doto, dp, dspi, dXc)
    #         where:
    #         - theta_omega = ‖omega‖ (scalar),
    #         - n_omega = omega / ‖omega‖ (unit axis, or 0 if tiny),
    #         - doto = ṙo (linear velocity),
    #         - dp   = external + actuator forces,
    #         - dspi = external + actuator torques (spatial spin rate),
    #         - dXc  = derivative of controller/aux state.
    #     • `_rk4_function(dtk, X, tk, Xk, parameters)`:
    #         returns an intermediate state used as RK stages.

    #     Model parameters
    #     ----------------
    #     parameters : dict
    #         - 'M'  (float)  : mass (default 1.0 if absent)
    #         - 'II' (3×3 SPD): body inertia matrix (default I₃)
    #         Any extra keys are passed through to your dynamics/controller.
    #     ICs : list-like
    #         Initial state in the same composite format as X:
    #             ICs = [[R0, o0], omega0, p0, Xc0]

    #     Algorithm (per step)
    #     --------------------
    #     1) Build three intermediate stage states with `_rk4_function`:
    #         Y1 = _rk4_function(0.5·dt, X, t,          X,  parameters)
    #         Y2 = _rk4_function(0.5·dt, X, t+0.5·dt,   Y1, parameters)
    #         Y3 = _rk4_function(    dt, X, t+0.5·dt,   Y2, parameters)
    #     Then evaluate `rigid_body_system` at [X, Y1, Y2, Y3] to obtain
    #     four stage tuples (theta_i, n_i, doto_i, dp_i, dspi_i, dXc_i).

    #     2) Rotational update on SO(3) (Lie-group increment):
    #         omega_k ≈ (dt/6) · Σ_i (theta_i · n_i)     # stage-average angular velocity (unweighted sum)
    #         q = (cos(‖omega_k‖/2), sin(‖omega_k‖/2)·hat(omega_k))
    #         R ← r_from_quaternions(q) @ R              # left-multiply incremental rotation

    #     NOTE: This implements a practical RK4-like angular increment; it is not a
    #     strict classical RK4 weighting (1,2,2,1). See "Notes" below.

    #     3) Translational/momentum/controller updates (stage means):
    #         o  ← o  + dt · mean(doto_i)
    #         p  ← p  + dt · mean(dp_i)
    #         spi← (R_old II R_old^T)·omega_old + dt · mean(dspi_i)
    #         Xc ← Xc + dt · mean(dXc_i)
    #         omega ← (R II^{-1} R^T) · spi

    #     4) Append the new state and continue.

    #     Returns
    #     -------
    #     Xout : list
    #         Sequence of states over the grid, including the initial state.
    #         Length is len(np.arange(0, Tmax+dt, dt)) + 1.

    #     Side effects
    #     ------------
    #     - Updates `self.state` each step to the most recent state.
    #     - Stores the full trajectory in `self.trajectory` at the end.

    #     Assumptions & requirements
    #     --------------------------
    #     - `II` must be symmetric positive definite; its inverse is computed once.
    #     - The external/actuation effects are provided through your
    #     `rigid_body_system` (which may internally call user-set hooks such as
    #     `self.externalForceModel` and `self.actuator`).
    #     - Small-angle handling for omega is robust via (theta, n) decomposition.

    #     Notes
    #     -----
    #     • Rotation integration uses a Lie-group update (quaternion → R) to preserve
    #     orthonormality of R exactly.
    #     • Stage averaging for translation/momentum uses a simple arithmetic mean, and
    #     rotational stage combination uses an unweighted sum in (dt/6)·Σ form.
    #     If you require *strict* classical RK4 weights (1, 2, 2, 1), adapt the
    #     stage accumulation accordingly.
    #     • The time grid includes both 0 and Tmax (inclusive). With `dt` that does not
    #     divide Tmax exactly, consider constructing the grid explicitly.

    #     Example
    #     -------
    #     >>> params = {'M': 2.0, 'II': np.diag([0.1, 0.2, 0.3])}
    #     >>> R0 = np.eye(3); o0 = np.zeros(3); omega0 = np.array([0.0, 0.0, 1.0])
    #     >>> p0 = np.array([0.0, 0.0, 0.0]); Xc0 = np.zeros(3)
    #     >>> ICs = [[R0, o0], omega0, p0, Xc0]
    #     >>> traj = sim.runga_kutta_method(dt=0.01, Tmax=1.0, parameters=params, ICs=ICs)
    #     >>> R1, o1 = traj[-1][0]
    #     >>> np.allclose(R1.T @ R1, np.eye(3), atol=1e-12)
    #     True
    #     """

    #     M = parameters.get('M',1)
    #     II = parameters.get('II',np.eye(3))
    #     invII = np.linalg.inv(II)
    #     timeSteps = np.arange(0, Tmax+dt, dt)
    #     X=ICs;
    #     Xout=[X];

    #     self.state=ICs
    #     for t in timeSteps:
    #         Y1 = self._rk4_function(0.5*dt, X, t, X, parameters)
    #         Y2 = self._rk4_function(0.5*dt, X, t+0.5*dt, Y1, parameters)
    #         Y3 = self._rk4_function(dt, X, t+0.5*dt, Y2, parameters)

    #         values = [self.rigid_body_system(parameters, t+i*dt, X_j) for i, X_j in enumerate([X, Y1, Y2, Y3])]
    #         thetas, n_omegas, dotos, dps, dspis, dXcs = zip(*values)

    #         omegak = (dt/6.0) * sum(t * n for t, n in zip(thetas, n_omegas))
    #         nomegak = omegak/np.linalg.norm(omegak) if np.linalg.norm(omegak) >= 0.0001 else np.array([0, 0, 0])
    #         qomegak = np.concatenate(([np.cos(np.linalg.norm(omegak)/2)], np.sin(np.linalg.norm(omegak)/2) * nomegak))
    #         Rk = self.r_from_quaternions(qomegak) @ X[0][0]

    #         ok = X[0][1] + dt * np.mean(dotos)
    #         pk = X[2] + dt * np.mean(dps)
    #         spik = X[0][0] @ II @ X[0][0].T @ X[1] + dt * np.mean(dspis)
    #         Xck = X[3] + dt * np.mean(dXcs)

    #         omegak = Rk @ invII @ Rk.T @ spik
    #         X = [[Rk, ok], omegak, pk, Xck]
    #         Xout.append(X)
    #         self.state=X
    #     self.trajectory=Xout    
    #     return Xout

    # def _rk4_function(self, dtk, X, tk, Xk, parameters):
    #     M, II = parameters['M'], parameters['II']
    #     thetaomega1, nomega1, doto1, dp1, dspi1, dXc1 = self.rigid_body_system(parameters, tk, Xk)
    #     qomega1 = np.concatenate(([np.cos(dtk*thetaomega1/2)], np.sin(dtk*thetaomega1/2) * nomega1))
    #     R1 = self.r_from_quaternions(qomega1) @ X[0][0]
    #     p1 = X[2] + dtk * dp1
    #     spi1 = X[0][0] @ II @ X[0][0].T @ X[1] + dtk * dspi1
    #     omega1 = R1 @ np.linalg.inv(II) @ R1.T @ spi1
    #     X1 = [[R1, X[0][1] + dtk * doto1], omega1, p1, X[3] + dtk * dXc1]
    #     return X1