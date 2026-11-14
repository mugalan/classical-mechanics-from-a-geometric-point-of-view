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
from plotly.subplots import make_subplots
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
        Register the external force and torque model used by the rigid-body integrators.

        The model is evaluated during integration to return the resultant external
        torque tau_e and force f_e, both expressed in the spatial (inertial) frame e.
        The torque must be taken about the same point used to define the integrated
        angular momentum: the center of mass for free motion, or the fixed pivot
        for rotation about a fixed point.

        Accepted call signatures
        ------------------------
        - fn(parameters, X) -> (tau_e, f_e)
        - fn(self, parameters, X) -> (tau_e, f_e)

        Parameters passed to the model
        ------------------------------
        parameters : dict
            Simulation parameters, including mass 'M' and the constant body-frame
            inertia tensor 'II' about the chosen reference point.
        X : list
            Current state:
                X = [[R, o], omega, p, Xc]
            where:
                R     : Rotation matrix mapping body to spatial components.
                o     : Position of the body-frame origin in the spatial frame.
                omega : Spatial angular velocity.
                p     : Spatial linear momentum.
                Xc    : Auxiliary/controller state.

        Returns
        -------
        None

        Requirements
        ------------
        - The model must return (tau_e, f_e), in that order.
        - Both outputs must be array-like with shape (3,).
        - Both outputs must be expressed in the spatial frame, not the body frame.
          Convert body-frame moments and forces using tau_e = R @ T_e and
          f_e = R @ F_e.
        - Use consistent units, typically N*m for torque and N for force.

        Notes
        -----
        The model may be evaluated multiple times per integration step.
        These inputs exclude the forces and torques returned by the actuator model.

        Examples
        --------
        Uniform gravity acting on a body whose origin is at its center of mass,
        with the spatial third axis pointing upwards:

        >>> def gravity_only(sim, parameters, X):
        ...     M = parameters['M']
        ...     g = parameters.get('g', 9.81)
        ...     tau_e = np.zeros(3)
        ...     f_e = np.array([0.0, 0.0, -M*g])
        ...     return tau_e, f_e
        >>> sim.set_external_force_model(gravity_only)
        """
        if not callable(fn):
            raise TypeError("externalForceModel must be callable")
        self.externalForceModel = fn

    def set_actuator(self, fn: Callable[..., tuple]) -> None:
        """
        Register the actuator force and torque model used by the rigid-body integrators.

        The model is evaluated during integration to return the resultant actuator
        torque tau_a and force f_a, both expressed in the spatial (inertial) frame e.
        The torque must be taken about the same point used to define the integrated
        angular momentum: the center of mass for free motion, or the fixed pivot
        for rotation about a fixed point.

        Accepted call signatures
        ------------------------
        - fn(parameters, t, X, tau_e, f_e) -> (tau_a, f_a)
        - fn(self, parameters, t, X, tau_e, f_e) -> (tau_a, f_a)

        Parameters passed to the model
        ------------------------------
        parameters : dict
            Simulation parameters, including mass 'M' and the constant body-frame
            inertia tensor 'II' about the chosen reference point.
        t : float
            Current simulation time.
        X : list
            Current state:
                X = [[R, o], omega, p, Xc]
            where:
                R     : Rotation matrix mapping body to spatial components.
                o     : Position of the body-frame origin in the spatial frame.
                omega : Spatial angular velocity.
                p     : Spatial linear momentum.
                Xc    : Auxiliary/controller state.
        tau_e, f_e : array-like
            External torque and force supplied by the external model, both
            expressed in the spatial frame. The torque is about the same
            reference point as tau_a.

        Returns
        -------
        None

        Requirements
        ------------
        - The model must return (tau_a, f_a), in that order.
        - Both outputs must be array-like with shape (3,).
        - Both outputs must be expressed in the spatial frame, not the body frame.
          Convert body-frame moments and forces using tau_a = R @ T_a and
          f_a = R @ F_a.
        - Use consistent units, typically N*m for torque and N for force.

        Notes
        -----
        The model may be evaluated multiple times per integration step.
        Its outputs are added to the external inputs:
            total torque = tau_e + tau_a
            total force  = f_e + f_a
        Return only the actuator contributions; do not include the external
        inputs again unless deliberately compensating for them.

        Examples
        --------
        Apply rotational damping using the spatial angular velocity:

        >>> def rotational_damping(sim, parameters, t, X, tau_e, f_e):
        ...     Kd = parameters.get('Kd', 0.1)
        ...     omega = np.asarray(X[1], dtype=float)
        ...     tau_a = -Kd * omega
        ...     f_a = np.zeros(3)
        ...     return tau_a, f_a
        >>> sim.set_actuator(rotational_damping)
        """
        if not callable(fn):
            raise TypeError("actuator must be callable")
        self.actuator = fn

    def rigid_body_system(self, parameters, t, X):
        """
        Models the spatial frame dynamics of a rigid body system.

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

    def animated_cube_flat_shading(self, cubeVertices, figTitle):
        """
        Animate a rigid-body block with a distinct pastel color on each face.
    
        Parameters
        ----------
        cubeVertices : list
            Frames in the format returned by simulating_a_cube:
                [[[X0, Y0, Z0]], [[X1, Y1, Z1]], ...]
            Each coordinate array contains eight vertex coordinates.
        figTitle : str
            Figure title.
    
        Returns
        -------
        plotly.graph_objects.Figure
            The figure, which is also displayed.
        """
        # Each consecutive pair of triangles forms one rectangular face.
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
        pastel_colors = [
            "#339AF0",  # Blue   — wide face (−b₂)
            "#F06595",  # Pink   — bottom    (−b₃)
            "#845EF7",  # Violet — top       (+b₃)
            "#FCC419",  # Yellow — wide face (+b₂)
            "#20C997",  # Teal   — end       (−b₁)
            "#FF922B",  # Orange — end       (+b₁)
        ]
        triangle_colors = [
            color for color in pastel_colors for _ in range(2)
        ]
    
        def make_mesh(pose):
            return go.Mesh3d(
                x=pose[0][0],
                y=pose[0][1],
                z=pose[0][2],
                i=i,
                j=j,
                k=k,
                facecolor=triangle_colors,
                flatshading=True,
                opacity=1.0,
                lighting=dict(
                    ambient=0.7,
                    diffuse=0.7,
                    specular=0.1,
                    roughness=0.9,
                ),
                name="Rigid body",
                showscale=False,
            )
    
        fig = go.Figure(
            data=[make_mesh(cubeVertices[0])],
            frames=[
                go.Frame(
                    data=[make_mesh(pose)],
                    traces=[0],
                    name=str(index),
                )
                for index, pose in enumerate(cubeVertices)
            ],
        )
    
        duration = 10
        fig.update_layout(
            title=figTitle,
            width=600,
            height=600,
            scene=dict(
                xaxis=dict(range=[-5.0, 5.0], autorange=False),
                yaxis=dict(range=[-5.0, 5.0], autorange=False),
                zaxis=dict(range=[-5.0, 5.0], autorange=False),
                aspectmode="cube",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": duration,
                                        "redraw": True,
                                    },
                                    "mode": "immediate",
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        )
    
        fig.show()
        return fig


    def animated_cube_with_momentum(
        self,
        cubeDimensions,
        parameters,
        figTitle="Rigid-body motion",
        *,
        states=None,
        duration=20,
        stride=2,
        show_trails=True,
        trail_stride=20,
        arrow_length=None,
        angular_velocity_arrow_length=None,
        body_opacity=1.0,
        show_body_frame=True,
        width=1100,
        height=650,
        show=True,
    ):
        """
        Animate rigid-body motion in synchronized spatial and body views.
    
        Left panel:
            The block moves in the spatial frame e.
            Spatial angular momentum pi and angular velocity omega are shown.
            Thin body-frame axes are attached to the moving block.
    
        Right panel:
            The block remains fixed in body coordinates.
            Body angular momentum Pi and angular velocity Omega are shown.
    
        Click each legend entry to toggle its arrow, label, and trail together.
        Animation updates preserve these visibility choices.
    
        Uses self.trajectory unless states is supplied. No integration is
        performed. Each state must have the form:
            X = [[R, o], omega, p, Xc]
        where omega is SPATIAL angular velocity.
    
        The quantities are reconstructed as:
            Omega = R.T @ omega
            Pi    = II @ Omega
            pi    = R @ Pi
    
        Parameters
        ----------
        cubeDimensions : dict
            Dimensions and origin offsets accepted by cube_vertices.
        parameters : dict
            Contains 'II', the constant body-frame inertia tensor.
            The body origin must be the center of mass or a fixed pivot,
            with inertia and angular momentum taken about that point.
        figTitle : str
            Figure title.
        states : sequence, optional
            State history. Defaults to self.trajectory.
        duration : float
            Display duration per animation frame, in milliseconds.
        stride : int
            Display every stride-th state, including the final state.
        show_trails : bool
            Include accumulated arrow-tip paths.
        trail_stride : int
            Retain every trail_stride-th sample in each path.
            The current tip is always included.
        arrow_length : float, optional
            Display length assigned to the largest angular-momentum magnitude.
            Defaults to 1.5 times the largest block dimension.
        angular_velocity_arrow_length : float, optional
            Display length assigned to the largest angular-velocity magnitude.
            Defaults to 1.15 times the largest block dimension.
        body_opacity : float
            Block opacity in [0, 1].
        show_body_frame : bool
            Draw thin body-frame axes in both panels.
        width, height : int
            Figure dimensions in pixels.
        show : bool
            Whether to display the figure.
    
        Returns
        -------
        plotly.graph_objects.Figure
    
        Notes
        -----
        Momentum and angular velocity use separate fixed display scales.
        Each spatial/body pair shares its scale throughout the animation.
    
        Spatial arrows start at o; body arrows start at the body origin.
        Spatial tip paths include any translation of o.
    
        Both panels use the same coordinate scale initially. Their cameras
        can be manipulated independently.
    
        Legend text and symbols have fixed pixel sizes. They do not scale
        with the 3D cameras. Browser zoom still scales the entire page.
    
        Growing trails increase animation size. Increase stride and/or
        trail_stride for long simulations.
        """
        import numpy as np
        import plotly.graph_objects as go
    
        # ------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------
        if states is None:
            states = getattr(self, "trajectory", None)
    
        if states is None or len(states) == 0:
            raise ValueError(
                "Run simulating_a_cube or runga_kutta_method first, "
                "or provide states explicitly."
            )
    
        for name, value in (("stride", stride), ("trail_stride", trail_stride)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
    
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite.")
    
        if not np.isfinite(body_opacity) or not 0 <= body_opacity <= 1:
            raise ValueError("body_opacity must lie between zero and one.")
    
        II = np.asarray(parameters["II"], dtype=float)
        if II.shape != (3, 3) or not np.all(np.isfinite(II)):
            raise ValueError("parameters['II'] must be a finite 3x3 matrix.")
        if not np.allclose(II, II.T, rtol=1e-10, atol=1e-12):
            raise ValueError("The inertia tensor must be symmetric.")
        if np.any(np.linalg.eigvalsh(II) <= 0):
            raise ValueError("The inertia tensor must be positive definite.")
    
        vertices_body = np.asarray(
            self.cube_vertices(cubeDimensions), dtype=float
        )
        if (
            vertices_body.shape != (3, 8)
            or not np.all(np.isfinite(vertices_body))
        ):
            raise ValueError("cube_vertices must return a finite (3, 8) array.")
    
        dimensions = np.ptp(vertices_body, axis=1)
        if np.any(dimensions <= 0):
            raise ValueError("All block dimensions must be positive.")
    
        block_size = float(np.max(dimensions))
    
        if arrow_length is None:
            arrow_length = 1.5 * block_size
        if angular_velocity_arrow_length is None:
            angular_velocity_arrow_length = 1.15 * block_size
    
        for name, value in (
            ("arrow_length", arrow_length),
            ("angular_velocity_arrow_length", angular_velocity_arrow_length),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
    
        # ------------------------------------------------------------
        # Reconstruct histories
        # ------------------------------------------------------------
        rotations, origins, velocities = [], [], []
    
        for X in states:
            R = np.asarray(X[0][0], dtype=float).reshape(3, 3)
            o = np.asarray(X[0][1], dtype=float).reshape(3)
            omega = np.asarray(X[1], dtype=float).reshape(3)
    
            if not all(np.all(np.isfinite(a)) for a in (R, o, omega)):
                raise ValueError("The state history contains nonfinite values.")
    
            rotations.append(R)
            origins.append(o)
            velocities.append(omega)
    
        rotations = np.asarray(rotations)
        origins = np.asarray(origins)
        omega_history = np.asarray(velocities)
    
        Omega_history = np.einsum("nji,nj->ni", rotations, omega_history)
        Pi_history = Omega_history @ II.T
        pi_history = np.einsum("nij,nj->ni", rotations, Pi_history)
    
        def maximum_norm(*histories):
            return max(
                float(np.max(np.linalg.norm(a, axis=1)))
                for a in histories
            )
    
        hmax = maximum_norm(pi_history, Pi_history)
        wmax = maximum_norm(omega_history, Omega_history)
    
        momentum_scale = arrow_length / hmax if hmax > 0 else 0.0
        velocity_scale = (
            angular_velocity_arrow_length / wmax if wmax > 0 else 0.0
        )
    
        zero_origins = np.zeros_like(origins)
    
        specifications = [
            dict(
                key="pi", label="π", name="π — spatial momentum",
                scene="scene", origins=origins,
                values=pi_history, scale=momentum_scale,
                units="kg·m²/s", color="#202020", dash="solid",
            ),
            dict(
                key="omega", label="ω", name="ω — spatial angular velocity",
                scene="scene", origins=origins,
                values=omega_history, scale=velocity_scale,
                units="rad/s", color="#0057B8", dash="solid",
            ),
            dict(
                key="Pi", label="Π", name="Π — body momentum",
                scene="scene2", origins=zero_origins,
                values=Pi_history, scale=momentum_scale,
                units="kg·m²/s", color="#C2185B", dash="dash",
            ),
            dict(
                key="Omega", label="Ω", name="Ω — body angular velocity",
                scene="scene2", origins=zero_origins,
                values=Omega_history, scale=velocity_scale,
                units="rad/s", color="#00875A", dash="dash",
            ),
        ]
    
        for spec in specifications:
            spec["tips"] = (
                spec["origins"] + spec["scale"] * spec["values"]
            )
    
        indices = list(range(0, len(states), stride))
        if indices[-1] != len(states) - 1:
            indices.append(len(states) - 1)
    
        # ------------------------------------------------------------
        # Block appearance
        # ------------------------------------------------------------
        triangle_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        triangle_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        triangle_k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
        colors = [
            "#339AF0",  # Blue   — −b₂
            "#F06595",  # Pink   — −b₃
            "#845EF7",  # Violet — +b₃
            "#FCC419",  # Yellow — +b₂
            "#20C997",  # Teal   — −b₁
            "#FF922B",  # Orange — +b₁
        ]
        triangle_colors = [c for c in colors for _ in range(2)]
    
        # Extend body axes beyond the corresponding block faces.
        axis_negative = (
            np.minimum(vertices_body.min(axis=1), 0.0) - 0.12 * block_size
        )
        axis_positive = (
            np.maximum(vertices_body.max(axis=1), 0.0) + 0.22 * block_size
        )
    
        def xyz(points):
            points = np.asarray(points)
            return dict(x=points[:, 0], y=points[:, 1], z=points[:, 2])
    
        # ------------------------------------------------------------
        # Construct full initial traces or geometry-only updates
        # ------------------------------------------------------------
        def build_traces(index, initial=False):
            result = []
    
            def add(trace_type, geometry, **style):
                # Frames never set visibility or overwrite legend styles.
                result.append(
                    trace_type(**geometry, **(style if initial else {}))
                )
    
            for scene in ("scene", "scene2"):
                if scene == "scene":
                    R = rotations[index]
                    o = origins[index]
                    vertices = o[:, None] + R @ vertices_body
                else:
                    R = np.eye(3)
                    o = np.zeros(3)
                    vertices = vertices_body
    
                add(
                    go.Mesh3d,
                    xyz(vertices.T),
                    scene=scene,
                    i=triangle_i, j=triangle_j, k=triangle_k,
                    facecolor=triangle_colors,
                    flatshading=True,
                    opacity=body_opacity,
                    lighting=dict(
                        ambient=0.7, diffuse=0.7,
                        specular=0.1, roughness=0.9,
                    ),
                    name="Block",
                    showlegend=False,
                    hoverinfo="skip",
                )
    
                if show_body_frame:
                    for j in range(3):
                        # Thin outlined axes with open arrowheads.
                        direction = R[:, j]
                        perpendicular = R[:, (j + 1) % 3]
                        start = o + axis_negative[j] * direction
                        tip = o + axis_positive[j] * direction
    
                        head_length = 0.08 * block_size
                        wing1 = (
                            tip - head_length * direction
                            + 0.4 * head_length * perpendicular
                        )
                        wing2 = (
                            tip - head_length * direction
                            - 0.4 * head_length * perpendicular
                        )
    
                        points = np.vstack([
                            start, tip,
                            [np.nan, np.nan, np.nan],
                            wing1, tip, wing2,
                        ])
    
                        add(
                            go.Scatter3d,
                            xyz(points),
                            scene=scene,
                            mode="lines",
                            line=dict(color="#555555", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        )
    
                        label_position = tip + 0.06 * block_size * direction
                        add(
                            go.Scatter3d,
                            dict(
                                **xyz(label_position[None, :]),
                                text=[("b₁", "b₂", "b₃")[j]],
                            ),
                            scene=scene,
                            mode="text",
                            textfont=dict(size=13, color="#444444"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
    
            for spec in specifications:
                origin = spec["origins"][index]
                value = spec["values"][index]
                displacement = spec["scale"] * value
                magnitude = float(np.linalg.norm(value))
    
                tip = origin + displacement
                head_base = origin + 0.82 * displacement
                head = tip - head_base
                head_length = float(np.linalg.norm(head))
    
                common = dict(
                    scene=spec["scene"],
                    legendgroup=spec["key"],
                )
    
                add(
                    go.Scatter3d,
                    dict(
                        **xyz(np.vstack([origin, head_base])),
                        hovertemplate=(
                            f"{spec['name']}<br>"
                            f"Components: [{value[0]:.4g}, "
                            f"{value[1]:.4g}, {value[2]:.4g}]<br>"
                            f"Magnitude: {magnitude:.5g} {spec['units']}"
                            "<extra></extra>"
                        ),
                    ),
                    **common,
                    mode="lines",
                    line=dict(
                        color=spec["color"], width=7, dash=spec["dash"]
                    ),
                    name=spec["name"],
                    showlegend=True,
                )
    
                if head_length > 0:
                    direction = head / head_length
                    cone_geometry = dict(
                        **xyz(tip[None, :]),
                        u=[direction[0]],
                        v=[direction[1]],
                        w=[direction[2]],
                        sizeref=head_length,
                    )
                else:
                    cone_geometry = dict(
                        x=[], y=[], z=[], u=[], v=[], w=[], sizeref=1.0
                    )
    
                add(
                    go.Cone,
                    cone_geometry,
                    **common,
                    anchor="tip",
                    sizemode="absolute",
                    colorscale=[
                        [0, spec["color"]], [1, spec["color"]]
                    ],
                    showscale=False,
                    showlegend=False,
                    hoverinfo="skip",
                )
    
                label_position = origin + 1.07 * displacement
                add(
                    go.Scatter3d,
                    dict(
                        **xyz(label_position[None, :]),
                        text=[spec["label"] if magnitude > 0 else ""],
                    ),
                    **common,
                    mode="text",
                    textfont=dict(
                        color=spec["color"],
                        size=18,
                        family="DejaVu Sans, Arial, sans-serif",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
    
                if show_trails:
                    samples = np.arange(0, index + 1, trail_stride)
                    if samples[-1] != index:
                        samples = np.append(samples, index)
    
                    add(
                        go.Scatter3d,
                        xyz(spec["tips"][samples]),
                        **common,
                        mode="lines",
                        line=dict(
                            color=spec["color"], width=3, dash=spec["dash"]
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
    
            return result
    
        initial_traces = build_traces(0, initial=True)
    
        fig = go.Figure(
            data=initial_traces,
            frames=[
                go.Frame(
                    name=str(index),
                    data=build_traces(index),
                    traces=list(range(len(initial_traces))),
                )
                for index in indices
            ],
        )
    
        # ------------------------------------------------------------
        # Fixed ranges, with matching scale in both panels
        # ------------------------------------------------------------
        bounds = {
            "scene": [origins],
            "scene2": [vertices_body.T, np.zeros((1, 3))],
        }
    
        for spec in specifications:
            bounds[spec["scene"]].extend([
                spec["tips"],
                spec["origins"] + 1.12 * spec["scale"] * spec["values"],
            ])
    
        local_axis_points = []
        if show_body_frame:
            for j in range(3):
                unit = np.eye(3)[:, j]
                local_axis_points.extend([
                    axis_negative[j] * unit,
                    (axis_positive[j] + 0.06 * block_size) * unit,
                ])
            local_axis_points = np.asarray(local_axis_points)
            bounds["scene2"].append(local_axis_points)
    
        for index in indices:
            R, o = rotations[index], origins[index]
            bounds["scene"].append((o[:, None] + R @ vertices_body).T)
    
            if show_body_frame:
                bounds["scene"].append(
                    o + local_axis_points @ R.T
                )
    
        centers = {}
        half_width = 0.58 * block_size
    
        for scene, sets in bounds.items():
            points = np.concatenate(sets, axis=0)
            lower, upper = points.min(axis=0), points.max(axis=0)
            centers[scene] = 0.5 * (lower + upper)
            half_width = max(
                half_width, 0.58 * float(np.max(upper - lower))
            )
    
        def scene_layout(scene, domain, labels):
            center = centers[scene]
            return dict(
                domain=dict(x=domain, y=[0, 1]),
                **{
                    axis: dict(
                        title=label,
                        range=[
                            center[j] - half_width,
                            center[j] + half_width,
                        ],
                        autorange=False,
                    )
                    for j, (axis, label) in enumerate(zip(
                        ("xaxis", "yaxis", "zaxis"), labels
                    ))
                },
                aspectmode="cube",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
                uirevision=f"keep-{scene}-camera",
            )
    
        # ------------------------------------------------------------
        # Fixed-size legend and playback controls outside the scenes
        # ------------------------------------------------------------
        fig.update_layout(
            title=dict(text=figTitle, x=0.5),
            width=width,
            height=height,
            autosize=False,
            scene=scene_layout(
                "scene", [0.0, 0.48], ("e₁", "e₂", "e₃")
            ),
            scene2=scene_layout(
                "scene2", [0.52, 1.0], ("b₁", "b₂", "b₃")
            ),
            font=dict(
                family="DejaVu Sans, Arial, sans-serif",
                size=13,
            ),
            annotations=[
                dict(
                    text="Spatial frame e",
                    x=0.24, y=1.04,
                    xref="paper", yref="paper",
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=16),
                ),
                dict(
                    text="Body frame b",
                    x=0.76, y=1.04,
                    xref="paper", yref="paper",
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=16),
                ),
            ],
            legend=dict(
                title=dict(text="Click to show/hide"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.08,
                yanchor="top",
                font=dict(size=13),
                itemsizing="constant",
                groupclick="togglegroup",
                itemclick="toggle",
                itemdoubleclick=False,
                uirevision="keep-visibility",
            ),
            margin=dict(l=20, r=20, t=100, b=150, autoexpand=False),
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.5,
                xanchor="center",
                y=-0.25,
                yanchor="top",
                font=dict(size=13),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": duration, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate",
                        }],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        }],
                    ),
                    dict(
                        label="Reset",
                        method="animate",
                        args=[["0"], {
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        }],
                    ),
                ],
            )],
        )
    
        if show:
            fig.show(config={"responsive": False, "scrollZoom": True})
    
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
        Fourth-order Runge–Kutta–Munthe-Kaas integration.
    
        Public state:
            X = [[R, o], omega, p, Xc]
    
        Conventions:
            R       : body-to-spatial rotation matrix
            o       : position of the center of mass
            omega   : SPATIAL angular velocity
            p       : spatial linear momentum
            Xc      : numerical auxiliary/controller state, any array shape
            II      : constant body-frame inertia about the center of mass
    
        Internally integrates:
            y = (a, o, p, spi, Xc)
    
        Here a is the local rotation increment, reset to zero at each
        time step, and spi is spatial angular momentum:
            R_stage = exp(hat(a)) @ R_base
            omega   = R_stage @ invII @ R_stage.T @ spi
    
        Required hooks:
            self.exp_map(a)
                Returns exp(hat(a)); must handle a = 0.
    
            self.rigid_body_system(parameters, t, X)
                Returns (theta, n, doto, dp, dspi, dXc).
    
                dspi must be the derivative of SPATIAL angular momentum.
                theta and n are accepted but not used by this integrator.
                The hook must evaluate the supplied stage state X without
                modifying it or relying on self.state as its stage state.
    
        Timing:
            Retains the original fixed-step convention:
            N = floor(Tmax / dt), up to roundoff.
            The final time is N * dt, which may be less than Tmax.
    
        Returns:
            List of states, including a copy of the initial state.
            Also sets self.state, self.trajectory, and self.times.
    
        For a fixed pivot, use inertia and angular momentum about the
        pivot, with the corresponding dynamics hook.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if not np.isfinite(Tmax) or Tmax < 0:
            raise ValueError("Tmax must be finite and nonnegative.")
    
        II = np.asarray(parameters.get("II", np.eye(3)), dtype=float)
        invII = np.linalg.inv(II)
    
        # Copy inputs so the initial conditions are not modified.
        R = np.array(ICs[0][0], dtype=float, copy=True)
        o = np.array(ICs[0][1], dtype=float, copy=True)
        omega = np.array(ICs[1], dtype=float, copy=True)
        p = np.array(ICs[2], dtype=float, copy=True)
        Xc = np.array(ICs[3], dtype=float, copy=True)
    
        # Spatial angular momentum is the variable we integrate.
        spi = R @ II @ R.T @ omega
    
        X = [[R, o], omega, p, Xc]
        Xout = [X]
        self.state = X
    
        N = int(np.floor(Tmax / dt + 1e-12))
    
        def advance(base, slope, step):
            """Add a scaled derivative to each local-state component."""
            return tuple(
                value + step * derivative
                for value, derivative in zip(base, slope)
            )
    
        for k in range(N):
            t = k * dt
    
            # All stages use this same reference orientation R.
            # The local rotation increment starts at zero.
            y0 = (np.zeros(3), o, p, spi, Xc)
    
            # Stage 1: derivative at the start of the step.
            k1 = self._rk4_function(
                t, y0, R, parameters, invII
            )
    
            # Stage 2: midpoint of the straight-line predictor using k1.
            y2 = advance(y0, k1, dt / 2)
            k2 = self._rk4_function(
                t + dt / 2, y2, R, parameters, invII
            )
    
            # Stage 3: revised midpoint predictor, again based at y0.
            y3 = advance(y0, k2, dt / 2)
            k3 = self._rk4_function(
                t + dt / 2, y3, R, parameters, invII
            )
    
            # Stage 4: endpoint predictor based at y0.
            y4 = advance(y0, k3, dt)
            k4 = self._rk4_function(
                t + dt, y4, R, parameters, invII
            )
    
            # Average derivatives of (a, o, p, spi, Xc).
            rate = tuple(
                (d1 + 2 * d2 + 2 * d3 + d4) / 6
                for d1, d2, d3, d4 in zip(k1, k2, k3, k4)
            )
    
            a, o, p, spi, Xc = advance(y0, rate, dt)
    
            # Reconstruct the new orientation and spatial angular velocity.
            R = self.exp_map(a) @ R
            omega = R @ invII @ R.T @ spi
    
            X = [[R, o], omega, p, Xc]
            Xout.append(X)
            self.state = X
    
        self.times = dt * np.arange(N + 1)
        self.trajectory = Xout
        return Xout
    
    
    def _rk4_function(self, tk, yk, R_base, parameters, invII):
        """
        Evaluate the local-state derivative at one RK4 stage.
    
        Arguments:
            yk = (a, o, p, spi, Xc)
            R_base = orientation at the START of the full time step
    
        Returns:
            (da, doto, dp, dspi, dXc)
    
        This replaces the old helper: it evaluates a derivative,
        rather than advancing a state.
        """
        a, o, p, spi, Xc = yk
    
        # Reconstruct the physical stage state.
        R_stage = self.exp_map(a) @ R_base
        omega_stage = R_stage @ invII @ R_stage.T @ spi
    
        X_stage = [[R_stage, o], omega_stage, p, Xc]
    
        _, _, doto, dp, dspi, dXc = self.rigid_body_system(
            parameters, tk, X_stage
        )
    
        # B(a, omega): local rotation-coordinate derivative.
        a_cross_omega = np.cross(a, omega_stage)
        da = (
            omega_stage
            - 0.5 * a_cross_omega
            + np.cross(a, a_cross_omega) / 12.0
        )
    
        return (
            da,
            np.asarray(doto, dtype=float),
            np.asarray(dp, dtype=float),
            np.asarray(dspi, dtype=float),
            np.asarray(dXc, dtype=float),
        )    


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
        sphere_opacity: float = 0.20,
        ellipsoid_opacity: float = 0.25,
        title: str | None = None,
        show: bool = True,
    ):
        """
        Plot the angular-momentum sphere, energy ellipsoid, and their intersection.
    
        The main plot uses BODY angular-momentum components Pi:
            ||Pi|| = h,
            Pi.T @ inv(II) @ Pi = 2*KE.
    
        The intersection is drawn as continuous dark curves, computed directly
        from these equations rather than detected using a surface-grid tolerance.
    
        Parameters
        ----------
        parameters : dict
            Must contain 'II': the symmetric positive-definite body-frame inertia
            tensor about the center of mass, or about a fixed pivot.
        ICOmega : (3,) array-like
            Initial BODY angular velocity Omega_0. This differs from the spatial
            angular velocity used in the simulator's state:
                ICOmega = ICR.T @ IComega.
        ICR : (3,3) array-like, optional
            Initial rotation mapping body components to spatial components.
        grid_phi, grid_theta : int
            Surface sampling resolutions. Curve resolution is chosen separately.
        tol : float
            Retained for compatibility with existing calls. No longer used:
            the intersection is computed directly.
        overlay_spatial : bool
            If True, requires ICR and adds a rotated copy of the energy ellipsoid
            and intersection using this fixed orientation. This is an instantaneous
            geometric overlay, NOT a spatial angular-momentum trajectory.
        sphere_opacity, ellipsoid_opacity : float
            Surface opacities between zero and one.
        title : str, optional
            Figure title.
        show : bool
            Whether to display the figure.
    
        Returns
        -------
        fig : plotly.graph_objects.Figure
            The constructed figure.
        info : dict
            Initial momentum, energy, inertia eigenvalues/eigenvectors, and
            intersection curves in body components. Each curve has shape (3, N).
    
        Notes
        -----
        For torque-free motion, the body angular momentum follows the intersection,
        while spatial angular momentum remains constant. The plotted intersection
        includes all components compatible with the selected energy and magnitude.
    
        At limiting energy levels, the intersection can reduce to equilibrium
        points. For isotropic inertia, the sphere and ellipsoid coincide, so there
        is no unique intersection curve.
        """
    
        II = np.asarray(parameters["II"], dtype=float)
        if II.shape != (3, 3) or not np.all(np.isfinite(II)):
            raise ValueError("parameters['II'] must be a finite 3x3 matrix.")
        if not np.allclose(II, II.T, rtol=1e-10, atol=1e-12):
            raise ValueError("The inertia tensor must be symmetric.")
    
        lam, Q = np.linalg.eigh(II)
        if np.any(lam <= 0):
            raise ValueError("The inertia tensor must be positive definite.")
    
        ICOmega = np.asarray(ICOmega, dtype=float).reshape(3)
        if not np.all(np.isfinite(ICOmega)):
            raise ValueError("ICOmega must contain finite values.")
    
        if grid_phi < 8 or grid_theta < 16:
            raise ValueError("Use grid_phi >= 8 and grid_theta >= 16.")
    
        R = None
        if ICR is not None:
            R = np.asarray(ICR, dtype=float).reshape(3, 3)
            if (
                not np.allclose(R.T @ R, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
            ):
                raise ValueError("ICR must be a rotation matrix.")
    
        if overlay_spatial and R is None:
            raise ValueError("ICR is required when overlay_spatial=True.")
    
        pi0_body = II @ ICOmega
        magPi = float(np.linalg.norm(pi0_body))
        KE = float(0.5 * ICOmega @ pi0_body)
    
        # Unit-sphere grid used for both surfaces.
        phi = np.linspace(0, np.pi, grid_phi)
        theta = np.linspace(0, 2 * np.pi, grid_theta)
        PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
    
        unit_sphere = np.stack([
            np.sin(PHI) * np.cos(THETA),
            np.sin(PHI) * np.sin(THETA),
            np.cos(PHI),
        ])
        shape = unit_sphere.shape
    
        sphere = magPi * unit_sphere
        axes = np.sqrt(2 * KE * lam)
        ellipsoid = (
            Q @ (axes[:, None] * unit_sphere.reshape(3, -1))
        ).reshape(shape)
    
        # Compute the intersection in principal-axis components.
        #
        # On ||Pi|| = h, the energy equation becomes
        #     sum(c_i * Pi_i**2) = 0,
        # where c_i = 1/lam_i - 2*KE/h**2.
        curves = []
        points = []
        coincident = False
    
        s = np.linspace(0, 2 * np.pi, max(1200, 12 * grid_theta))
        eye = np.eye(3)
    
        if magPi == 0:
            points.append(np.zeros(3))
        else:
            inverse_moments = 1.0 / lam
            c = inverse_moments - 2 * KE / magPi**2
            eps = 1e-12 * np.max(inverse_moments)
    
            zero = np.flatnonzero(np.abs(c) <= eps)
            positive = np.flatnonzero(c > eps)
            negative = np.flatnonzero(c < -eps)
    
            if len(zero) == 3:
                coincident = True
    
            elif len(zero) == 2:
                # A great circle in the two-dimensional zero eigenspace.
                j, k = zero
                principal = magPi * (
                    eye[:, j, None] * np.cos(s)
                    + eye[:, k, None] * np.sin(s)
                )
                curves.append(Q @ principal)
    
            elif len(zero) == 1:
                z = zero[0]
    
                if len(positive) == 1 and len(negative) == 1:
                    # Critical energy: two great circles containing
                    # the intermediate-axis equilibria.
                    j, k = positive[0], negative[0]
                    for sign in (-1, 1):
                        direction = (
                            np.sqrt(-c[k]) * eye[:, j]
                            + sign * np.sqrt(c[j]) * eye[:, k]
                        )
                        direction /= np.linalg.norm(direction)
                        principal = magPi * (
                            eye[:, z, None] * np.cos(s)
                            + direction[:, None] * np.sin(s)
                        )
                        curves.append(Q @ principal)
                else:
                    # Extremal energy: two equilibrium points.
                    points.extend([
                        magPi * Q[:, z],
                        -magPi * Q[:, z],
                    ])
    
            else:
                # One coefficient has the opposite sign from the other two.
                if len(positive) == 1:
                    single, pair = positive[0], negative
                else:
                    single, pair = negative[0], positive
    
                j, k = pair
                root = np.sqrt(np.maximum(
                    -(c[j] * np.cos(s)**2 + c[k] * np.sin(s)**2)
                    / c[single],
                    0.0,
                ))
    
                for sign in (-1, 1):
                    principal = np.zeros((3, len(s)))
                    principal[j] = np.cos(s)
                    principal[k] = np.sin(s)
                    principal[single] = sign * root
                    principal *= magPi / np.linalg.norm(principal, axis=0)
                    curves.append(Q @ principal)
    
        fig = go.Figure()
    
        def add_surface(surface, color, opacity, name):
            fig.add_trace(go.Surface(
                x=surface[0],
                y=surface[1],
                z=surface[2],
                surfacecolor=np.zeros_like(surface[0]),
                colorscale=[[0, color], [1, color]],
                cmin=0,
                cmax=1,
                showscale=False,
                showlegend=True,
                opacity=opacity,
                name=name,
                hoverinfo="skip",
                lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            ))
    
        add_surface(
            sphere, "#AECDE8", sphere_opacity,
            "Angular-momentum sphere",
        )
        add_surface(
            ellipsoid, "#F2C4AE", ellipsoid_opacity,
            "Energy ellipsoid",
        )
    
        def add_curves(curve_list, color, name, dash="solid"):
            # None separates disconnected curves.
            coordinates = [[], [], []]
            for curve in curve_list:
                for axis in range(3):
                    coordinates[axis].extend(curve[axis].tolist())
                    coordinates[axis].append(None)
    
            if curve_list:
                fig.add_trace(go.Scatter3d(
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    mode="lines",
                    line=dict(color=color, width=8, dash=dash),
                    name=name,
                    connectgaps=False,
                    hoverinfo="skip",
                ))
    
        add_curves(curves, "#29243D", "Intersection")
    
        # Isolated equilibria cannot be represented by a curve.
        if points:
            pts = np.column_stack(points)
            fig.add_trace(go.Scatter3d(
                x=pts[0], y=pts[1], z=pts[2],
                mode="markers",
                marker=dict(size=6, color="#29243D"),
                name="Degenerate intersection",
            ))
    
        if overlay_spatial:
            rotated_ellipsoid = (
                R @ ellipsoid.reshape(3, -1)
            ).reshape(shape)
    
            # The rotated sphere is identical, so do not draw it twice.
            add_surface(
                rotated_ellipsoid, "#D0BCE3", 0.15,
                "Energy ellipsoid at R(0), spatial components",
            )
            add_curves(
                [R @ curve for curve in curves],
                "#9C4767",
                "Intersection rotated by R(0)",
                dash="dash",
            )
    
            if points:
                pts = R @ np.column_stack(points)
                fig.add_trace(go.Scatter3d(
                    x=pts[0], y=pts[1], z=pts[2],
                    mode="markers",
                    marker=dict(size=6, color="#9C4767"),
                    name="Rotated degenerate intersection",
                ))
    
        if coincident:
            fig.add_annotation(
                text="The two surfaces coincide; there is no unique intersection curve.",
                x=0.5, y=1.02,
                xref="paper", yref="paper",
                showarrow=False,
            )
    
        extent = 1.1 * max(magPi, float(np.max(axes)))
        if extent == 0:
            extent = 1.0
        axis_range = [-extent, extent]
    
        fig.update_layout(
            title=title or (
                "Angular-momentum sphere and energy ellipsoid "
                "(body components Π)"
            ),
            scene=dict(
                xaxis=dict(title="Π₁", range=axis_range, autorange=False),
                yaxis=dict(title="Π₂", range=axis_range, autorange=False),
                zaxis=dict(title="Π₃", range=axis_range, autorange=False),
                aspectmode="cube",
            ),
            legend=dict(orientation="h", y=-0.10),
            margin=dict(l=10, r=10, b=80, t=70),
            showlegend=True,
        )
    
        if show:
            fig.show()
    
        info = {
            "magPi": magPi,
            "KE": KE,
            "pi0_body": pi0_body,
            "pi0_spatial": R @ pi0_body if R is not None else None,
            "eigvals": lam,
            "eigvecs": Q,
            "intersection_curves": curves,
            "intersection_points": points,
            "surfaces_coincide": coincident,
        }
        return fig, info

    def plot_momentum_sphere(
        self,
        inertia=(3.0, 2.0, 1.0),
        h=1.0,
        curves_per_family=4,
        samples=500,
        show_arrows=True,
        sphere_opacity=0.85,
        perspective=True,
        sphere_inset=0.005,
        width=850,
        height=850,
        show=True,
    ):
        """
        Plot free rigid-body angular momentum curves on a fixed sphere.
    
        Parameters
        ----------
        inertia : length-3 sequence
            Principal moments, ordered I1 > I2 > I3 > 0.
        h : float
            Conserved angular momentum magnitude.
        curves_per_family : int
            Number of regular energy levels on each side of Ecrit.
            Each energy level produces two closed curves.
        samples : int
            Number of samples per closed curve.
        show_arrows : bool
            Show the direction of motion, Pi_dot = Pi x I^{-1} Pi.
        sphere_opacity : float, default 0.85
            Sphere opacity between 0 and 1:
            0 = transparent; 1 = opaque, hiding rear curves.
            Intermediate values allow rear curves to remain visible.
        perspective : bool
            True for perspective projection; False for orthographic.
        sphere_inset : float, default 0.005
            Fractional reduction of the rendered sphere radius to reduce
            overlap artifacts. Curves remain on the exact sphere |Pi| = h.
            This is a visual adjustment only.
        width, height : int
            Figure dimensions in pixels.
        show : bool
            Display the figure.
    
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive figure with an opacity slider.
    
        Notes
        -----
        The curves are exact parametrizations of the conserved-surface
        intersections, sampled for plotting. Their parameter is not time;
        arrows are calculated from Euler's equations.
    
        Plotly's transparent 3D rendering may produce minor depth-sorting
        artifacts. Rear curves are not explicitly dashed.
        """
        I = np.asarray(inertia, dtype=float)
    
        if I.shape != (3,) or not np.all(np.isfinite(I)):
            raise ValueError("inertia must contain three finite moments.")
        if not (I[0] > I[1] > I[2] > 0):
            raise ValueError("Require I1 > I2 > I3 > 0.")
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be positive.")
        if not isinstance(curves_per_family, (int, np.integer)):
            raise ValueError("curves_per_family must be an integer.")
        if not isinstance(samples, (int, np.integer)):
            raise ValueError("samples must be an integer.")
        if curves_per_family < 1 or samples < 32:
            raise ValueError("Use curves_per_family >= 1 and samples >= 32.")
        if not 0 <= sphere_opacity <= 1:
            raise ValueError("sphere_opacity must lie between 0 and 1.")
        if not 0 <= sphere_inset < 0.05:
            raise ValueError("Use 0 <= sphere_inset < 0.05.")
    
        invI = 1.0 / I
        Emin = h**2 / (2.0 * I[0])
        Ecrit = h**2 / (2.0 * I[1])
        Emax = h**2 / (2.0 * I[2])
    
        fig = go.Figure()
    
        # Slightly inset the rendered sphere to avoid overlap artifacts.
        polar = np.linspace(0.0, np.pi, 100)
        azimuth = np.linspace(0.0, 2.0 * np.pi, 200)
        p, a = np.meshgrid(polar, azimuth, indexing="ij")
        radius = (1.0 - sphere_inset) * h
    
        # Keep this as trace 0: the opacity slider targets it.
        fig.add_trace(go.Surface(
            x=radius * np.sin(p) * np.cos(a),
            y=radius * np.sin(p) * np.sin(a),
            z=radius * np.cos(p),
            surfacecolor=np.zeros_like(p),
            colorscale=[[0, "#DCE3EA"], [1, "#DCE3EA"]],
            showscale=False,
            opacity=sphere_opacity,
            hoverinfo="skip",
            showlegend=False,
            name="Angular momentum sphere",
            lighting=dict(
                ambient=0.65,
                diffuse=0.75,
                specular=0.08,
                roughness=0.9,
            ),
            lightposition=dict(x=100, y=150, z=200),
        ))
    
        def add_direction_arrow(P, color, group, index):
            """Draw an arrow tangent to the Euler vector field."""
            point = P[index]
            velocity = np.cross(point, invI * point)
            speed = np.linalg.norm(velocity)
    
            if speed <= 1e-12 * h**2 * np.max(invI):
                return
    
            tangent = velocity / speed
    
            fig.add_trace(go.Cone(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                u=[tangent[0]],
                v=[tangent[1]],
                w=[tangent[2]],
                anchor="center",
                sizemode="absolute",
                sizeref=0.07 * h,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                showlegend=False,
                legendgroup=group,
                hoverinfo="skip",
            ))
    
        def add_curve(
            P, color, name, group, energy,
            legend=False, line_width=3, arrows=2,
        ):
            fig.add_trace(go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode="lines",
                line=dict(color=color, width=line_width),
                name=name,
                legendgroup=group,
                showlegend=legend,
                hovertemplate=(
                    f"E = {energy:.5g}<br>"
                    "Π₁ = %{x:.4g}<br>"
                    "Π₂ = %{y:.4g}<br>"
                    "Π₃ = %{z:.4g}<extra></extra>"
                ),
            ))
    
            if show_arrows:
                for fraction in np.linspace(0.25, 0.75, arrows):
                    index = int(fraction * (len(P) - 1))
                    add_direction_arrow(P, color, group, index)
    
        def regular_curves(E):
            """Return both regular intersection components."""
            c = invI - 2.0 * E / h**2
    
            # Below Ecrit, Pi1 retains its sign; above Ecrit, Pi3 does.
            isolated = 0 if E < Ecrit else 2
            j, k = [index for index in range(3) if index != isolated]
    
            t = np.linspace(0.0, 2.0 * np.pi, samples)
            P = np.zeros((samples, 3))
            P[:, j] = np.cos(t)
            P[:, k] = np.sin(t)
    
            squared = -(
                c[j] * np.cos(t)**2 + c[k] * np.sin(t)**2
            ) / c[isolated]
    
            P[:, isolated] = np.sqrt(np.maximum(squared, 0.0))
    
            # Normalization preserves sum_i c_i Pi_i^2 = 0.
            P *= h / np.linalg.norm(P, axis=1, keepdims=True)
    
            other = P.copy()
            other[:, isolated] *= -1.0
            return P, other
    
        # Regular energy levels, excluding extrema and the separatrix.
        fractions = np.linspace(0.08, 0.88, curves_per_family)
    
        families = [
            (
                Emin + fractions * (Ecrit - Emin),
                "#154C79",  # Dark blue: below critical energy
                "Emin < E < Ecrit",
                "low",
            ),
            (
                Emax - fractions * (Emax - Ecrit),
                "#176B45",  # Dark green: above critical energy
                "Ecrit < E < Emax",
                "high",
            ),
        ]
    
        for energies, color, name, group in families:
            first = True
            for E in energies:
                for P in regular_curves(E):
                    add_curve(
                        P, color, name, group, E,
                        legend=first,
                    )
                    first = False
    
        # Separatrix: -A Pi1^2 + B Pi3^2 = 0.
        A = invI[1] - invI[0]
        B = invI[2] - invI[1]
        slope = np.sqrt(A / B)
        e2 = np.array([0.0, 1.0, 0.0])
    
        # Split the two great circles into four connecting branches.
        first = True
        for sign in (-1.0, 1.0):
            direction = np.array([1.0, 0.0, sign * slope])
            direction /= np.linalg.norm(direction)
    
            for start in (0.0, np.pi):
                t = np.linspace(start, start + np.pi, samples // 2)
                P = h * (
                    np.cos(t)[:, None] * e2
                    + np.sin(t)[:, None] * direction
                )
    
                add_curve(
                    P,
                    color="#D1495B",
                    name="Separatrix: E = Ecrit",
                    group="critical",
                    energy=Ecrit,
                    legend=first,
                    line_width=5,
                    arrows=1,
                )
                first = False
    
        # Principal-axis relative equilibria.
        stable = h * np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 0, 1], [0, 0, -1],
        ])
        unstable = h * np.array([
            [0, 1, 0], [0, -1, 0],
        ])
    
        for points, color, name in [
            (stable, "#202020", "Stable equilibria"),
            (unstable, "#D1495B", "Unstable equilibria"),
        ]:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=5, color=color),
                name=name,
                hovertemplate=(
                    "Π = (%{x:.3g}, %{y:.3g}, %{z:.3g})"
                    "<extra>%{fullData.name}</extra>"
                ),
            ))
    
        # Coordinate axes through the sphere.
        extent = 1.35 * h
    
        for index, label in enumerate(["Π₁", "Π₂", "Π₃"]):
            direction = np.eye(3)[index]
            endpoints = np.array([
                -extent * direction,
                extent * direction,
            ])
    
            fig.add_trace(go.Scatter3d(
                x=endpoints[:, 0],
                y=endpoints[:, 1],
                z=endpoints[:, 2],
                mode="lines",
                line=dict(color="#666666", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))
    
            tip = extent * direction
            fig.add_trace(go.Cone(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                anchor="tip",
                sizemode="absolute",
                sizeref=0.07 * h,
                colorscale=[[0, "#333333"], [1, "#333333"]],
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
            ))
    
            label_point = 1.10 * tip
            fig.add_trace(go.Scatter3d(
                x=[label_point[0]],
                y=[label_point[1]],
                z=[label_point[2]],
                mode="text",
                text=[label],
                textfont=dict(size=22, color="black"),
                showlegend=False,
                hoverinfo="skip",
            ))
    
        # Interactive opacity control, including the supplied initial value.
        opacity_values = np.unique(np.round(
            np.append(np.linspace(0.0, 1.0, 21), sphere_opacity),
            6,
        ))
        active = int(np.argmin(np.abs(opacity_values - sphere_opacity)))
    
        slider_steps = [
            dict(
                method="restyle",
                args=[{"opacity": float(value)}, [0]],
                label=f"{value:.2f}",
            )
            for value in opacity_values
        ]
    
        axis = dict(
            visible=False,
            range=[-1.55 * h, 1.55 * h],
            autorange=False,
        )
    
        fig.update_layout(
            title="Free rigid body: angular momentum curves",
            width=width,
            height=height,
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=65, b=170),
            scene=dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis,
                aspectmode="cube",
                bgcolor="white",
                uirevision="momentum-sphere-camera",
                camera=dict(
                    eye=dict(x=1.55, y=1.65, z=1.10),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(
                        type="perspective" if perspective else "orthographic"
                    ),
                ),
            ),
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.03,
                yanchor="top",
                itemsizing="constant",
                groupclick="togglegroup",
            ),
            sliders=[dict(
                active=active,
                steps=slider_steps,
                x=0.15,
                len=0.70,
                y=-0.16,
                currentvalue=dict(prefix="Sphere opacity: "),
                pad=dict(t=15),
            )],
        )
    
        if show:
            fig.show()
    
        return fig


    def plot_energy_ellipsoid(
        self,
        inertia=(3.0, 2.0, 1.0),
        E=0.5,
        curves_per_family=1,
        samples=600,
        ellipsoid_opacity=0.65,
        surface_inset=0.004,
        show_wireframe=True,
        perspective=True,
        show=True,
        show_spheres=True,
        sphere_opacity=0.12,
    ):
        """
        Plot a fixed energy ellipsoid and its intersections with
        angular-momentum spheres of different radii.
    
        Require I1 > I2 > I3 > 0 and E > 0.
    
        Colors
        ------
        Dark blue  : hcrit < h < hmax
        Dark green : hmin < h < hcrit
        Red        : h = hcrit (separatrix)
    
        Sphere surfaces are omitted. An opacity slider controls the
        ellipsoid surface. Its slight inset is a rendering adjustment;
        intersection curves use the exact energy and momentum equations.
    
        Returns
        -------
        plotly.graph_objects.Figure
        """
        I = np.asarray(inertia, dtype=float)
    
        if I.shape != (3,) or not np.all(np.isfinite(I)):
            raise ValueError("inertia must contain three finite moments.")
        if not (I[0] > I[1] > I[2] > 0):
            raise ValueError("Require I1 > I2 > I3 > 0.")
        if not np.isfinite(E) or E <= 0:
            raise ValueError("E must be positive.")
        if not 0 <= ellipsoid_opacity <= 1:
            raise ValueError("ellipsoid_opacity must lie between 0 and 1.")
        if not 0 <= surface_inset < 0.05:
            raise ValueError("Use 0 <= surface_inset < 0.05.")
        if (
            not isinstance(curves_per_family, (int, np.integer))
            or curves_per_family < 1
        ):
            raise ValueError("curves_per_family must be a positive integer.")
        if not isinstance(samples, (int, np.integer)) or samples < 32:
            raise ValueError("samples must be an integer >= 32.")
    
        invI = 1.0 / I
        axes = np.sqrt(2.0 * E * I)
    
        hmin, hcrit, hmax = axes[2], axes[1], axes[0]
    
        blue = "#154C79"
        green = "#176B45"
        red = "#D1495B"
    
        fig = go.Figure()
    
        # Fixed energy ellipsoid: Pi^T I^{-1} Pi = 2E.
        polar = np.linspace(0.0, np.pi, 100)
        azimuth = np.linspace(0.0, 2.0 * np.pi, 180)
        p, a = np.meshgrid(polar, azimuth, indexing="ij")
    
        inset_axes = (1.0 - surface_inset) * axes
    
        # Trace 0 is controlled by the opacity slider.
        fig.add_trace(go.Surface(
            x=inset_axes[0] * np.sin(p) * np.cos(a),
            y=inset_axes[1] * np.sin(p) * np.sin(a),
            z=inset_axes[2] * np.cos(p),
            surfacecolor=np.zeros_like(p),
            colorscale=[[0, "#DFE5EB"], [1, "#DFE5EB"]],
            opacity=ellipsoid_opacity,
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
            lighting=dict(
                ambient=0.7,
                diffuse=0.6,
                specular=0.08,
                roughness=0.9,
            ),
        ))
    
        def add_line(P, color, width=3, name=None, group=None,
                     legend=False, h=None):
            fig.add_trace(go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode="lines",
                line=dict(color=color, width=width),
                name=name,
                legendgroup=group,
                showlegend=legend,
                hoverinfo="skip" if h is None else None,
                hovertemplate=None if h is None else (
                    f"E = {E:.5g}<br>h = {h:.5g}<br>"
                    "Π₁ = %{x:.4g}<br>"
                    "Π₂ = %{y:.4g}<br>"
                    "Π₃ = %{z:.4g}<extra></extra>"
                ),
            ))
    
        # Light wireframe, slightly outside the inset surface but inside
        # the exact ellipsoid so that colored curves stand out.
        if show_wireframe:
            wire_axes = (1.0 - 0.5 * surface_inset) * axes
            t = np.linspace(0.0, 2.0 * np.pi, 250)
    
            for latitude in np.linspace(0.0, np.pi, 9)[1:-1]:
                P = np.column_stack([
                    wire_axes[0] * np.sin(latitude) * np.cos(t),
                    wire_axes[1] * np.sin(latitude) * np.sin(t),
                    np.full_like(t, wire_axes[2] * np.cos(latitude)),
                ])
                add_line(P, "#A3ABB3", width=1)
    
            t = np.linspace(0.0, np.pi, 200)
            for longitude in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                P = np.column_stack([
                    wire_axes[0] * np.sin(t) * np.cos(longitude),
                    wire_axes[1] * np.sin(t) * np.sin(longitude),
                    wire_axes[2] * np.cos(t),
                ])
                add_line(P, "#A3ABB3", width=1)
    
        def intersection_curves(h):
            """Exact intersection components for a noncritical radius."""
            c = invI - 2.0 * E / h**2
            isolated = 0 if h > hcrit else 2
            j, k = [index for index in range(3) if index != isolated]
    
            t = np.linspace(0.0, 2.0 * np.pi, samples)
            P = np.zeros((samples, 3))
            P[:, j] = np.cos(t)
            P[:, k] = np.sin(t)
    
            squared = -(
                c[j] * np.cos(t)**2 + c[k] * np.sin(t)**2
            ) / c[isolated]
    
            P[:, isolated] = np.sqrt(np.maximum(squared, 0.0))
            P *= h / np.linalg.norm(P, axis=1, keepdims=True)
    
            other = P.copy()
            other[:, isolated] *= -1
            return P, other
    
        # Choose radii by spacing h^2 within each admissible interval.
        fractions = (
            np.array([0.45])
            if curves_per_family == 1
            else np.linspace(0.15, 0.85, curves_per_family)
        )
    
        families = [
            (
                np.sqrt(hcrit**2 + fractions * (hmax**2 - hcrit**2)),
                blue,
                "h > hcrit",
                "above",
            ),
            (
                np.sqrt(hcrit**2 - fractions * (hcrit**2 - hmin**2)),
                green,
                "h < hcrit",
                "below",
            ),
        ]
    
        if not 0 <= sphere_opacity <= 1:
            raise ValueError("sphere_opacity must lie between 0 and 1.")
    
        if show_spheres:
            # Display one representative sphere from each family.
            # With curves_per_family=1, these match the two plotted radii exactly.
            for radii, color, name, group in families:
                radius = radii[len(radii) // 2]
    
                fig.add_trace(go.Surface(
                    x=radius * np.sin(p) * np.cos(a),
                    y=radius * np.sin(p) * np.sin(a),
                    z=radius * np.cos(p),
                    surfacecolor=np.zeros_like(p),
                    colorscale=[[0, color], [1, color]],
                    opacity=sphere_opacity,
                    showscale=False,
                    showlegend=False,
                    legendgroup=group,
                    name=f"Sphere: {name}",
                    hoverinfo="skip",
                    lighting=dict(
                        ambient=0.85,
                        diffuse=0.35,
                        specular=0.05,
                        roughness=0.9,
                    ),
                ))
    
    
        for radii, color, name, group in families:
            first = True
            for h in radii:
                for P in intersection_curves(h):
                    add_line(
                        P, color, width=5, name=name,
                        group=group, legend=first, h=h,
                    )
                    first = False
    
        # Critical intersection: two great circles on the hcrit sphere.
        slope = np.sqrt(
            (invI[1] - invI[0]) / (invI[2] - invI[1])
        )
        e2 = np.array([0.0, 1.0, 0.0])
        t = np.linspace(0.0, 2.0 * np.pi, samples)
    
        for index, sign in enumerate((-1.0, 1.0)):
            direction = np.array([1.0, 0.0, sign * slope])
            direction /= np.linalg.norm(direction)
    
            P = hcrit * (
                np.cos(t)[:, None] * e2
                + np.sin(t)[:, None] * direction
            )
            add_line(
                P, red, width=6,
                name="h = hcrit: separatrix",
                group="critical",
                legend=(index == 0),
                h=hcrit,
            )
    
        # Intermediate-axis equilibria shared by the critical curves.
        fig.add_trace(go.Scatter3d(
            x=[0.0, 0.0],
            y=[hcrit, -hcrit],
            z=[0.0, 0.0],
            mode="markers",
            marker=dict(size=5, color=red),
            name="Intermediate-axis equilibria",
            legendgroup="critical",
            showlegend=False,
            hovertemplate=(
                "Intermediate-axis equilibrium<br>"
                "Π = (%{x:.4g}, %{y:.4g}, %{z:.4g})"
                "<extra></extra>"
            ),
        ))
    
        # Axis labels outside the ellipsoid.
        for index, label in enumerate(["Π₁", "Π₂", "Π₃"]):
            point = np.zeros(3)
            point[index] = 1.15 * axes[index]
    
            fig.add_trace(go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],
                mode="text",
                text=[label],
                textfont=dict(size=21, color="#303030"),
                showlegend=False,
                hoverinfo="skip",
            ))
    
        opacity_values = np.unique(np.round(
            np.append(np.linspace(0.0, 1.0, 21), ellipsoid_opacity),
            6,
        ))
    
        fig.update_layout(
            title="Fixed energy ellipsoid: angular momentum intersections",
            width=900,
            height=750,
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=65, b=150),
            scene=dict(
                xaxis=dict(visible=False, range=[-1.25 * hmax, 1.25 * hmax]),
                yaxis=dict(visible=False, range=[-1.25 * hmax, 1.25 * hmax]),
                zaxis=dict(visible=False, range=[-1.25 * hmax, 1.25 * hmax]),
                aspectmode="data",
                bgcolor="white",
                uirevision="energy-ellipsoid-camera",
                camera=dict(
                    eye=dict(x=0.35, y=2.0, z=0.65),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(
                        type="perspective" if perspective else "orthographic"
                    ),
                ),
            ),
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.03,
                yanchor="top",
                itemsizing="constant",
                groupclick="togglegroup",
            ),
            sliders=[dict(
                active=int(np.argmin(
                    np.abs(opacity_values - ellipsoid_opacity)
                )),
                x=0.15,
                len=0.70,
                y=-0.14,
                currentvalue=dict(prefix="Ellipsoid opacity: "),
                steps=[
                    dict(
                        method="restyle",
                        args=[{"opacity": float(value)}, [0]],
                        label=f"{value:.2f}",
                    )
                    for value in opacity_values
                ],
            )],
        )
    
        if show:
            fig.show()
    
        return fig
######################
    
    # def runga_kutta_method(self, dt, Tmax, parameters, ICs):
    #     """
    #     Integrate a rigid body's pose and momenta with a **Lie-group RK4** stepper.

    #     This integrator advances the composite state
    #         X = [[R, o], omega, p, Xc]
    #     over t ∈ [0, Tmax] using a fourth-order Runge–Kutta scheme that:
    #     • updates attitude R ∈ SO(3) via a **left-increment** R ← exp(dt·ω̄) @ R
    #         (preserves orthonormality by construction),
    #     • updates translation, linear momentum, and auxiliary/controller state
    #         with classical RK4 stage averaging.

    #     ---------- State & Dynamics ----------
    #     State layout:
    #         R     : (3,3) rotation matrix (orthonormal, det=+1)
    #         o     : (3,)  position
    #         omega : (3,)  body angular velocity
    #         p     : (3,)  linear momentum
    #         Xc    : (...) auxiliary/controller state (any shape)

    #     Required hooks on `self`:
    #         rigid_body_system(parameters, t, X) -> tuple
    #             Returns (theta, n, doto, dp, dspi, dXc) at (t, X), where
    #             • theta = ||omega|| (scalar),
    #             • n     = omega / ||omega|| (unit axis, zero if tiny),
    #             • doto  = ȯ (linear velocity),
    #             • dp    = ṗ (external/actuator forces),
    #             • dspi  = spatial spin rate (time derivative of spatial angular momentum),
    #             • dXc   = time derivative of Xc.
    #             The product w := theta * n is treated as the **body angular rate** used
    #             to build RK4 stage predictors on SO(3).
    #         exp_map(v: (3,)) -> (3,3)
    #             Exponential map on SO(3): exp_map(v) = exp(hat(v)).
    #         r_from_quaternions / hat_matrix (optional elsewhere): not used here.

    #     Parameters
    #     ----------
    #     dt : float
    #         Fixed time step (seconds).
    #     Tmax : float
    #         Total simulated time horizon (seconds). The loop executes N=floor(Tmax/dt) steps.
    #         Use ceil if you prefer to land exactly on Tmax.
    #     parameters : dict
    #         Model parameters; recognized keys:
    #         • 'II' : (3,3) SPD body inertia matrix (default I3).
    #         • Any additional keys are forwarded to `rigid_body_system`.
    #     ICs : list/tuple
    #         Initial composite state in the same layout as X: [[R0, o0], omega0, p0, Xc0].

    #     Returns
    #     -------
    #     Xout : list
    #         Sequence of states over the grid (length N+1), including the initial state `ICs`.
    #         Each item has the layout [[R, o], omega, p, Xc].

    #     Algorithm (per step)
    #     --------------------
    #     1) Evaluate k1 at (t, X).
    #     2) Build stage predictor Y1 at t+dt/2 with exp(0.5·dt·w1) and Euler drifts; evaluate k2.
    #     3) Build stage predictor Y2 at t+dt/2 with exp(0.5·dt·w2); evaluate k3.
    #     4) Build stage predictor Y3 at t+dt with exp(dt·w3); evaluate k4.
    #     5) Form RK4 weighted averages:
    #         w̄   = (w1 + 2w2 + 2w3 + w4)/6,
    #         d•̄  = (k1 + 2k2 + 2k3 + k4)/6 for each translational/momentum/controller rate.
    #     6) Update:
    #         R_new   = exp(dt·w̄) @ R,
    #         o_new   = o   + dt·dō̄,
    #         p_new   = p   + dt·dp̄,
    #         spi_new = R II Rᵀ omega + dt·ds̄,
    #         omega_new = R_new II⁻¹ R_newᵀ spi_new,
    #         Xc_new  = Xc + dt·dXc̄.

    #     Numerical Properties & Notes
    #     ----------------------------
    #     • SO(3) update is group-consistent and maintains RᵀR=I, det R=+1 up to machine precision.
    #     • Stage times are (t, t+dt/2, t+dt/2, t+dt); weights (1,2,2,1)/6 (classical RK4).
    #     • The angular stage vector w_i := theta_i n_i must represent a **body-frame** rate to
    #     match the left-increment kinematics Ṙ = R·hat(omega_body).
    #     • A small-angle safeguard is applied implicitly by exp_map; you may add an explicit check
    #     if your exp_map assumes ||v|| > 0.

    #     Caveats
    #     -------
    #     • If your dynamics produce **spatial** quantities, convert them to/from the body frame
    #     consistently (this routine assumes body w_i and spatial momentum spin dynamics).
    #     • The loop uses floor(Tmax/dt); consider adjusting if exact endpoint alignment is required.

    #     Example
    #     -------
    #     >>> params = {'II': np.diag([0.1, 0.2, 0.3])}
    #     >>> R0 = np.eye(3); o0 = np.zeros(3); omega0 = np.array([0.1, 0.0, 0.0])
    #     >>> p0 = np.zeros(3); Xc0 = np.zeros(2)
    #     >>> traj = sim.runga_kutta_method(0.01, 1.0, params, [[R0, o0], omega0, p0, Xc0])
    #     >>> R1 = traj[-1][0][0]
    #     >>> np.allclose(R1.T @ R1, np.eye(3), atol=1e-12)
    #     True
    #     """
    #     II = parameters.get('II', np.eye(3))
    #     invII = np.linalg.inv(II)

    #     # steps: floor hits t in [0, N*dt); use ceil if you want to land on Tmax exactly
    #     N = int(np.floor(Tmax / dt + 1e-12))
    #     t0 = 0.0

    #     X = ICs
    #     Xout = [X]
    #     self.state = ICs

    #     for k in range(N):
    #         t = t0 + k*dt
    #         R, o, omega, p, Xc = X[0][0], X[0][1], X[1], X[2], X[3]

    #         # k1 at (t, X)
    #         th1, n1, do1, dp1, ds1, dXc1 = self.rigid_body_system(parameters, t, X)

    #         # k2 from Y1 at t+dt/2
    #         w1 = th1 * n1
    #         Y1 = [[ self.exp_map(0.5*dt * w1) @ R, o + 0.5*dt*do1 ],
    #             omega + 0.5*dt*w1, p + 0.5*dt*dp1, Xc + 0.5*dt*dXc1]
    #         th2, n2, do2, dp2, ds2, dXc2 = self.rigid_body_system(parameters, t + 0.5*dt, Y1)

    #         # k3 from Y2 at t+dt/2
    #         w2 = th2 * n2
    #         Y2 = [[ self.exp_map(0.5*dt * w2) @ R, o + 0.5*dt*do2 ],
    #             omega + 0.5*dt*w2, p + 0.5*dt*dp2, Xc + 0.5*dt*dXc2]
    #         th3, n3, do3, dp3, ds3, dXc3 = self.rigid_body_system(parameters, t + 0.5*dt, Y2)

    #         # k4 from Y3 at t+dt
    #         w3 = th3 * n3
    #         Y3 = [[ self.exp_map(dt * w3) @ R, o + dt*do3 ],
    #             omega + dt*w3, p + dt*dp3, Xc + dt*dXc3]
    #         th4, n4, do4, dp4, ds4, dXc4 = self.rigid_body_system(parameters, t + dt, Y3)

    #         # RK4 weights
    #         w_bar   = (th1*n1 + 2*th2*n2 + 2*th3*n3 + th4*n4) / 6.0
    #         do_bar  = (do1     + 2*do2    + 2*do3    + do4   ) / 6.0
    #         dp_bar  = (dp1     + 2*dp2    + 2*dp3    + dp4   ) / 6.0
    #         ds_bar  = (ds1     + 2*ds2    + 2*ds3    + ds4   ) / 6.0
    #         dXc_bar = (dXc1    + 2*dXc2   + 2*dXc3   + dXc4  ) / 6.0

    #         # group-consistent updates
    #         theta = np.linalg.norm(w_bar)
    #         if theta > 1e-12:
    #             R_new = self.exp_map(dt * w_bar) @ R
    #         else:
    #             R_new = R

    #         o_new   = o   + dt * do_bar
    #         p_new   = p   + dt * dp_bar
    #         spi_new = R @ II @ R.T @ omega + dt * ds_bar
    #         omega_new = R_new @ invII @ R_new.T @ spi_new
    #         Xc_new  = Xc + dt * dXc_bar

    #         X = [[R_new, o_new], omega_new, p_new, Xc_new]
    #         Xout.append(X)
    #         self.state = X

    #     self.trajectory = Xout
    #     return Xout

    # def _rk4_function(self, dtk, X_base, tk, Xk, parameters, invII=None):
    #     """
    #     One RK substep: advance a state Xk forward by dtk using dynamics at (tk, Xk).
    #     Returns a new temporary state (R_next, o_next, omega_next, p_next, Xc_next)
    #     built RELATIVE TO Xk (not X_base).
    #     """
    #     II = parameters.get('II', np.eye(3))
    #     if invII is None:
    #         invII = np.linalg.inv(II)

    #     # dynamics at the stage point (tk, Xk)
    #     theta, n, doto, dp, dspi, dXc = self.rigid_body_system(parameters, tk, Xk)

    #     # unpack the STAGE BASE (use Xk as the reference)
    #     Rk, ok   = Xk[0]
    #     omegak   = Xk[1]
    #     pk       = Xk[2]
    #     Xck      = Xk[3]

    #     # incremental rotation from stage angular vector w = theta*n
    #     w = theta * n
    #     R_next = self.exp_map(dtk * w) @ Rk

    #     # advance translational/momentum/controller from STAGE BASE
    #     o_next   = ok  + dtk * doto
    #     p_next   = pk  + dtk * dp
    #     spi_next = Rk @ II @ Rk.T @ omegak + dtk * dspi    # spatial angular momentum
    #     omega_next = R_next @ invII @ R_next.T @ spi_next  # body rate at new frame
    #     Xc_next = Xck + dtk * dXc

    #     return [[R_next, o_next], omega_next, p_next, Xc_next]


    # --- Rigid Body EKF --- OBSOLTE moved to EKF repo

    # def set_sensor(self, fn: Callable[..., tuple]) -> None:
    #     """
    #     Register a **sensor callback** used to simulate/ingest IMU-like measurements.

    #     The callback will be invoked by your code (e.g., inside tests or higher-level
    #     loops) to produce:
    #     - a 3-vector gyro reading, and
    #     - two 3-vectors for the measured body-frame directions of two inertial axes
    #         (by convention here: e1 and e3, but your implementation may choose them).

    #     Expected callable signature
    #     ---------------------------
    #     fn(R: np.ndarray, omega_body: np.ndarray, *[, ...]) -> tuple
    #         Parameters
    #         ----------
    #         R : (3,3) ndarray
    #             Current attitude (body → inertial).
    #         omega_body : (3,) ndarray
    #             Body-frame angular velocity used to form the gyro measurement.
    #         ... : optional keyword args
    #             Noise stds, RNG handle, flags (e.g., renormalize), etc.

    #         Returns
    #         -------
    #         Omega_meas : (3,) ndarray
    #             Gyro measurement (usually `omega_body + noise`).
    #         A_n_meas : (3,) ndarray
    #             Noisy measurement of `Rᵀ e1` (direction #1 expressed in body frame).
    #         A_g_meas : (3,) ndarray
    #             Noisy measurement of `Rᵀ e3` (direction #2 expressed in body frame).

    #     Notes
    #     -----
    #     - Your EKF’s measurement covariance `Σ_m` should be compatible with the
    #     distribution of `A_n_meas` and `A_g_meas` produced by this function.
    #     - If you dynamically choose the inertial axes (not strictly e1/e3), keep the
    #     EKF’s H/S construction consistent with that choice.

    #     Examples
    #     --------
    #     >>> def my_sensor(R, omega_body, sigma_omega=5e-3, sigma_dir=2e-2):
    #     ...     # return (Omega_meas, A_n_meas, A_g_meas)
    #     ...     ...
    #     >>> rb.set_sensor(my_sensor)
    #     """
    #     self.sensor = fn

    # def set_KF_innovation(self, fn: Callable[..., np.ndarray]) -> None:
    #     """
    #     Register the **innovation function** L = y - ŷ used by the EKF update.

    #     The callback computes the stacked residual for two direction measurements.
    #     It must return a column vector shaped (6,1) (or a 1-D array convertible to that),
    #     consistent with `H ∈ ℝ^{6×3}`.

    #     Expected callable signature
    #     ---------------------------
    #     fn(R_pred_minus: np.ndarray, A_n: np.ndarray, A_g: np.ndarray) -> np.ndarray
    #         Parameters
    #         ----------
    #         R_pred_minus : (3,3) ndarray
    #             Predicted-minus attitude used to form the predicted measurements.
    #         A_n : (3,) ndarray
    #             Measured body-frame direction of inertial axis #1 (e.g., `Rᵀ e1`).
    #         A_g : (3,) ndarray
    #             Measured body-frame direction of inertial axis #2 (e.g., `Rᵀ e3`).

    #         Returns
    #         -------
    #         L : (6,1) ndarray
    #             Innovation (residual) stacked as:
    #                 L = vec([A_n; A_g] - [R^-ᵀ e1; R^-ᵀ e3])

    #     Notes
    #     -----
    #     - If you change which inertial axes are used (not strictly e1/e3), this
    #     function must mirror that choice so that `ŷ` matches how H is built.
    #     - Returning a 1-D array of length 6 is acceptable; it will be reshaped to (6,1)
    #     by the EKF before multiplication.

    #     Examples
    #     --------
    #     >>> def my_innovation(Rm, A_n, A_g):
    #     ...     e1 = np.array([1.,0.,0.]); e3 = np.array([0.,0.,1.])
    #     ...     y      = np.vstack([A_n, A_g])
    #     ...     y_pred = np.vstack([Rm.T @ e1, Rm.T @ e3])
    #     ...     return (y - y_pred).reshape(-1, 1)
    #     >>> rb.set_KF_innovation(my_innovation)
    #     """
    #     self.kf_innovation = fn

    # def _linearization_attitude_kinematics(self, DeltaT: float, Omega: np.ndarray, R_for_H: np.ndarray):
    #     """
    #     A_k-1 = I 
    #     G_k-1 = (DeltaT ** 0.5) * R @ Phi(-DeltaT * Omega)
    #     H_k-1 = [ -R^T hat(e1) R ; -R^T hat(e2) R ; -R^T hat(e3) R ]  (stacked 6x3), using predicted-minus attitude.
    #     Uses the identity R^T hat(e) R = hat(R^T e) for efficiency.
    #     """
    #     I3 = np.eye(3)
    #     Omega = np.asarray(Omega, float).reshape(3,)
    #     R = np.asarray(R_for_H, float).reshape(3, 3)

    #     # A and G
    #     A_km1 = np.eye(3) #self.exp_map(DeltaT * Omega) #
    #     G_km1 = (DeltaT) * R @  self._Phi_SO3(-DeltaT * Omega) # sqrt Brownian

    #     # H using e1 & e3
    #     e1 = np.array([1., 0., 0.])
    #     e2 = np.array([0., 1., 0.])
    #     e3 = np.array([0., 0., 1.])
    #     H1 = -self.hat_matrix(R.T @ e1) 
    #     H2 = -self.hat_matrix(R.T @ e2) 
    #     H3 = -self.hat_matrix(R.T @ e3)
    #     H_km1 = np.vstack([H1, H2, H3])    # (9,3)

    #     return A_km1, G_km1, H_km1

    # # --- EKF predict/update (intrinsic, e1 & e3) ---

    # def predict_update_attitude(
    #     self,
    #     DeltaT: float,
    #     Omega_km1: np.ndarray,       # (3,) body angular velocity
    #     R_previous: np.ndarray,      # (3,3) previous attitude (k-1)
    #     P_previous: np.ndarray,      # (3,3) previous covariance (k-1)
    #     Sigma_q: np.ndarray,         # (3,3) process noise cov (gyro PSD discretized)
    #     Sigma_m: np.ndarray,         # (6,6) meas noise cov for stacked [R^T e1; R^T e3]
    #     A_1_meas: np.ndarray,        # (3,) measured R^T e1
    #     A_2_meas: np.ndarray,        # (3,) measured R^T e2
    #     A_3_meas: np.ndarray,        # (3,) measured R^T e3
    # ):
    #     """
    #     Intrinsic EKF on SO(3) with two direction measurements.

    #     Discretization (ΔT in the correction):
    #         R_k^- = R_{k-1} · exp(ΔT · Ω_{k-1})
    #         P_k^- = A P_{k-1} A^T + G Σ_q G^T,    A = I - ΔT·hat(Ω_{k-1}),  G = √ΔT·I
    #         H_k   = [ -hat(R_k^-T e1) ; -hat(R_k^-T e3) ]
    #         K_k   = P_k^- H_k^T (H_k P_k^- H_k^T + Σ_m)^{-1}
    #         R_k   = R_k^- · exp(ΔT · K_k (y_k - ŷ_k^-))
    #         P_k   = (I - K_k H_k) P_k^- (I - K_k H_k)^T + K_k Σ_m K_k^T

    #     Shapes:
    #         Ω, delta: (3,),  R: (3,3),  P, Σ_q: (3,3),  Σ_m, S: (9,9),
    #         H: (6,3),  K: (3,6),  innovation L: (6,1) or (6,)

    #     Notes:
    #         • Ω must be **body-frame** gyro rate to match A’s definition.
    #         • Uses Joseph update and light symmetrization for numerical SPD robustness.
    #     """
    #     # --- shape checks
    #     assert R_previous.shape == (3, 3)
    #     assert P_previous.shape == (3, 3)
    #     assert Sigma_q.shape == (3, 3)
    #     assert Sigma_m.shape == (9, 9)
    #     assert A_1_meas.shape == (3,)
    #     assert A_2_meas.shape == (3,)
    #     assert A_3_meas.shape == (3,)

    #     I3 = np.eye(3)

    #     # 1) State prediction
    #     R_pred_minus = R_previous @ self.exp_map(DeltaT * Omega_km1) 

    #     # 2) Linearize at predicted-minus attitude (H_k at R_k^-)
    #     A_km1, G_km1, H_km1 = self._linearization_attitude_kinematics(DeltaT, Omega_km1, R_pred_minus)

    #     # 3) Covariance prediction
    #     P_pred_minus = A_km1 @ P_previous @ A_km1.T + G_km1 @ Sigma_q @ G_km1.T

    #     # 4) Innovation 
    #     L = self.kf_innovation(R_pred_minus, A_1_meas, A_2_meas, A_3_meas)

    #     # 5) Kalman gain (stable solve; enforce symmetry + tiny jitter on S)
    #     S = H_km1 @ P_pred_minus @ H_km1.T + Sigma_m
    #     # Symmetrize for numerical stability
    #     S = 0.5 * (S + S.T)
    #     # Add a small jitter (size based on H_km1 output dimension)
    #     S += 1e-12 * np.eye(S.shape[0])

    #     # K = P^- H^T S^{-1}  via solve on S^T · K^T = (H P^-) → K = [(S^T)\(H P^-)]^T
    #     K = DeltaT * np.linalg.solve(S.T, (H_km1 @ P_pred_minus)).T

    #     # 6) correction with dt and clamp
    #     # delta = (DeltaT * (K @ L)).reshape(3,)
    #     delta = ((K @ L)).reshape(3,)
    #     n = np.linalg.norm(delta)
    #     if n > 0.2:       # ~11.5 deg cap
    #         delta *= 0.2 / n
    #     # R_pred = R_pred_minus @ self.exp_map(delta) #Left invariant outputs
    #     R_pred = self.exp_map(delta) @ R_pred_minus #Right invariant outputs


    #     # 7) Covariance update (Joseph + symmetrize)
    #     IKH = (I3 - K @ H_km1)
    #     P_pred = IKH @ P_pred_minus @ IKH.T + K @ Sigma_m @ K.T
    #     P_pred = 0.5 * (P_pred + P_pred.T)

    #     return R_pred, P_pred, K, H_km1, S

    # def run_offline_EKF_analysis(self, trajectory, dt=0.01, 
    #                             Sigma_q_factor=1.0, Sigma_m_factor=1.0,
    #                             sigma_omega=5e-3, sigma_dir=2e-2,
    #                             sigma_init_deg=10.0):
    #     """
    #     Run an offline attitude EKF over a simulated trajectory
    #     and produce diagnostic plots.

    #     Parameters
    #     ----------
    #     trajectory : list
    #         List of simulation states [(X_k)], where each X_k contains
    #         [ [R, o], omega, p, Xc ], i.e. rotation, angular velocity, etc.
    #     dt : float
    #         Time step [s].
    #     Sigma_q : ndarray(3x3), optional
    #         Process noise covariance. If None, default uses (sigma_omega**2)*I3.
    #     Sigma_m : ndarray(9x9), optional
    #         Measurement noise covariance. If None, default uses (sigma_dir**2)*I9.
    #     sigma_omega, sigma_dir : float
    #         Sensor noise standard deviations.
    #     sigma_init_deg : float
    #         Initial attitude uncertainty [deg].
    #     """

    #     # --- defaults ---
    #     sigma_omega_discrete = sigma_omega /(dt**0.5) # Scaled to discrete time step
    #     Sigma_q = Sigma_q_factor * (sigma_omega ** 2) * np.eye(3)
    #     Sigma_m = Sigma_m_factor * (sigma_dir ** 2) * np.eye(9)

    #     deg_to_rad = np.pi / 180.0
    #     Sigma_p0 = (sigma_init_deg * deg_to_rad)**2 * np.eye(3)

    #     # --- initial filter state ---
    #     R_hat = trajectory[0][0][0]  # use true initial orientation
    #     P_hat = Sigma_p0.copy()

    #     # --- preallocate metrics ---
    #     N = len(trajectory)
    #     t = np.arange(N) * dt
    #     err_deg, trace_err, lambda_max, ang_1sigma_deg = [], [], [], []

    #     # --- helper: rotation angle from R ---
    #     def angle_from_R(R):
    #         c = (np.trace(R) - 1.0) * 0.5
    #         c = float(np.clip(c, -1.0, 1.0))
    #         return np.arccos(c)

    #     # --- main loop ---
    #     for k in range(N):
    #         Xk = trajectory[k]
    #         R_true = Xk[0][0]
    #         omega_spatial = Xk[1]
    #         omega_body = R_true.T @ omega_spatial

    #         # noisy sensor measurements
    #         Omega_meas, A_1_meas, A_2_meas, A_3_meas = self.sensor(
    #             dt, R_true, omega_body, sigma_omega, sigma_dir
    #         )

    #         # EKF step
    #         R_hat, P_hat, K, H, S = self.predict_update_attitude(
    #             DeltaT=dt,
    #             Omega_km1=Omega_meas,
    #             R_previous=R_hat,
    #             P_previous=P_hat,
    #             Sigma_q=Sigma_q,
    #             Sigma_m=Sigma_m,
    #             A_1_meas=A_1_meas,
    #             A_2_meas=A_2_meas,
    #             A_3_meas=A_3_meas,
    #         )

    #         # attitude error metrics
    #         R_err = R_hat.T @ R_true
    #         err_deg.append(np.degrees(angle_from_R(R_err)))
    #         trace_err.append(3.0 - np.trace(R_err))

    #         # covariance quality
    #         P_sym = 0.5 * (P_hat + P_hat.T)
    #         w = np.linalg.eigvalsh(P_sym)
    #         lam_max = float(w[-1])
    #         lambda_max.append(lam_max)
    #         ang_1sigma_deg.append(np.degrees(np.sqrt(max(lam_max, 0.0))))

    #     # --- summary printout ---
    #     print("==== Offline EKF Analysis ====")
    #     print(f"Final attitude error: {err_deg[-1]:.3f} deg")
    #     print(f"Median attitude error: {np.median(err_deg):.3f} deg")
    #     print(f"Final trace error: {trace_err[-1]:.6f}")
    #     print(f"Median trace error: {np.median(trace_err):.6f}")
    #     print(f"Final √λ_max(P): {ang_1sigma_deg[-1]:.3f} deg (1σ equivalent)")

    #     # --- plots ---
    #     # 1. Trace error vs time
    #     fig_line = go.Figure()
    #     fig_line.add_trace(go.Scatter(x=t, y=trace_err, mode="lines",
    #                                 name="ε_tr = 3 - tr(R_trueᵀ R_hat)"))
    #     fig_line.update_layout(
    #         title="Trace misalignment vs time (offline EKF on simulated trajectory)",
    #         xaxis_title="Time (s)",
    #         yaxis_title="Trace error (0 = perfect alignment)",
    #     )

    #     # 2. Histogram of trace error
    #     fig_hist = go.Figure()
    #     fig_hist.add_trace(go.Histogram(x=trace_err, nbinsx=50, name="trace error"))
    #     fig_hist.update_layout(
    #         title="Histogram of trace misalignment (offline EKF)",
    #         xaxis_title="Trace error",
    #         yaxis_title="Count",
    #         bargap=0.05,
    #     )

    #     # 3. Covariance eigenvalues
    #     fig_cov = go.Figure()
    #     fig_cov.add_trace(go.Scatter(x=t, y=lambda_max, mode="lines", name="λ_max(P) [rad²]"))
    #     fig_cov.add_trace(go.Scatter(x=t, y=ang_1sigma_deg, mode="lines",
    #                                 name="√λ_max(P) [deg]", yaxis="y2"))
    #     fig_cov.update_layout(
    #         title="Covariance magnitude over time (offline EKF)",
    #         xaxis_title="Time (s)",
    #         yaxis=dict(title="Largest eigenvalue λ_max(P) [rad²]"),
    #         yaxis2=dict(title="1σ angle √λ_max(P) [deg]", overlaying="y", side="right"),
    #         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    #     )

    #     # show
    #     fig_line.show()
    #     fig_hist.show()
    #     fig_cov.show()

    #     return dict(
    #         time=t,
    #         err_deg=np.array(err_deg),
    #         trace_err=np.array(trace_err),
    #         lambda_max=np.array(lambda_max),
    #         ang_1sigma_deg=np.array(ang_1sigma_deg),
    #         figs=(fig_line, fig_hist, fig_cov),
    #     )

    # def auto_tune_EKF(self, trajectory, Sigma_p0,
    #                 sigma_omega=5e-3, sigma_dir=2e-2,
    #                 dt=0.01, steps=10000,
    #                 omega_body=None,
    #                 q_scales=[0.1, 1.0, 10.0, 100.0],
    #                 m_scales=[0.01, 0.1, 1.0, 10.0],
    #                 verbose=True):
    #     """
    #     Auto-tune EKF by sweeping over Sigma_q and Sigma_m scales.
    #     Evaluates performance based on final and median attitude RMSE.

    #     Parameters
    #     ----------
    #     trajectory : ndarray
    #         The trajectory.
    #     Sigma_p0 : ndarray(3x3)
    #         Initial covariance.
    #     sigma_omega, sigma_dir : float
    #         Sensor noise standard deviations (gyro, direction).
    #     dt : float
    #         Time step [s].
    #     omega_body : ndarray(3,), optional
    #         Constant body angular velocity [rad/s]; if None → [1,1,1].
    #     q_scales, m_scales : list of floats
    #         Multiplicative scales for Sigma_q and Sigma_m.
    #     verbose : bool
    #         Print intermediate results.

    #     Returns
    #     -------
    #     best_config : tuple (q_scale, m_scale)
    #     results : list of tuples (q_scale, m_scale, final_err, median_err)
    #     """
    #     sigma_omega_discrete = sigma_omega / (dt**0.5) #Scaled to discrete time step
    #     # --- search results ---
    #     best_config = None
    #     best_score = np.inf
    #     results = []

    #     for q_scale in q_scales:
    #         for m_scale in m_scales:
    #             # Covariances
    #             Sigma_q = q_scale * (sigma_omega ** 2) * np.eye(3)
    #             Sigma_m = m_scale * (sigma_dir ** 2) * np.eye(9)

    #             # Initialize filter
    #             R_hat = deepcopy(trajectory[0][0][0])  # initial attitude
    #             P_hat = Sigma_p0.copy()

    #             # Metrics
    #             err_deg = []

    #             for Xk in trajectory:
    #                 R_true = Xk[0][0]
    #                 omega_spatial = Xk[1]
    #                 omega_body_k = R_true.T @ omega_spatial

    #                 # generate noisy measurements
    #                 Omega_meas, A_1_meas, A_2_meas, A_3_meas = self.sensor(
    #                     dt, R_true, omega_body_k, sigma_omega, sigma_dir
    #                 )

    #                 # EKF predict + update
    #                 R_hat, P_hat, *_ = self.predict_update_attitude(
    #                     DeltaT=dt,
    #                     Omega_km1=Omega_meas,
    #                     R_previous=R_hat,
    #                     P_previous=P_hat,
    #                     Sigma_q=Sigma_q,
    #                     Sigma_m=Sigma_m,
    #                     A_1_meas=A_1_meas,
    #                     A_2_meas=A_2_meas,
    #                     A_3_meas=A_3_meas,
    #                 )

    #                 # compute attitude error
    #                 R_err = R_hat.T @ R_true
    #                 c = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    #                 angle_rad = np.arccos(c)
    #                 err_deg.append(np.degrees(angle_rad))

    #             # summarize
    #             final_err = err_deg[-1]
    #             median_err = np.median(err_deg)
    #             score = final_err + median_err

    #             results.append((q_scale, m_scale, final_err, median_err))

    #             if verbose:
    #                 print(f"[q={q_scale:.2e}, m={m_scale:.2e}] Final: {final_err:.3f}°, Median: {median_err:.3f}°")

    #             if score < best_score:
    #                 best_score = score
    #                 best_config = (q_scale, m_scale)

    #     print("\n✅ Best configuration found:")
    #     print(f"Sigma_q = {best_config[0]:.2e} × base (σ_ω² I₃)")
    #     print(f"Sigma_m = {best_config[1]:.2e} × base (σ_dir² I₉)")

    #     return best_config, results

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
