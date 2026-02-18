import numpy as np


class ObservationModel:
    """
    JAIS inspection observation model (Sections B + C).

    - Perception cone / in-view: Eq. (19)
    - Illumination: Blinn–Phong Eq. (20) with halfway vector Eq. (21)
    - Light/material params: Table A1
    """

    def __init__(
        self,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        base_rgb: np.ndarray,
        chief_radius: float,
        bright_thresh: float = 0.83,
        dark_thresh: float = 0.30,
        # --- Table A1: light properties ---
        light_ambient=(1.0, 1.0, 1.0),
        light_diffuse=(1.0, 1.0, 1.0),
        light_specular=(1.0, 1.0, 1.0),
        # --- Table A1: chief surface properties ---
        chief_ambient=(0.4, 0.4, 0.4),
        chief_diffuse=(0.1, 0.1, 0.1),
        chief_specular=(1.0, 1.0, 1.0),
        shininess: float = 100.0,
    ):
        self.surface_points = np.asarray(surface_points, dtype=np.float64)
        self.surface_normals = np.asarray(surface_normals, dtype=np.float64)
        self.base_rgb = np.asarray(base_rgb, dtype=np.float64)

        if self.surface_points.shape != self.surface_normals.shape:
            raise ValueError("surface_points and surface_normals must have the same shape")
        if self.base_rgb.shape != self.surface_points.shape:
            raise ValueError("base_rgb must have the same shape as surface_points")

        self.Np = self.surface_points.shape[0]
        self.r_c = float(chief_radius)

        self.bright_thresh = float(bright_thresh)
        self.dark_thresh = float(dark_thresh)

        # Table A1
        self.Ia = np.asarray(light_ambient, dtype=np.float64).reshape(3)
        self.Id = np.asarray(light_diffuse, dtype=np.float64).reshape(3)
        self.Is = np.asarray(light_specular, dtype=np.float64).reshape(3)

        self.ka = np.asarray(chief_ambient, dtype=np.float64).reshape(3)
        self.kd = np.asarray(chief_diffuse, dtype=np.float64).reshape(3)
        self.ks = np.asarray(chief_specular, dtype=np.float64).reshape(3)

        self.beta = float(shininess)

    # Sun direction r^S (Eq. 18): unit vector chief->sun in Hill x-y plane
    @staticmethod
    def sun_direction(theta_s: float) -> np.ndarray:
        return np.array([np.cos(theta_s), np.sin(theta_s), 0.0], dtype=np.float64)


    @staticmethod
    def _fibonacci_sphere(n_points: int) -> np.ndarray:
        """
        Unit sphere points using Fibonacci (golden angle) sampling.
        Returns (N,3) array of unit vectors.
        """
        n_points = int(n_points)
        if n_points <= 0:
            raise ValueError("n_points must be positive")

        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(n_points, dtype=np.float64)

        z = 1.0 - (2.0 * i + 1.0) / n_points
        r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        theta = golden_angle * i

        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)

        return np.stack([x, y, z], axis=1)  # (N,3)

    @classmethod
    def from_spherical_chief(
        cls,
        radius: float,
        n_points: int,
        base_rgb_value=1.0,
        **kwargs,
    ):
        """
        Construct an ObservationModel for a spherical chief (centred at origin).

        Parameters
        ----------
        radius : float
            Chief radius (same units as your dynamics position units).
        n_points : int
            Number of surface sample points.
        base_rgb_value : float or array-like length 3
            Per-tile colour/albedo factor in [0,1]. If scalar, uses grey.
            If length-3, uses constant RGB on all tiles.
        kwargs :
            Forwarded to ObservationModel.__init__ (e.g., fov_half_angle, thresholds, Table A1 params).

        Returns
        -------
        ObservationModel instance
        """
        radius = float(radius)
        n_points = int(n_points)
        if radius <= 0.0:
            raise ValueError("radius must be positive")

        unit_pts = cls._fibonacci_sphere(n_points)         # (N,3)
        surface_normals = unit_pts.copy()                  # unit normals
        surface_points = radius * unit_pts                 # points on sphere

        base_rgb_value = np.asarray(base_rgb_value, dtype=np.float64)
        if base_rgb_value.size == 1:
            base_rgb = np.ones((n_points, 3), dtype=np.float64) * float(base_rgb_value)
        elif base_rgb_value.size == 3:
            base_rgb = np.tile(base_rgb_value.reshape(1, 3), (n_points, 1))
        else:
            raise ValueError("base_rgb_value must be scalar or length-3")

        base_rgb = np.clip(base_rgb, 0.0, 1.0)

        return cls(
            surface_points=surface_points,
            surface_normals=surface_normals,
            base_rgb=base_rgb,
            chief_radius=radius,  
            **kwargs,
        )
        
    # ------------------------------------------------------------------
    # Section B: Perception cone (Eq. 19)
    # ------------------------------------------------------------------
    def is_in_view(self, p_d: np.ndarray) -> np.ndarray:
        """
        Eq. (19): point p_s is in view iff (p_d/||p_d||) · p_s >= r_c * [1 - (||p_d|| - r_c)/||p_d||]
        Algebra simplifies RHS to r_c^2 / ||p_d||.
        """
        p_d = np.asarray(p_d, dtype=np.float64).reshape(3,)
        d = np.linalg.norm(p_d)
        if d < 1e-12:
            raise ValueError("||p_d|| too small")

        u_hat = p_d / d
        rhs = (self.r_c * self.r_c) / d
        lhs = self.surface_points @ u_hat
        return lhs >= rhs

    # ------------------------------------------------------------------
    # Section C: Blinn–Phong illumination (Eq. 20–21)
    # ------------------------------------------------------------------
    def blinn_phong_rgb(self, p_d: np.ndarray, theta_s: float):
        """
        For each surface point p_s:
          V_hat = (p_d - p_s) / ||p_d - p_s||        (point -> sensor)
          L_hat = r^S(theta_s)                        (point -> sun, approx constant for distant sun)
          H_hat = (L_hat + V_hat) / ||L_hat + V_hat|| (Eq. 21)

        Eq. (20) with l_max = 1:
          I_rgb = ka*ia + kd*(L·N)_+*id + ks*(N·H)_+^beta*is
        """
        p_d = np.asarray(p_d, dtype=np.float64).reshape(3,)
        L_hat = self.sun_direction(float(theta_s))
        L_hat = L_hat / max(np.linalg.norm(L_hat), 1e-12)

        p_s = self.surface_points
        N_hat = self.surface_normals

        # V_hat: point -> sensor
        V = p_d[None, :] - p_s
        V_norm = np.linalg.norm(V, axis=1, keepdims=True)
        V_hat = V / np.maximum(V_norm, 1e-12)

        # H_hat: Eq. (21)
        H = L_hat[None, :] + V_hat
        H_norm = np.linalg.norm(H, axis=1, keepdims=True)
        H_hat = H / np.maximum(H_norm, 1e-12)

        # Dot products with clamping (standard Blinn–Phong)
        ndotl = np.maximum(N_hat @ L_hat, 0.0)  # diffuse gate
        ndoth = np.maximum(np.einsum("ij,ij->i", N_hat, H_hat), 0.0)  # specular gate

        # Channel-wise constants
        ambient_rgb  = (self.ka * self.Ia)[None, :]                            # (1,3)
        diffuse_rgb  = (ndotl[:, None]) * (self.kd * self.Id)[None, :]         # (N,3)
        specular_rgb = (ndoth[:, None] ** self.beta) * (self.ks * self.Is)[None, :]  # (N,3)

        I_rgb = ambient_rgb + diffuse_rgb + specular_rgb

        # # Apply chief “grey panel” / albedo factor (your per-tile base_rgb)
        # I_rgb = np.clip(I_rgb * self.base_rgb, 0.0, 1.0)

        # A scalar magnitude is sometimes useful for thresholding
        I_mag = np.linalg.norm(I_rgb, axis=1)
        return I_rgb, I_mag

    def classify_brightness(self, rgb: np.ndarray):
        mag = np.linalg.norm(rgb, axis=1)
        too_bright = mag > self.bright_thresh
        too_dark = mag < self.dark_thresh
        usable = ~(too_bright | too_dark)
        return too_bright, too_dark, usable

    def get_observation(self, state: np.ndarray):
        """
        state = [x,y,z,vx,vy,vz,theta_S]  (theta_S used for illumination)
        """
        s = np.asarray(state, dtype=np.float64).ravel()
        if s.size < 7:
            raise ValueError("state must be [x,y,z,vx,vy,vz,theta_S]")

        p_d = s[:3]
        theta_s = float(s[6])

        # Section B: in-view (Eq. 19)
        in_view = self.is_in_view(p_d)
        
        L_hat = self.sun_direction(theta_s)
        L_hat = L_hat / max(np.linalg.norm(L_hat), 1e-12)

        # --- NEW: shadow / “ray-to-sun unobstructed” gate (paper’s step-1) ---
        illuminated = (self.surface_normals @ L_hat) > 0.0

        # Section C: illumination (Eq. 20–21)
        rgb, _ = self.blinn_phong_rgb(p_d, theta_s)
        
        rgb_eval = rgb.copy()
        rgb_eval[~in_view, :] = 0.0
        rgb_eval[~illuminated, :] = 0.0   # <-- critical

        too_bright, too_dark, usable = self.classify_brightness(rgb_eval)
        inspected = in_view & usable & illuminated
        
        

        return {
            "rgb": rgb_eval,
            "in_view": in_view,
            "too_bright": too_bright,
            "too_dark": too_dark,
            "usable": usable,
            "inspected": inspected,
            "visible_mask": inspected,
        }
