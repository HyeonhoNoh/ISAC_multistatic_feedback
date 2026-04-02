"""
ISAC Channel Generation
  H : (K, N)    BS → User          Rayleigh + 3GPP TR 38.901
  G : (Q, N, N) BS → Target → BS   LOS, round-trip FSPL + ULA  (β_rt = FSPL²)
  F : (Q, K)    Target → User       LOS, one-way FSPL scalar
"""

import numpy as np

C      = 3e8       # speed of light (m/s)
K_B    = 1.38e-23  # Boltzmann constant (J/K)
T_REF  = 290.0     # reference temperature (K)


# ──────────────────────────────────────────────────────────────
# RF utilities
# ──────────────────────────────────────────────────────────────

def dBm_to_W(dBm: float) -> float:
    return 10 ** ((dBm - 30) / 10)

def W_to_dBm(w: float) -> float:
    return 10 * np.log10(w) + 30

def noise_power_W(bandwidth: float, noise_figure_dB: float) -> float:
    """Thermal noise power: N = k*T*B * NF  (Watts)."""
    nf_linear = 10 ** (noise_figure_dB / 10)
    return K_B * T_REF * bandwidth * nf_linear


# ──────────────────────────────────────────────────────────────
# 3GPP TR 38.901 path loss  (used for H only)
# ──────────────────────────────────────────────────────────────

def _breakpoint_distance(h_bs: float, h_ut: float, carrier_freq: float) -> float:
    """Effective environment height: 1 m (TR 38.901 Sec 7.4.1)."""
    h_e = 1.0
    return 4 * (h_bs - h_e) * (h_ut - h_e) * carrier_freq / C


def _d3d(d2d: float, h_bs: float, h_ut: float) -> float:
    return np.sqrt(d2d**2 + (h_bs - h_ut)**2)


def _los_probability(d2d: float, scenario: str) -> float:
    """LOS probability (TR 38.901 Table 7.4.2-1)."""
    if d2d <= 0:
        return 1.0
    if scenario == "UMi":
        return min(18 / d2d, 1.0) * (1 - np.exp(-d2d / 36)) + np.exp(-d2d / 36)
    else:  # UMa
        return min(18 / d2d, 1.0) * (1 - np.exp(-d2d / 63)) + np.exp(-d2d / 63)


def _pl_umi(d2d: float, h_bs: float, h_ut: float, fc_GHz: float, is_los: bool) -> tuple[float, float]:
    """UMi-Street Canyon path loss + shadowing std (TR 38.901 Table 7.4.1-1)."""
    d3d  = max(_d3d(d2d, h_bs, h_ut), 0.5)
    d_bp = _breakpoint_distance(h_bs, h_ut, fc_GHz * 1e9)

    if is_los:
        if d2d <= d_bp:
            pl = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc_GHz)
        else:
            pl = (32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc_GHz)
                  - 9.5 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
        sigma = 4.0
    else:
        if d2d <= d_bp:
            pl_los = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc_GHz)
        else:
            pl_los = (32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc_GHz)
                      - 9.5 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
        pl_nlos = (35.3 * np.log10(d3d) + 22.4
                   + 21.3 * np.log10(fc_GHz) - 0.3 * (h_ut - 1.5))
        pl    = max(pl_los, pl_nlos)
        sigma = 7.82
    return pl, sigma


def _pl_uma(d2d: float, h_bs: float, h_ut: float, fc_GHz: float, is_los: bool) -> tuple[float, float]:
    """UMa path loss + shadowing std (TR 38.901 Table 7.4.1-1)."""
    d3d  = max(_d3d(d2d, h_bs, h_ut), 0.5)
    d_bp = _breakpoint_distance(h_bs, h_ut, fc_GHz * 1e9)

    if is_los:
        if d2d <= d_bp:
            pl = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc_GHz)
        else:
            pl = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc_GHz)
                  - 9 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
        sigma = 4.0
    else:
        if d2d <= d_bp:
            pl_los = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc_GHz)
        else:
            pl_los = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc_GHz)
                      - 9 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
        pl_nlos = (13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc_GHz)
                   - 0.6 * (h_ut - 1.5))
        pl    = max(pl_los, pl_nlos)
        sigma = 6.0
    return pl, sigma


def path_loss_3gpp(
    d2d: float,
    carrier_freq: float,
    scenario: str,
    h_bs: float,
    h_ut: float,
    los_condition: str,
    shadowing: bool,
    rng: np.random.Generator,
) -> tuple[float, float, bool, float]:
    """
    3GPP TR 38.901 path loss.

    Returns:
        pl_total_dB : path loss including shadowing (dB)
        beta        : linear large-scale coefficient = 10^(-pl/10)
        is_los      : LOS/NLOS flag
        shadow_dB   : shadowing realization (dB)
    """
    fc_GHz = carrier_freq / 1e9
    d2d    = max(d2d, 1.0)

    if los_condition == "LOS":
        is_los = True
    elif los_condition == "NLOS":
        is_los = False
    else:
        p_los  = _los_probability(d2d, scenario)
        is_los = (rng.uniform() < p_los)

    if scenario == "UMi":
        pl, sigma = _pl_umi(d2d, h_bs, h_ut, fc_GHz, is_los)
    else:
        pl, sigma = _pl_uma(d2d, h_bs, h_ut, fc_GHz, is_los)

    shadow_dB = float(rng.normal(0, sigma)) if shadowing else 0.0
    pl_total  = pl + shadow_dB
    beta      = 10 ** (-pl_total / 10)
    return pl_total, beta, is_los, shadow_dB


# ──────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────

def xy_to_angle(x: float, y: float) -> float:
    """AoD from BS broadside (+y): arctan2(x, y) in degrees."""
    return np.degrees(np.arctan2(x, y))

def xy_to_distance(x: float, y: float) -> float:
    return float(np.sqrt(x**2 + y**2))

def steering_vector(N: int, angle_deg: float, d_over_lambda: float = 0.5) -> np.ndarray:
    """Normalised ULA steering vector, shape (N,)."""
    phi = np.deg2rad(angle_deg)
    return np.exp(1j * 2 * np.pi * d_over_lambda * np.sin(phi) * np.arange(N)) / np.sqrt(N)

def _cn(shape, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. CN(0,1) samples."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


# ──────────────────────────────────────────────────────────────
# Channel: H (K, N)  — BS → User, Rician (LOS) / Rayleigh (NLOS) + 3GPP PL
# ──────────────────────────────────────────────────────────────

def _pl_kwargs(cfg: dict) -> dict:
    ch = cfg["channel"]
    return dict(carrier_freq=cfg["system"]["carrier_freq"],
                scenario=ch["scenario"],
                h_bs=ch["h_bs"], h_ut=ch["h_ut"],
                los_condition=ch["los_condition"],
                shadowing=ch["shadowing"])


def _comm_channel_one(x, y, N, pl_kw, K_rice, rng):
    """
    Rician (LOS) or Rayleigh (NLOS) channel with 3GPP path loss.

    LOS:  h = sqrt(beta) * (sqrt(K/(K+1)) * a(theta) + sqrt(1/(K+1)) * CN(0,I))
    NLOS: h = sqrt(beta) * CN(0,I)

    K_rice : Rician K-factor (linear).  K=0 → pure Rayleigh.
    """
    d = xy_to_distance(x, y)
    pl, beta, is_los, shad = path_loss_3gpp(d, **pl_kw, rng=rng)

    if is_los and K_rice > 0:
        theta = xy_to_angle(x, y)
        a = steering_vector(N, theta)
        h_los  = np.sqrt(K_rice / (K_rice + 1)) * a
        h_nlos = np.sqrt(1.0 / (K_rice + 1)) * _cn((N,), rng)
        h = np.sqrt(beta) * (h_los + h_nlos)
    else:
        h = np.sqrt(beta) * _cn((N,), rng)

    return h, pl, beta, is_los, shad


def _get_rician_K(cfg: dict) -> float:
    """Rician K-factor (linear) from config. Default 10 dB for UMi LOS."""
    K_dB = cfg["channel"].get("rician_K_dB", 10.0)
    return 10.0 ** (K_dB / 10.0)


def generate_comm_channel_random(cfg: dict, N: int, rng: np.random.Generator):
    """
    Generate H for K randomly placed users.

    Returns:
        H        : (K, N) complex channel matrix
        pos      : (K, 2) user positions [m]
        angles   : (K,)   AoD from BS [deg]
        pls      : (K,)   path loss [dB]
        betas    : (K,)   large-scale coefficients
        los_flags: (K,)   LOS/NLOS flags
        shads    : (K,)   shadowing realizations [dB]
    """
    rc  = cfg["random"]
    plk = _pl_kwargs(cfg)
    K   = rc["num_users"]
    K_rice = _get_rician_K(cfg)

    x = rng.uniform(*rc["user_area"]["x_range"], size=K)
    y = rng.uniform(*rc["user_area"]["y_range"], size=K)
    pos = np.stack([x, y], axis=1)

    H, pls, betas, los_flags, shads = [], [], [], [], []
    for k in range(K):
        h, pl, beta, is_los, shad = _comm_channel_one(
            x[k], y[k], N, plk, K_rice, rng)
        H.append(h); pls.append(pl); betas.append(beta)
        los_flags.append(is_los); shads.append(shad)

    return (np.stack(H), pos,
            np.degrees(np.arctan2(x, y)),
            np.array(pls), np.array(betas),
            los_flags, np.array(shads))


def generate_comm_channel_manual(cfg: dict, N: int, rng: np.random.Generator):
    """
    Generate H for manually specified users.

    Returns: same structure as generate_comm_channel_random.
    """
    plk = _pl_kwargs(cfg)
    manual = cfg["manual"]
    K_rice = _get_rician_K(cfg)

    # Build user positions: either explicit list or random drop around targets
    if "users" in manual:
        uc = [{"x": u["x"], "y": u["y"], "shadowing_dB": u.get("shadowing_dB", 0.0)}
              for u in manual["users"]]
    else:
        targets = manual["targets"]
        n_per_tgt = int(manual.get("users_per_target", 2))
        radius = float(manual.get("user_drop_radius", 10.0))
        uc = []
        for tgt in targets:
            tx, ty = tgt["x"], tgt["y"]
            for _ in range(n_per_tgt):
                ang = rng.uniform(0, 2 * np.pi)
                r = radius * np.sqrt(rng.uniform())
                uc.append({"x": tx + r * np.cos(ang),
                           "y": ty + r * np.sin(ang),
                           "shadowing_dB": 0.0})

    H, pos, angles, pls, betas, los_flags, shads = [], [], [], [], [], [], []
    for ucfg in uc:
        x, y         = ucfg["x"], ucfg["y"]
        extra_shadow = ucfg.get("shadowing_dB", 0.0)
        d            = xy_to_distance(x, y)
        pl, beta, is_los, shad = path_loss_3gpp(d, **plk, rng=rng)
        pl   += extra_shadow
        beta  = 10 ** (-pl / 10)

        if is_los and K_rice > 0:
            theta = xy_to_angle(x, y)
            a = steering_vector(N, theta)
            h_los  = np.sqrt(K_rice / (K_rice + 1)) * a
            h_nlos = np.sqrt(1.0 / (K_rice + 1)) * _cn((N,), rng)
            h = np.sqrt(beta) * (h_los + h_nlos)
        else:
            h = np.sqrt(beta) * _cn((N,), rng)

        H.append(h); pos.append([x, y]); angles.append(xy_to_angle(x, y))
        pls.append(pl); betas.append(beta)
        los_flags.append(is_los); shads.append(shad + extra_shadow)

    if not H:   # users_per_target == 0  →  sensing-only mode
        return (np.empty((0, N), dtype=complex), np.empty((0, 2)), np.array([]),
                np.array([]), np.array([]), [], np.array([]))
    return (np.stack(H), np.array(pos), np.array(angles),
            np.array(pls), np.array(betas), los_flags, np.array(shads))


# ──────────────────────────────────────────────────────────────
# Channel: G (Q, N, N)  — BS → Target → BS  (two-way LOS, FSPL)
#
# Round-trip FSPL + ULA steering vector (no RCS):
#   β_rt = (λ / (4π·d))⁴   = FSPL²
#   α_t  = √β_rt · exp(j·4π·d/λ)    (round-trip deterministic phase)
#   G[t] = α_t · a(θ_t) · a(θ_t)^H   shape (N, N)
# ──────────────────────────────────────────────────────────────

def generate_sensing_channel_random(cfg: dict, N: int, rng: np.random.Generator):
    """
    Generate G for Q randomly placed targets.

    Returns:
        G       : (Q, N, N) sensing channel matrices
        pos     : (Q, 2)    target positions [m]
        angles  : (Q,)      AoD from BS [deg]
        gains   : (Q,)      complex α_q
        pls_rt  : (Q,)      round-trip path loss [dB]
        betas_rt: (Q,)      round-trip large-scale coefficients
    """
    rc           = cfg["random"]
    carrier_freq = cfg["system"]["carrier_freq"]
    lam          = C / carrier_freq
    Q            = rc["num_targets"]

    x = rng.uniform(*rc["target_area"]["x_range"], size=Q)
    y = rng.uniform(*rc["target_area"]["y_range"], size=Q)
    pos = np.stack([x, y], axis=1)

    G, pls_rt, betas_rt, gains = [], [], [], []
    for q in range(Q):
        d       = max(xy_to_distance(x[q], y[q]), 1.0)
        beta_rt = (lam / (4 * np.pi * d)) ** 4
        alpha   = np.sqrt(beta_rt) * np.exp(1j * 4 * np.pi * d / lam)
        theta   = np.degrees(np.arctan2(x[q], y[q]))
        a       = steering_vector(N, theta)
        G.append(alpha * np.outer(a, a.conj()))
        pls_rt.append(-10 * np.log10(beta_rt))
        betas_rt.append(beta_rt)
        gains.append(alpha)

    angles = np.degrees(np.arctan2(x, y))
    return (np.stack(G), pos, angles,
            np.array(gains, dtype=complex), np.array(pls_rt), np.array(betas_rt))


def generate_sensing_channel_manual(cfg: dict, N: int, rng: np.random.Generator):
    """
    Generate G for manually specified targets.

    Returns: same structure as generate_sensing_channel_random.
    """
    qc           = cfg["manual"]["targets"]
    carrier_freq = cfg["system"]["carrier_freq"]
    lam          = C / carrier_freq

    G, pos, angles, pls_rt, betas_rt, gains = [], [], [], [], [], []
    for qcfg in qc:
        x, y    = qcfg["x"], qcfg["y"]
        d       = max(xy_to_distance(x, y), 1.0)
        beta_rt = (lam / (4 * np.pi * d)) ** 4
        alpha   = np.sqrt(beta_rt) * np.exp(1j * 4 * np.pi * d / lam)
        theta   = xy_to_angle(x, y)
        a       = steering_vector(N, theta)
        G.append(alpha * np.outer(a, a.conj()))
        pos.append([x, y]); angles.append(theta)
        pls_rt.append(-10 * np.log10(beta_rt))
        betas_rt.append(beta_rt)
        gains.append(alpha)

    return (np.stack(G), np.array(pos), np.array(angles),
            np.array(gains, dtype=complex), np.array(pls_rt), np.array(betas_rt))


# ──────────────────────────────────────────────────────────────
# Channel: F (Q, K)  — BS → Target → User  (bistatic LOS, FSPL)
#
# Bistatic path = BS→Target (one-way) × Target→User (one-way):
#   β_BQ  = (λ / (4π·d_BQ))²      BS → Target
#   β_QU  = (λ / (4π·d_QK))²      Target → User
#   β_qk  = β_BQ · β_QU           combined bistatic gain
#   f_qk  = √β_qk · exp(j·2π·(d_BQ + d_QK)/λ)
# ──────────────────────────────────────────────────────────────

def generate_target_user_channel(
    tgt_pos: np.ndarray,
    user_pos: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate F (Q, K) BS→Target→User bistatic LOS channel.

    The path loss combines BS→Target and Target→User FSPL:
        β_qk = (λ/4πd_BQ)² · (λ/4πd_QK)²

    Returns:
        F    : (Q, K) complex scalars
        phi  : (Q, K) AoA at user from target [deg]
        betas: (Q, K) bistatic FSPL gains (linear)
        pls  : (Q, K) bistatic path loss [dB]
    """
    carrier_freq = cfg["system"]["carrier_freq"]
    lam          = C / carrier_freq
    Q, K         = len(tgt_pos), len(user_pos)
    F     = np.zeros((Q, K), dtype=complex)
    phi   = np.zeros((Q, K))
    betas = np.zeros((Q, K))
    pls   = np.zeros((Q, K))

    for q in range(Q):
        # BS→Target distance (BS at origin)
        d_BQ    = max(xy_to_distance(tgt_pos[q, 0], tgt_pos[q, 1]), 1.0)
        beta_BQ = (lam / (4 * np.pi * d_BQ)) ** 2

        for k in range(K):
            # Target→User distance
            dx      = tgt_pos[q, 0] - user_pos[k, 0]
            dy      = tgt_pos[q, 1] - user_pos[k, 1]
            d_QK    = max(float(np.sqrt(dx**2 + dy**2)), 1.0)
            beta_QK = (lam / (4 * np.pi * d_QK)) ** 2

            # Combined bistatic gain
            beta = beta_BQ * beta_QK
            phi[q, k]   = np.degrees(np.arctan2(dx, dy))
            betas[q, k] = beta
            pls[q, k]   = -10 * np.log10(beta)
            F[q, k]     = np.sqrt(beta) * np.exp(1j * 2 * np.pi * (d_BQ + d_QK) / lam)

    return F, phi, betas, pls
