"""Arm C state encoding: LIBERO state relative to the episode's first state.

LIBERO state (openpi / OpenVLA layout, 8-D):
  [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]   robot base frame

Episode-relative state (pilot design, section 2.1):
  pos_t - pos_0                    translation since the start, base frame
  axis_angle(R_0^T R_t)            rotation since the start, in the start EEF frame
  gripper_qpos_t                   unchanged (absolute)

The same function is used to convert the training data (convert_relative_state.py)
and online by the eval client (eval_libero.py --state-mode relative), so training
and serving cannot drift apart. numpy only; runs on Python 3.8 (LIBERO client)
and 3.11 (openpi).
"""
import numpy as np

_EPS = 1e-8


def axis_angle_to_matrix(v):
    """Rodrigues. v: (..., 3) axis * angle (any angle, including > pi)."""
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(theta > _EPS, v / np.maximum(theta, _EPS), 0.0)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([np.stack([zero, -kz, ky], -1), np.stack([kz, zero, -kx], -1),
                  np.stack([-ky, kx, zero], -1)], -2)
    th = theta[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def matrix_to_axis_angle(R):
    """Inverse of axis_angle_to_matrix, angle in [0, pi]. R: (..., 3, 3)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(tr)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(theta)
    out = np.where((s > 1e-6)[..., None], w / (2 * np.maximum(s, 1e-6))[..., None] * theta[..., None], 0.0)
    # near pi: axis from the symmetric part (sin -> 0 makes w unreliable)
    near_pi = theta > np.pi - 1e-3
    if np.any(near_pi):
        Rn, c = R[near_pi], np.cos(theta[near_pi])[:, None, None]
        B = ((Rn + np.swapaxes(Rn, -1, -2)) / 2 - c * np.eye(3)) / (1 - c)  # = k k^T exactly
        i = np.argmax(np.diagonal(B, axis1=-2, axis2=-1), axis=-1)
        rows = B[np.arange(len(i)), i]
        axis = rows / np.linalg.norm(rows, axis=-1, keepdims=True)
        # the symmetric part fixes the axis only up to sign; the (small) skew part breaks the tie
        sign = np.where(np.einsum("ij,ij->i", axis, w[near_pi]) < 0, -1.0, 1.0)[:, None]
        out[near_pi] = sign * axis * theta[near_pi][..., None]
    return out


def to_relative(states, anchor):
    """states: (T, 8) or (8,); anchor: (8,) first state of the episode. Returns same shape, float32."""
    states = np.asarray(states, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    single = states.ndim == 1
    s = states[None] if single else states
    R0 = axis_angle_to_matrix(anchor[3:6])
    Rt = axis_angle_to_matrix(s[:, 3:6])
    rel_rot = matrix_to_axis_angle(np.einsum("ji,tjk->tik", R0, Rt))  # R0^T Rt
    out = np.concatenate([s[:, :3] - anchor[:3], rel_rot, s[:, 6:8]], axis=1).astype(np.float32)
    return out[0] if single else out


def from_relative(rel, anchor):
    """Inverse of to_relative (used only to verify the conversion)."""
    rel = np.asarray(rel, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    R0 = axis_angle_to_matrix(anchor[3:6])
    Rt = np.einsum("ij,tjk->tik", R0, axis_angle_to_matrix(rel[:, 3:6]))
    return np.concatenate([rel[:, :3] + anchor[:3], matrix_to_axis_angle(Rt), rel[:, 6:8]], axis=1)


class EpisodeRelativeState:
    """Online version for the eval client: the first state seen after reset() is the anchor."""

    def __init__(self):
        self.anchor = None

    def reset(self):
        self.anchor = None

    def __call__(self, state):
        if self.anchor is None:
            self.anchor = np.asarray(state, dtype=np.float64).copy()
        return to_relative(state, self.anchor)
