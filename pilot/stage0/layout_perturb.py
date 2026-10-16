"""K-layout perturbation (Stage 1): move the task's object of interest on the table.

Pilot condition `layout:<m>`: starting from the LIBERO init state, translate one
object of interest horizontally by exactly m metres in a random direction, then
check the placement is physically valid before the episode starts.

Why not LIBERO-Plus's own layout files: its `levelK_sampleN` BDDL files shift
every object's placement region (measured on the four suites: max shift 30*K mm,
median ~8*K mm, all regions moved, 6% with yaw) and come with regenerated init
states. Those instances cannot be paired with the LIBERO init states the other
conditions use. Here the init state stays the same and only the object of
interest moves, so every arm and seed sees the identical instance, as for
robot_init. Magnitudes 0.03 and 0.06 m equal LIBERO-Plus's level-1 and level-2
maximum shifts (the design's low/high levels).

Which object: the first entry of the BDDL `obj_of_interest` that is a movable
object (in `objects_dict`, not a fixture). That is the grasped object in the
spatial, object and most goal/long tasks, and a destination plate in some
long-horizon tasks; the moved object is logged. Two LIBERO-Goal tasks ("open
the middle drawer of the cabinet", "turn on the stove") have only fixtures of
interest; they are logged as not applicable and skipped.

Validity is judged against the UNPERTURBED settle, because LIBERO init states
leave objects a few cm above their resting height and they fall during the 10
settle steps anyway (measured: ~70 mm for the LIBERO-Spatial bowl). The base
state is settled once; each try is settled the same way and must end with
  * the moved object within 10 mm (xy) of its baseline rest + the offset, within
    10 mm of its baseline rest height (not on top of something, not off an
    edge), and within 8 degrees of its baseline orientation (a first MuJoCo run
    with 15 mm / 15 deg accepted bowls leaning on a plate rim at 10-13 deg);
  * every other movable object within 10 mm (xy) / 10 mm (z) of its baseline rest
    (an overlap would push it away);
  * the task not already solved.
Otherwise the next deterministic direction is tried, up to 8 tries; the k-th try
rotates the direction by k golden angles.
The validated placement is then re-entered through the same path as an
unperturbed episode: a fresh env reset (reset_fn) followed by
env.set_init_state(placed). Restoring the saved MuJoCo state without the reset
left state behind from the validation settles (in a LIBERO run the robot then
started 1-7 mm, up to 13 mm, away from its ID start; MuJoCo's saved state omits
e.g. solver warm-starts). With the fresh reset the difference was 0.0 mm.

numpy only; Python 3.8 (LIBERO client).
"""
import math

import numpy as np

GOLDEN = math.pi * (3 - math.sqrt(5))  # ~137.5 deg: successive tries spread around the circle
MAX_TRIES = 8
SETTLE_STEPS = 10
TOL_XY_M, TOL_Z_M, TOL_TILT_DEG = 0.010, 0.010, 8.0


def movable_target(renv):
    for name in renv.obj_of_interest:
        if name in renv.objects_dict:
            return name
    return None


def _free_joints(renv):
    return {n: o.joints[-1] for n, o in renv.objects_dict.items() if getattr(o, "joints", None)}


def _poses(renv, joints):
    return {n: np.array(renv.sim.data.get_joint_qpos(j), dtype=np.float64) for n, j in joints.items()}


def _tilt_deg(q_before, q_after):
    """Angle between two MuJoCo quaternions (w, x, y, z)."""
    d = abs(float(np.dot(q_before / np.linalg.norm(q_before), q_after / np.linalg.norm(q_after))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def _settle(env, dummy_action, renv, joints):
    for _ in range(SETTLE_STEPS):
        env.step(dummy_action)
    return _poses(renv, joints)


def perturb_layout(env, magnitude_m, rng, dummy_action, reset_fn=None):
    """Returns (obs, record). obs is None when the episode must be skipped.
    reset_fn: a fresh env reset (eval_libero._safe_reset); used to re-enter the validated
    placement exactly like an unperturbed episode. Without it the placement is restored
    from the saved state only (small robot-start offset; tests/fakes only)."""
    renv = env.env
    target = movable_target(renv)
    if target is None:
        return None, dict(kind="layout", applicable=False, valid=False, requested_m=magnitude_m,
                          reason="no movable object of interest")
    joints = _free_joints(renv)
    jn = joints[target]
    base = env.get_sim_state()
    q0 = np.array(renv.sim.data.get_joint_qpos(jn), dtype=np.float64)
    phi0 = float(rng.uniform(0, 2 * math.pi))
    rest = _settle(env, dummy_action, renv, joints)  # unperturbed rest poses
    tries = []
    for k in range(MAX_TRIES):
        phi = phi0 + k * GOLDEN
        env.set_init_state(base)
        q = q0.copy()
        q[:2] += magnitude_m * np.array([math.cos(phi), math.sin(phi)])
        renv.sim.data.set_joint_qpos(jn, q)
        renv.sim.forward()
        placed = env.get_sim_state()
        after = _settle(env, dummy_action, renv, joints)
        offset = q[:2] - q0[:2]
        expect = {n: rest[n][:2] + (offset if n == target else 0.0) for n in joints}
        shift = {n: float(np.linalg.norm(after[n][:2] - expect[n])) for n in joints}
        dz = {n: float(abs(after[n][2] - rest[n][2])) for n in joints}
        tilt = _tilt_deg(rest[target][3:7], after[target][3:7])
        solved = bool(env.check_success())
        worst = max(shift, key=shift.get)
        worst_z = max(dz, key=dz.get)
        ok = (shift[worst] <= TOL_XY_M and dz[worst_z] <= TOL_Z_M and tilt <= TOL_TILT_DEG and not solved)
        tries.append(dict(direction_rad=round(phi % (2 * math.pi), 4), max_xy_dev_mm=round(1000 * shift[worst], 2),
                          most_moved=worst, max_z_dev_mm=round(1000 * dz[worst_z], 2), z_worst=worst_z,
                          tilt_deg=round(tilt, 2), solved=solved, ok=ok))
        if ok:
            if reset_fn is not None:
                reset_fn()
            obs = env.set_init_state(placed)
            return obs, dict(kind="layout", applicable=True, valid=True, target=target, requested_m=magnitude_m,
                             offset_m=(q[:2] - q0[:2]).round(5).tolist(), direction_rad=tries[-1]["direction_rad"],
                             n_tries=k + 1, tries=tries)
    env.set_init_state(base)
    return None, dict(kind="layout", applicable=True, valid=False, target=target, requested_m=magnitude_m,
                      n_tries=MAX_TRIES, tries=tries, reason="no valid placement")
