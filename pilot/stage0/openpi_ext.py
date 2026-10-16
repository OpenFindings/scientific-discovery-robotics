"""Pilot train configs, registered into openpi's config registry at import.

Checked against openpi @ 215abfb (2026-08-24). Every config is the stock
`pi05_libero` config with the Blackout-vs-Freeze simulation recipe
(2609.39145, Table 3) and one switch changed:

  pilot_N   discrete_state_input=False  (no state; stock openpi LIBERO behaviour)
  pilot_R   discrete_state_input=True   (raw absolute state as prompt tokens)
  pilot_base_probe  serving-only: pi05_base weights with LIBERO norm stats,
                    state on (pi05_base was pretrained with state tokens)

openpi reads `discrete_state_input` in the transform factory, not from the
checkpoint, so a checkpoint must be served with the config it was trained
with. serve.py logs the decoded prompt to make a mismatch visible.

Arm C (episode-relative state) is a Stage 1 config and is not defined here.
"""
import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer

BASE = "pi05_libero"

# Blackout vs Freeze Table 3 (pi0.5, simulation): 10k updates, batch 32,
# linear warmup over 10k updates to 5e-5 (so the evaluated update is the end
# of warmup), AdamW (0.9, 0.95), wd 1e-10, clip 1.0, EMA 0.999.
# CosineDecaySchedule with decay_lr == peak_lr is warmup-then-constant, the
# same schedule object stock pi05_libero uses.
RECIPE = dict(
    batch_size=32,
    num_train_steps=10_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
    optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0),
    ema_decay=0.999,
    save_interval=5_000,
    keep_period=5_000,
)


def _arm(name: str, discrete_state_input: bool) -> _config.TrainConfig:
    base = _config.get_config(BASE)
    model = dataclasses.replace(base.model, discrete_state_input=discrete_state_input)
    return dataclasses.replace(base, name=name, model=model, **RECIPE)


def register() -> list[str]:
    """Add the pilot configs to openpi's registry (idempotent)."""
    configs = [
        _arm("pilot_N", discrete_state_input=False),
        _arm("pilot_R", discrete_state_input=True),
        _arm("pilot_base_probe", discrete_state_input=True),
    ]
    for c in configs:
        if c.name not in _config._CONFIGS_DICT:  # noqa: SLF001 - openpi has no public registration hook
            _config._CONFIGS.append(c)  # noqa: SLF001
            _config._CONFIGS_DICT[c.name] = c  # noqa: SLF001
    return [c.name for c in configs]


def describe(name: str) -> dict:
    """The settings that must match the pilot design, for the run log."""
    c = _config.get_config(name)
    return dict(name=c.name, discrete_state_input=c.model.discrete_state_input, pi05=c.model.pi05,
                action_horizon=c.model.action_horizon, batch_size=c.batch_size,
                num_train_steps=c.num_train_steps, warmup=c.lr_schedule.warmup_steps,
                peak_lr=c.lr_schedule.peak_lr, ema_decay=c.ema_decay,
                weight_loader=repr(c.weight_loader), repo_id=getattr(c.data, "repo_id", None))
