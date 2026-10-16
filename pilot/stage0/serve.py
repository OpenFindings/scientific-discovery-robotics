"""Serve a pilot checkpoint over openpi's websocket protocol, with prompt logging.

Differences from openpi's scripts/serve_policy.py:
  * pilot configs are registered (openpi_ext);
  * --norm-stats-from loads normalization stats from another checkpoint
    (needed for the pi05_base zero-shot probe, whose assets have no LIBERO
    stats);
  * if an observation carries "pilot/log_prompt": True, the reply carries
    "pilot/prompt": the exact text the model was conditioned on, decoded
    from the tokenized prompt. The eval client asks for it on the first
    query of every episode, so a train/serve flag mismatch shows up in the
    episode log instead of only as lower success.

  uv run python $PILOT/serve.py --config pilot_R --dir checkpoints/pilot_R/s41/9999 --port 8000
  uv run python $PILOT/serve.py --config pilot_base_probe \
      --dir gs://openpi-assets/checkpoints/pi05_base \
      --norm-stats-from gs://openpi-assets/checkpoints/pi05_libero
"""
import dataclasses
import json
import logging
import pathlib
import sys

import jax
import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import openpi_ext  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.shared import download  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi_client import base_policy as _base_policy  # noqa: E402


@dataclasses.dataclass
class Args:
    config: str
    dir: str
    norm_stats_from: str | None = None  # checkpoint dir whose assets/<asset_id> to use
    port: int = 8000


class PromptLoggingPolicy(_base_policy.BasePolicy):
    def __init__(self, policy, tokenizer):
        self._policy = policy
        self._sp = tokenizer._tokenizer  # noqa: SLF001 - sentencepiece processor inside PaligemmaTokenizer

    def infer(self, obs: dict) -> dict:
        log = bool(obs.pop("pilot/log_prompt", False))
        out = self._policy.infer(obs)
        if log:
            inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
            ids = np.asarray(inputs["tokenized_prompt"])[np.asarray(inputs["tokenized_prompt_mask"], bool)]
            out = dict(out, **{"pilot/prompt": self._sp.decode([int(i) for i in ids])})
        return out

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self):
        return self._policy.metadata


def _find_tokenizer(policy):
    for t in policy._input_transform.transforms:  # noqa: SLF001
        if hasattr(t, "tokenizer"):
            return t.tokenizer
    raise RuntimeError("no tokenizer found in the policy's input transforms")


def main(args: Args) -> None:
    openpi_ext.register()
    cfg = _config.get_config(args.config)
    norm_stats = None
    if args.norm_stats_from:
        asset_id = cfg.data.create(cfg.assets_dirs, cfg.model).asset_id
        src = download.maybe_download(str(args.norm_stats_from))
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(src) / "assets", asset_id)
    policy = _policy_config.create_trained_policy(cfg, args.dir, norm_stats=norm_stats)
    wrapped = PromptLoggingPolicy(policy, _find_tokenizer(policy))
    meta = dict(policy.metadata or {}, pilot=dict(openpi_ext.describe(args.config), checkpoint=args.dir,
                                                  norm_stats_from=args.norm_stats_from))
    logging.info("serving %s", json.dumps(meta["pilot"]))
    websocket_policy_server.WebsocketPolicyServer(policy=wrapped, host="0.0.0.0", port=args.port,
                                                  metadata=meta).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
