"""Build the arm C training set: physical-intelligence/libero with episode-relative state.

Copies the LeRobot v2.0 dataset and rewrites only the `state` column of each
episode parquet with relative_state.to_relative(state, state[frame 0]). Images,
actions, indices, timestamps, task ids and the HF schema metadata are copied
unchanged (image bytes are not re-encoded). meta/stats.json gets new `state`
stats; everything else in meta/ is copied as is.

Every episode is checked before it is written:
  * frame 0 is exactly zero in position and rotation; gripper columns untouched;
  * reconstructing the absolute state from (relative, anchor) recovers the
    original position and rotation matrix (axis-angle is compared as a rotation,
    since LIBERO angles straddle pi and the representation is not unique).
A manifest (pilot_relstate_manifest.json) records the anchors and checks.

Run in openpi's environment (pyarrow comes with `datasets`):
  uv run python $PILOT/convert_relative_state.py                       # full dataset
  uv run python $PILOT/convert_relative_state.py --max-episodes 5 --dst /tmp/relstate_test

Source defaults to $HF_LEROBOT_HOME/physical-intelligence/libero (openpi downloads it
on first training run); the output repo id `local/libero_relstate` is what
openpi_ext.pilot_C reads. Needs about as much disk as the source (~30 GB).
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from relative_state import axis_angle_to_matrix, from_relative, to_relative  # noqa: E402

SRC_REPO = "physical-intelligence/libero"
DST_REPO = "local/libero_relstate"


def lerobot_home() -> pathlib.Path:
    if "HF_LEROBOT_HOME" in os.environ:
        return pathlib.Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    hf = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf / "lerobot"


def convert_episode(src: pathlib.Path, dst: pathlib.Path) -> dict:
    pf = pq.ParquetFile(src)
    table = pf.read()
    state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
    frame = np.asarray(table.column("frame_index").to_pylist())
    if frame[0] != 0 or np.any(np.diff(frame) != 1):
        raise ValueError(f"{src.name}: frames not contiguous from 0")
    anchor = state[0].astype(np.float64)
    rel = to_relative(state, anchor)

    # checks
    if np.abs(rel[0, :6]).max() > 1e-6:
        raise ValueError(f"{src.name}: frame 0 not zero after conversion")
    if not np.array_equal(rel[:, 6:8], state[:, 6:8]):
        raise ValueError(f"{src.name}: gripper columns changed")
    back = from_relative(rel, anchor)
    pos_err = np.abs(back[:, :3] - state[:, :3]).max()
    rot_err = np.abs(axis_angle_to_matrix(back[:, 3:6]) - axis_angle_to_matrix(state[:, 3:6])).max()
    if pos_err > 1e-5 or rot_err > 1e-4:
        raise ValueError(f"{src.name}: round trip error pos {pos_err:.2e} rot {rot_err:.2e}")

    field = table.schema.field("state")
    new_col = pa.FixedSizeListArray.from_arrays(pa.array(rel.reshape(-1), type=pa.float32()), 8)
    out = table.set_column(table.schema.get_field_index("state"), field, new_col.cast(field.type))
    out = out.replace_schema_metadata(table.schema.metadata)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst, compression=pf.metadata.row_group(0).column(0).compression.lower(),
                   row_group_size=pf.metadata.row_group(0).num_rows)
    return dict(file=str(src.relative_to(src.parents[2])), n=len(state), anchor=anchor.round(6).tolist(),
                pos_err=float(pos_err), rot_err=float(rot_err),
                rel_min=rel.min(0).round(5).tolist(), rel_max=rel.max(0).round(5).tolist()), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help=f"default: $HF_LEROBOT_HOME/{SRC_REPO}")
    ap.add_argument("--dst", default=None, help=f"default: $HF_LEROBOT_HOME/{DST_REPO}")
    ap.add_argument("--max-episodes", type=int, default=None, help="convert only the first N (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    src = pathlib.Path(a.src) if a.src else lerobot_home() / SRC_REPO
    dst = pathlib.Path(a.dst) if a.dst else lerobot_home() / DST_REPO
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.0":
        sys.exit(f"expected LeRobot v2.0, got {info.get('codebase_version')}")
    if dst.exists():
        if not a.overwrite:
            sys.exit(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)

    files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if a.max_episodes:
        files = files[: a.max_episodes]
    if not a.max_episodes and len(files) != info["total_episodes"]:
        sys.exit(f"found {len(files)} episode files, info.json says {info['total_episodes']}")

    shutil.copytree(src / "meta", dst / "meta")
    manifest, rels = [], []
    for i, f in enumerate(files):
        rec, rel = convert_episode(f, dst / f.relative_to(src))
        manifest.append(rec)
        rels.append(rel)
        if i % 100 == 0:
            print(f"{i + 1}/{len(files)} {rec['file']}", flush=True)

    allrel = np.concatenate(rels).astype(np.float64)
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    stats["state"] = dict(mean=allrel.mean(0).tolist(), std=allrel.std(0).tolist(),
                          min=allrel.min(0).tolist(), max=allrel.max(0).tolist())
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    if a.max_episodes:  # keep the subset loadable as a dataset
        info.update(total_episodes=len(files), total_frames=int(sum(m["n"] for m in manifest)),
                    splits={"train": f"0:{len(files)}"})
        (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
        keep = {int(f.stem.split("_")[1]) for f in files}
        eps = [l for l in (dst / "meta" / "episodes.jsonl").read_text().splitlines()
               if l.strip() and json.loads(l)["episode_index"] in keep]
        (dst / "meta" / "episodes.jsonl").write_text("\n".join(eps) + "\n")

    summary = dict(
        source=str(src), dest=str(dst), repo_id=DST_REPO, episodes=len(files), frames=int(len(allrel)),
        encoding="pos - pos0, axis_angle(R0^T R), gripper absolute; anchor = frame 0",
        max_pos_err=max(m["pos_err"] for m in manifest), max_rot_err=max(m["rot_err"] for m in manifest),
        state_q01=np.quantile(allrel, 0.01, axis=0).round(5).tolist(),
        state_q99=np.quantile(allrel, 0.99, axis=0).round(5).tolist(),
        source_info_sha256=hashlib.sha256((src / "meta" / "info.json").read_bytes()).hexdigest(),
        episodes_detail=manifest)
    (dst / "pilot_relstate_manifest.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}, indent=1))


if __name__ == "__main__":
    main()
