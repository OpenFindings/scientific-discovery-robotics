"""Run the pilot's check_anchors comparison with training anchors from the fetched npz."""
import json, sys, glob
import numpy as np
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import check_anchors as ca
from relative_state import to_relative
d = np.load(sys.argv[1]); lens = d["lengths"]; S = d["state"].astype(np.float64)
starts = np.r_[0, np.cumsum(lens)[:-1]]
train = [dict(episode=int(e), task=ca.norm_task(str(t)), anchor=S[s], rel=to_relative(S[s:s + n], S[s]))
         for e, t, s, n in zip(d["episode"], d["tasks"], starts, lens)]
starts_eval = ca.load_eval(sorted(glob.glob(sys.argv[2])))
rep = ca.summarize(*ca.compare(train, starts_eval), n_train=len(train))
rep.pop("relative_state_quantiles", None)
print(json.dumps(rep, indent=1)[:4000])
