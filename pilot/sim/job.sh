#!/bin/bash
# one queued job: kind args...
SP=/tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad
cd $SP/sim
export PYTHONPATH=$SP/libero_src:$SP/sim OMP_NUM_THREADS=1
kind=$1; shift
waitfor() { for f in "$@"; do while [ ! -f $f ]; do sleep 60; done; done; }
case $kind in
  pixtrain) arm=$1; seed=$2
    [ -f $SP/pixm/pix_${arm}$( [ $seed = 0 ] || echo _s$seed ).pt ] && exit 0
    SEED=$seed THREADS=1 ../venv323/bin/python pix_train.py $SP/oracle $SP/pix $SP/pixm $arm 20000 >> $SP/pixm/train_${arm}_s$seed.log 2>&1 ;;
  pixeval) suite=$1; seed=$2; tasks=$3; suf=$( [ $seed = 0 ] || echo _s$seed )
    waitfor $SP/pixm/pix_N$suf.pt $SP/pixm/pix_R$suf.pt $SP/pixm/pix_C$suf.pt
    SEED=$seed ../venv323/bin/python pix_eval.py $suite $SP/pixm $SP/pixm/eval/${suite}_s${seed}_$tasks.jsonl id,robot_init:0.1,layout:0.06,scene_blackout 0-4 N,R,C $tasks >> $SP/pixm/eval/${suite}_s${seed}_$tasks.log 2>&1 ;;
  pixevalD) suite=$1; seed=$2; tasks=$3; suf=$( [ $seed = 0 ] || echo _s$seed )
    waitfor $SP/pixm/pix_D$suf.pt
    SEED=$seed ../venv323/bin/python pix_eval.py $suite $SP/pixm $SP/pixm/eval/${suite}_s${seed}_${tasks}_D.jsonl id,robot_init:0.1,layout:0.06,scene_blackout 0-4 D $tasks >> $SP/pixm/eval/${suite}_s${seed}_${tasks}_D.log 2>&1 ;;
  blindeval) suite=$1; seed=$2; conds=$3; out=$4
    SEED=$seed ../venv323/bin/python blind_eval.py $suite $SP/blind $SP/blind/eval_seeds/$out.jsonl $conds 0-4 T,A,Rl > $SP/blind/eval_seeds/$out.log 2>&1 ;;
  egoeval) suite=$1; seed=$2
    SEED=$seed ../venv323/bin/python ego_eval.py $suite $SP/ego $SP/ego/eval_seeds/${suite}_s$seed.jsonl id,robot_init:0.1,layout:0.06 0-4 E,EA,ERl > $SP/ego/eval_seeds/${suite}_s$seed.log 2>&1 ;;
esac
echo "$(date +%T) done $kind $*" >> $SP/queue_done.log
