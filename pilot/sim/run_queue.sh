#!/bin/bash
SP=/tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad
# wait for image extraction and MLP seed training
while ! grep -q ALLDONE $SP/pix/extract.log; do sleep 60; done
while pgrep -f run_seeds_train.sh > /dev/null; do sleep 60; done
cd $SP/sim && PYTHONPATH=$SP/libero_src:$SP/sim THREADS=4 ../venv323/bin/python -c "
import sys; sys.argv=['x','$SP/oracle','$SP/pix','$SP/pixm','N']
import pix_train; pix_train.oracle_dir, pix_train.pix_dir = sys.argv[1], sys.argv[2]; print(pix_train.pack())" > $SP/pixm/pack.log 2>&1
xargs -P 4 -L 1 $SP/sim/job.sh < $SP/queue.txt
echo QUEUEDONE >> $SP/queue_done.log
