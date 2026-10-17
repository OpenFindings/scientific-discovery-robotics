#!/bin/bash
# restartable: skips jobs already recorded in queue_done.log (evals and training also resume internally)
SP=/tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad
grep -v -x -F -f <(sed 's/^.* done //' $SP/queue_done.log) $SP/queue2.txt > $SP/queue2_todo.txt
xargs -P 4 -L 1 $SP/sim/job.sh < $SP/queue2_todo.txt
echo QUEUEDONE >> $SP/queue_done.log
