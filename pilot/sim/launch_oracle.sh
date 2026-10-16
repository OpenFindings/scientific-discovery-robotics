cd /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/sim
for s in libero_spatial libero_object; do for half in 0-2 3-4; do
  PYTHONPATH=/tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/libero_src:/tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/sim OMP_NUM_THREADS=1 nohup ../venv323/bin/python oracle_eval.py $s /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/oracle /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/oracle/eval/${s}_${half}.jsonl id,robot_init:0.1,robot_init:0.2,layout:0.03,layout:0.06,blackout $half O,OA,ORl > /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/oracle/eval/${s}_${half}.log 2>&1 &
done; done
