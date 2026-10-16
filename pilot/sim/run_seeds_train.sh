cd /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/sim
for seed in 1 2; do
  SEED=$seed OMP_NUM_THREADS=1 ../venv323/bin/python blind_train.py /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/libero_train_state_actions.npz /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/blind > /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/blind/train_s$seed.log 2>&1 &
  SEED=$seed OMP_NUM_THREADS=1 ../venv323/bin/python ego_train.py "/tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/oracle/libero_*_t*.npz" /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/ego > /tmp/claude-0/-home-user/d4709d28-02d5-592b-a655-aadd22705cd1/scratchpad/ego/train_s$seed.log 2>&1 &
done
wait
