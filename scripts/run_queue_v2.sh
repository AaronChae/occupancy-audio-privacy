#!/usr/bin/env bash
# Second queue: smaller embeddings (d = 8) and "posteriors only" representations for the
# privacy-utility trade-off. Starts after stage D of run_queue_v0.sh; attacks wait for that queue to end.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { local name=$1; shift; "$PY" "$@" > "logs/$name.log" 2>&1; }

echo "queue v2 start $(date)"
until grep -q "D done" logs/queue_v0.log; do sleep 60; done
run office_v2_learned_floor_d8_adv0 src/train_office.py --rep learned --floor --dim 8 --out experiments/office_v2/learned_floor_d8_adv0 &
run office_v2_learned_floor_d8_adv3 src/train_office.py --rep learned --floor --dim 8 --adv 3.0 --out experiments/office_v2/learned_floor_d8_adv3 &
wait
echo "v2 training done $(date)"
until grep -q "queue done" logs/queue_v0.log; do sleep 60; done
for n in learned_floor_d8_adv0 learned_floor_d8_adv3; do
  run privacy_v2_$n src/attack_privacy.py --rep learned --ckpt experiments/office_v2/$n/model.pt --out experiments/privacy_v2/$n &
done
wait
for n in learned_floor_adv0 learned_floor_adv3; do
  run privacy_v2_posteriors_$n src/attack_privacy.py --rep posteriors --ckpt experiments/office_v1/$n/model.pt --out experiments/privacy_v2/posteriors_$n &
done
wait
echo "queue v2 done $(date)"
