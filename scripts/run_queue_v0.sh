#!/usr/bin/env bash
# Runs the remaining v0/v1 experiments in order on the compute node, without user input.
# Each stage waits for the previous one. Logs go to logs/.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { local name=$1; shift; "$PY" "$@" > "logs/$name.log" 2>&1; }

echo "queue start $(date)"
# A. wait for the v0 office runs that are already running
while pgrep -f "src/train_office.py" > /dev/null; do sleep 60; done
echo "A done $(date)"

# B. privacy attacks on the v0 representations
run privacy_v0_learned_adv0 src/attack_privacy.py --rep learned --ckpt experiments/office_v0/learned_adv0/model.pt --out experiments/privacy_v0/learned_adv0 &
run privacy_v0_learned_adv1 src/attack_privacy.py --rep learned --ckpt experiments/office_v0/learned_adv1/model.pt --out experiments/privacy_v0/learned_adv1 &
run privacy_v0_logmel_stats src/attack_privacy.py --rep logmel_stats --out experiments/privacy_v0/logmel_stats &
wait
run privacy_v0_envelope src/attack_privacy.py --rep envelope --out experiments/privacy_v0/envelope &
run privacy_v0_logmel_frames src/attack_privacy.py --rep logmel_frames --out experiments/privacy_v0/logmel_frames &
wait
echo "B done $(date)"

# C. v1 office models: noise-floor normalization, adversary weight 0, 1, 3
run office_v1_learned_floor_adv0 src/train_office.py --rep learned --floor --out experiments/office_v1/learned_floor_adv0 &
run office_v1_learned_floor_adv1 src/train_office.py --rep learned --floor --adv 1.0 --out experiments/office_v1/learned_floor_adv1 &
run office_v1_learned_floor_adv3 src/train_office.py --rep learned --floor --adv 3.0 --out experiments/office_v1/learned_floor_adv3 &
run office_v1_logmel_stats_floor src/train_office.py --rep logmel_stats --floor --out experiments/office_v1/logmel_stats_floor &
wait
echo "C done $(date)"

# D. privacy attacks on the v1 representations
for n in learned_floor_adv0 learned_floor_adv1 learned_floor_adv3; do
  run privacy_v1_$n src/attack_privacy.py --rep learned --ckpt experiments/office_v1/$n/model.pt --out experiments/privacy_v1/$n &
done
run privacy_v1_logmel_stats_floor src/attack_privacy.py --rep logmel_stats --floor --out experiments/privacy_v1/logmel_stats_floor &
wait
echo "D done $(date)"

# E. HPDmobile: label efficiency (transductive pretraining), then two more seeds (one run at a time)
for ld in 1 3 7; do run hpd_ld$ld src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0 --transductive --label-days $ld; done
for s in 1 2; do
  run hpd_strict_s$s src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0 --seed $s
  run hpd_trans_s$s src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0 --transductive --seed $s
done
echo "queue done $(date)"
