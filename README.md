# occupancy-audio-privacy

Code for the preprint *From Sound to Occupancy: Privacy-Aware Audio Sensing for Buildings* (link will be added after submission).

The work has two parts:

- **Office activity sensing.** A small encoder runs on the sensing device and turns each second of audio into 32 numbers. A server-side head detects 16 office sound classes (13 made by people, 3 by devices) from these numbers. Every representation is attacked with speech reconstruction plus speech recognition, speaker identification and sex classification to measure what it leaks.
- **Long-term occupancy.** A causal dilated CNN is pretrained on unlabelled sensor streams with the hierarchical contrastive loss of TS2Vec and evaluated leave-one-home-out on HPDmobile.

## Layout

```
scripts/                 dataset downloads and the experiment queues
src/                     data preparation, models, training and attacks
notes/data_sources.md    dataset links, licenses and data checks
experiments/summary/     result tables used in the paper
experiments/hpd_ts2vec_v0/   per-home HPDmobile results
paper/                   LaTeX source and figures of the preprint
```

Raw data, processed arrays, model checkpoints and logs are not tracked.

## Setup

Python 3.11, PyTorch 2.8 and torchaudio 2.8 (CUDA). The remaining packages are listed in `requirements.txt`.

```
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## Data

All datasets are public. Links and licenses are in `notes/data_sources.md`.

```
bash scripts/download_hpdmobile.sh                  # HPDmobile, 13.6 GB, writes files.tsv
JOBS=6 bash scripts/download_hpdmobile_parallel.sh  # optional parallel download once files.tsv exists
python scripts/download_archive_org.py --dest data/raw/dcase2016_task2 \
    dcase2016_task2_train_dev dcase2016_task2_test_public
python scripts/download_archive_org.py --dest data/raw/dcase2013 \
    dcase2013_event_detection_training dcase2013_event_detection_development \
    dcase2013_event_detection_testset_OL dcase2013_event_detection_development_OS \
    dcase2013_event_detection_testset_OS
bash scripts/download_librispeech.sh                # dev-clean, test-clean, train-clean-100
```

Unpack the DCASE archives where the code expects them:

```
(cd data/raw/dcase2016_task2 && for z in */*.zip; do unzip -q -o "$z" -d "$(dirname "$z")"; done)
(cd data/raw/dcase2013 && for z in */*.zip; do unzip -q -o "$z" -d "$(dirname "$z")/x"; done)
```

The BUT Speech@FIT Reverb Database (RIR-only release 19_06, 8.7 GB) is linked from its project page and goes to `data/raw/but_reverbdb`.

## HPDmobile

```
python src/preprocess_hpdmobile.py --home H1 H2 H3 H4 H5 H6
python src/hpd_slots.py --home H1 H2 H3 H4 H5 H6
python src/hpd_check_alignment.py --home H1 H2 H3 H4 H5 H6
python src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0                  # pretraining on the training homes
python src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0 --transductive   # pretraining on all homes, no labels
python src/hpd_ts2vec.py --out experiments/hpd_ts2vec_v0 --transductive --label-days 1
```

`src/features_hpdmobile.py` and `src/baseline_occupancy.py` are the first feature baseline and are kept for reference.

## Office sounds and privacy attacks

```
python src/train_office.py --rep learned --floor --adv 3.0 --out experiments/office_v1/learned_floor_adv3
python src/attack_privacy.py --rep learned --ckpt experiments/office_v1/learned_floor_adv3/model.pt \
    --out experiments/privacy_v1/learned_floor_adv3
```

`scripts/run_queue_v0.sh` and `scripts/run_queue_v2.sh` run every configuration reported in the paper on one GPU. `src/summarize_results.py` collects the outputs into `experiments/summary/`.

`src/office_ci.py` computes bootstrap intervals of the activity AUC and per-room results for the trained models (`experiments/summary/office_ci_auc.csv`, `office_ci_diff.csv`).

`src/office_rescore.py` re-scores the trained models with AUC, average precision and EER on all seconds and on seconds without speech (`experiments/summary/office_rescore.csv`, `office_rescore_ci.json`). `src/check_dcase2016_leak.py` checks by normalized cross-correlation whether the DCASE 2016 test mixtures contain the released training sounds (`experiments/summary/dcase2016_leak.csv`).

## End-to-end pipeline: audio -> stage 1 -> stage 2

The paper evaluates the two stages separately because no public dataset pairs office audio with occupancy labels. `src/pipeline.py` joins them the way a deployment would, on one audio recording:

```
python src/pipeline.py stage1  --ckpt experiments/office_v1/learned_floor_adv3/model.pt \
    --start 2026-11-05T09:00:00 --out streams/room1 recordings/room1_day1.wav     # audio -> embeddings per second
python src/pipeline.py pretrain --streams streams/room1 streams/room2 --out models/stage2.pt   # TS2Vec, no labels
python src/pipeline.py fit      --streams streams/room1 --labels labels/room1.csv \
    --stage2 models/stage2.pt --out models/probe.joblib                           # linear probe with labels
python src/pipeline.py predict  recordings/room1_day2.wav --ckpt .../model.pt --start 2026-11-06T09:00:00 \
    --stage2 models/stage2.pt --probe models/probe.joblib --out pred.csv          # occupancy per 10-s slot
python src/pipeline.py evaluate --pred pred.csv --labels labels/room1_day2.csv
```

Stage 1 runs the trained encoder and activity head in 10-s blocks, as on the device. Stage 2 reads, per 10-s slot, the mean and maximum of the embeddings and three statistics of the human-activity score (plus coverage and a missing flag, 69 channels for d = 32), standardized per site without labels, and uses the temporal encoder, TS2Vec loss and probe of `src/hpd_ts2vec.py`. Labels are interval files (`start,end,occupied`); `src/labels_from_logs.py` builds them from entry and exit logs or head counts. `src/device_capture.py` is the on-device loop: microphone -> stage 1 -> `.npz` streams on disk, audio never written (`--source file` replays a wav for testing).

No stage-2 model trained on embedding streams exists yet; it needs the office recordings planned in the paper. Until then `stage1` and `predict` without `--stage2` give the per-second and per-slot activity scores of the trained encoder. See `notes/pipeline_2026-11-05.md`.
