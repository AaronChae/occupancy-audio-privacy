# occupancy-audio-privacy

Code for the preprint *Privacy-Preserving Acoustic Occupancy Sensing with One-Second Embeddings and Self-Supervised Temporal Models* (link will be added after submission).

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
