"""Office audio data: isolated events, speech, rooms (RIR + empty-room noise) and a GPU scene synthesizer.

Sources under data/raw (licenses checked on the hosting pages):
    dcase2016_task2  isolated office events, 11 classes (archive.org: CC BY 3.0 train/dev, CC BY 4.0 test)
    dcase2013        isolated office events, 16 classes, and real office recordings OL (archive.org: CC BY 4.0)
    librispeech      read speech: train-clean-100, dev-clean, test-clean (openslr: CC BY 4.0)
    but_reverbdb     room impulse responses and empty-room noise ("silence") of 9 rooms
                     (project page: CC BY 4.0; read_me.txt: Apache 2.0)
"""
import concurrent.futures as cf
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000
CLASSES = ["alert", "clearthroat", "cough", "doorslam", "drawer", "keyboard", "keys", "knock", "laughter",
           "mouse", "pageturn", "pendrop", "phone", "printer", "speech", "switch"]
DEVICE_CLASSES = ("alert", "phone", "printer")       # sounds that do not need a person in the room
HUMAN = torch.tensor([c not in DEVICE_CLASSES for c in CLASSES])
SPEECH = CLASSES.index("speech")


_RESAMPLERS = {}


def load(path, sr=SR):
    x, r = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if r == sr:
        return x
    if (r, sr) not in _RESAMPLERS:   # the resampling kernel is built once per rate pair
        _RESAMPLERS[(r, sr)] = torchaudio.transforms.Resample(r, sr)
    return _RESAMPLERS[(r, sr)](x)


def active_frames(x, db=30.0):
    """Boolean mask of 10-ms frames whose energy is within db of the loudest frame."""
    n = x.numel() // 160
    e = 10 * torch.log10(x[: n * 160].reshape(n, 160).pow(2).mean(1) + 1e-12)
    return e > e.max() - db


def unit_active_rms(x, act):
    a = x[: act.numel() * 160].reshape(-1, 160)[act]
    return x / a.pow(2).mean().sqrt().clamp(min=1e-8)


def event_bank(raw, split):
    """[(class index, waveform, active mask)]; per dataset and class, instances 1-15 are train, 16-20 test."""
    dirs = [raw / "dcase2016_task2/dcase2016_task2_train_dev/dcase2016_task2_train_dev/dcase2016_task2_train",
            raw / "dcase2013/dcase2013_event_detection_training/x/singlesounds_stereo"]
    items = []
    for d in dirs:
        byc = {}
        for p in sorted(d.glob("*.wav")):
            c = re.fullmatch(r"([a-zA-Z]+?)(\d+)\.wav", p.name)[1]
            byc.setdefault("keys" if c == "keysDrop" else c, []).append(p)
        for c, ps in sorted(byc.items()):
            for p in (ps[:15] if split == "train" else ps[15:]):
                x = load(p)
                act = active_frames(x)
                items.append((CLASSES.index(c), unit_active_rms(x, act), act))
    return items


def librispeech_index(raw, subset, per_speaker=None, seed=0):
    """[(speaker, path, transcript)] for a LibriSpeech subset; optional random subset per speaker."""
    root = raw / "librispeech/LibriSpeech" / subset
    rng = np.random.default_rng(seed)
    out = []
    for spk in sorted(root.iterdir(), key=lambda p: int(p.name)):
        trans = {}
        for t in spk.glob("*/*.trans.txt"):
            for line in open(t):
                k, v = line.strip().split(" ", 1)
                trans[k] = v
        files = sorted(spk.glob("*/*.flac"))
        if per_speaker and len(files) > per_speaker:
            files = [files[i] for i in sorted(rng.choice(len(files), per_speaker, replace=False))]
        out += [(spk.name, f, trans[f.stem]) for f in files]
    return out


def speaker_sex(raw):
    sex = {}
    for line in open(raw / "librispeech/LibriSpeech/SPEAKERS.TXT"):
        if not line.startswith(";"):
            p = [s.strip() for s in line.split("|")]
            sex[p[0]] = p[1]
    return sex


def load_many(paths, workers=16):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(load, paths))


def room_bank(raw, rooms, rir_seconds=1.0, max_noise=24, max_rir=300, seed=0):
    """{room: {"rir": [tensor], "noise": [tensor]}} from BUT ReverbDB.

    Noise is taken from the 60-s "silence" recordings (empty room). Speaker setups whose microphone
    metadata lists a second loudspeaker ($EnvSpk2*, a background radio) are skipped for noise.
    A random subset of max_noise noise recordings and max_rir RIRs is kept per room.
    """
    base = raw / "but_reverbdb"
    rng = np.random.default_rng(seed)
    out = {}
    for room in rooms:
        place = next(p for p in base.iterdir() if p.is_dir() and p.name.endswith(room))
        rir_paths, noise_paths = [], []
        for setup in sorted(place.glob("MicID*/SpkID*")):
            radio = any("$EnvSpk2" in m.read_text(errors="ignore") for m in setup.glob("*/mic_meta.txt"))
            rir_paths += sorted(setup.glob("*/RIR/*.wav"))
            if not radio:
                noise_paths += sorted(setup.glob("*/silence/*.wav"))

        def pick(ps, k):
            return [ps[i] for i in sorted(rng.choice(len(ps), min(k, len(ps)), replace=False))]

        rirs = [h[: int(rir_seconds * SR)] for h in load_many(pick(rir_paths, max_rir))]
        out[room] = {"rir": [h / h.pow(2).sum().sqrt().clamp(min=1e-8) for h in rirs],
                     "noise": [n for n in load_many(pick(noise_paths, max_noise)) if n.numel() >= SR]}
    return out


def _fftconv(x, h):
    """Batched full convolution of (K, N) and (K, M) on the current device."""
    n = x.shape[-1] + h.shape[-1] - 1
    L = 1 << (n - 1).bit_length()
    return torch.fft.irfft(torch.fft.rfft(x, L) * torch.fft.rfft(h, L), L)[..., :n]


class SceneSynth:
    """Synthesizes reverberant office scenes on the GPU.

    Each scene: empty-room noise of one room + up to max_events isolated events and, with probability
    p_speech, one speech segment; every source is convolved with an RIR of the same room.
    Returns waveforms (B, seconds*SR), per-second labels (B, seconds, C) and speaker ids (-1 = no speech).
    A second is labelled active for a class if any 10-ms frame of the dry source is active in it.
    Labels are built on the CPU so that synthesis needs no GPU synchronization.
    """

    def __init__(self, events, speech, speakers, rooms, device, seconds=10, p_speech=0.4, max_events=4,
                 ebr=(-10, 15), snr=(-5, 20), gain_db=10):
        self.sp, self.spk, self.rooms = speech, speakers, rooms
        self.dev, self.T, self.p_speech, self.max_events = device, seconds, p_speech, max_events
        self.ebr, self.snr, self.gain_db = ebr, snr, gain_db
        self.room_names = sorted(rooms)
        for r in rooms.values():   # keep audio sources on the GPU
            r["rir"] = [h.to(device) for h in r["rir"]]
            r["noise"] = [n.to(device) for n in r["noise"]]
        self.ev = [(c, x.to(device), np.asarray(act.cpu())) for c, x, act in events]

    def _label(self, act, onset, lab, b, c):
        f = np.zeros(self.T * 100, bool)
        idx = np.nonzero(act)[0] + onset // 160
        f[idx[(idx >= 0) & (idx < f.size)]] = True
        lab[b, :, c] += f.reshape(self.T, 100).sum(1)

    def batch(self, B, rng):
        N, dev = self.T * SR, self.dev
        scene = torch.zeros(B, N, device=dev)
        lab = np.zeros((B, self.T, len(CLASSES)), np.int32)
        spk = torch.full((B,), -1, dtype=torch.long)
        srcs = []   # (scene, dry waveform, active frames, class, onset, level in dB, rir)
        for b in range(B):
            room = self.rooms[self.room_names[rng.integers(len(self.room_names))]]
            noise = room["noise"][rng.integers(len(room["noise"]))]
            start = int(rng.integers(0, noise.numel()))
            bg = noise.repeat(-(-N // noise.numel()) + 1)[start: start + N]
            scene[b] = bg / bg.pow(2).mean().sqrt().clamp(min=1e-8)
            for _ in range(rng.integers(0, self.max_events + 1)):
                c, x, act = self.ev[rng.integers(len(self.ev))]
                onset = int(rng.integers(-x.numel() // 2, N))
                srcs.append((b, x, act, c, onset, rng.uniform(*self.ebr), room["rir"][rng.integers(len(room["rir"]))]))
            if rng.random() < self.p_speech:
                i = rng.integers(len(self.sp))
                x = torch.from_numpy(self.sp[i].astype(np.float32) / 32768)
                L = int(rng.uniform(2, 8) * SR)
                if x.numel() > L:
                    s0 = rng.integers(0, x.numel() - L)
                    x = x[s0: s0 + L]
                act = active_frames(x)
                if act.any():
                    onset = int(rng.integers(0, max(N - x.numel(), 1)))
                    srcs.append((b, unit_active_rms(x, act).to(dev), act.numpy(), SPEECH, onset,
                                 rng.uniform(*self.snr), room["rir"][rng.integers(len(room["rir"]))]))
                    spk[b] = self.spk[i]
        if srcs:
            X = torch.zeros(len(srcs), max(s[1].numel() for s in srcs), device=dev)
            H = torch.zeros(len(srcs), max(s[6].numel() for s in srcs), device=dev)
            for k, s in enumerate(srcs):
                X[k, : s[1].numel()] = s[1]
                H[k, : s[6].numel()] = s[6]
            W = _fftconv(X, H)
            for k, (b, x, act, c, onset, db, h) in enumerate(srcs):
                n = x.numel() + h.numel() - 1
                s0, s1 = max(onset, 0), min(onset + n, N)
                if s1 > s0:
                    scene[b, s0:s1] += 10 ** (db / 20) * W[k, s0 - onset: s1 - onset]
                    self._label(act, onset, lab, b, c)
        gain = torch.from_numpy(10 ** (rng.uniform(-self.gain_db, self.gain_db, B) / 20)).float().to(dev)
        return scene * gain[:, None] * 0.05, torch.from_numpy(lab >= 1).float().to(dev), spk


def _labels_from_txt(paths, T):
    y = torch.zeros(T, len(CLASSES))
    for ann in paths:
        for line in open(ann):
            p = line.split()
            if len(p) >= 3 and p[2] in CLASSES:
                on, off = float(p[0]), float(p[1])
                y[int(on): min(T, int(np.ceil(off))), CLASSES.index(p[2])] = 1
    return y


def dcase2013_ol(raw):
    """Real office recordings (DCASE 2013 OL test): [(name, waveform, per-second labels)], union of annotators."""
    d = raw / "dcase2013/dcase2013_event_detection_testset_OL/x/dcase2013_event_detection_testset_OL"
    out = []
    for w in sorted(d.glob("*.wav")):
        x = load(w)
        T = x.numel() // SR
        out.append((w.stem, x[: T * SR], _labels_from_txt(sorted(d.glob(w.stem + "_*.txt")), T)))
    return out


def dcase2016_test(raw):
    """Synthetic office mixtures (DCASE 2016 task 2 public test): [(name, waveform, per-second labels)]."""
    d = raw / "dcase2016_task2/dcase2016_task2_test_public/dcase2016_task2_test_public"
    out = []
    for w in sorted((d / "sound").glob("*.wav")):
        x = load(w)
        T = x.numel() // SR
        out.append((w.stem, x[: T * SR], _labels_from_txt([d / "annotation" / (w.stem + ".txt")], T)))
    return out
