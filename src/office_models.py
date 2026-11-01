"""Representations and heads for privacy-preserving office activity sensing.

All representations map 16 kHz audio to a sequence of feature vectors:
    SecondEncoder  learned on-device encoder, one d-dim vector per second (default d = 32)
    logmel_stats   mean and std of each log-mel band per second (128 values)
    envelope       HPDmobile-style envelope: 8 kHz, mean shift, full-wave rectify, keep every
                   100th sample (80 values per second, log-compressed)
    logmel_frames  log-mel frames at 100 Hz (no privacy, reference for attacks only)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 16000


class LogMel(nn.Module):
    """Log-mel frames. With floor=True the 10th percentile of each band over the input (the room's
    noise floor) is subtracted, so features are relative to the background of the room and microphone."""

    def __init__(self, n_mels=64, floor=False):
        super().__init__()
        self.floor = floor
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, win_length=400, hop_length=160,
                                                        n_mels=n_mels)

    def forward(self, wav):
        """(B, N) -> (B, n_mels, N // 160) log-mel frames."""
        m = torch.log(self.mel(wav.float())[..., : wav.shape[-1] // 160] + 1e-6)
        return m - torch.quantile(m, 0.1, dim=-1, keepdim=True) if self.floor else m


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class SecondEncoder(nn.Module):
    """On-device encoder: each 1-s window is encoded independently into one d-dim vector."""

    def __init__(self, d=32, n_mels=64, floor=False):
        super().__init__()
        self.logmel = LogMel(n_mels, floor)
        self.norm = nn.BatchNorm1d(n_mels)
        self.cnn = nn.Sequential(block(1, 32), nn.AvgPool2d(2), block(32, 64), nn.AvgPool2d(2),
                                 block(64, 128), nn.AvgPool2d(2), block(128, 128))
        self.proj = nn.Linear(256, d)

    def forward(self, wav):
        """(B, T * 16000) -> (B, T, d)."""
        B = wav.shape[0]
        m = self.norm(self.logmel(wav))                      # (B, mel, T*100)
        T = m.shape[-1] // 100
        m = m[..., : T * 100].reshape(B, m.shape[1], T, 100).permute(0, 2, 1, 3)
        h = self.cnn(m.reshape(B * T, 1, m.shape[2], 100)).mean(2)   # (B*T, 128, time)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.proj(h).reshape(B, T, -1)


class LogMelStats(nn.Module):
    dim = 128

    def __init__(self, floor=False):
        super().__init__()
        self.logmel = LogMel(64, floor)

    def forward(self, wav):
        m = self.logmel(wav)
        T = m.shape[-1] // 100
        m = m[..., : T * 100].reshape(m.shape[0], 64, T, 100)
        return torch.cat([m.mean(-1), m.std(-1)], 1).transpose(1, 2)   # (B, T, 128)


class Envelope(nn.Module):
    """HPDmobile-style envelope as described in Scientific Data 8, 280 (2021), log-compressed."""
    dim = 80

    def forward(self, wav):
        x = torchaudio.functional.resample(wav.float(), SR, 8000)
        x = (x - x.mean(-1, keepdim=True)).abs()[..., ::100]           # 80 Hz
        T = x.shape[-1] // 80
        return torch.log1p(1e3 * x[..., : T * 80]).reshape(x.shape[0], T, 80)


class LogMelFrames(nn.Module):
    """Reference without privacy: log-mel frames, 100 per second."""

    def __init__(self):
        super().__init__()
        self.logmel = LogMel(64)

    def forward(self, wav):
        return self.logmel(wav).transpose(1, 2)   # (B, T*100, 64)


class EventHead(nn.Module):
    """Server-side head: per-second multi-label event logits."""

    def __init__(self, d, n_classes, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, z):
        return self.mlp(self.norm(z))


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam):
    return _Reverse.apply(x, lam)


class SpeakerAdversary(nn.Module):
    """Predicts the speaker from the speech-active seconds of a scene (mean and max pooled)."""

    def __init__(self, d, n_speakers, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_speakers))

    def forward(self, z, active):
        """z: (B, T, d), active: (B, T) bool with at least one True per row."""
        w = active.float().unsqueeze(-1)
        mean = (z * w).sum(1) / w.sum(1).clamp(min=1)
        mx = z.masked_fill(~active.unsqueeze(-1), -1e4).amax(1)
        return self.mlp(torch.cat([mean, mx], 1))


def dann_lambda(step, total, lam):
    """Ramp from 0 to lam (Ganin and Lempitsky, 2015)."""
    p = step / max(total, 1)
    return lam * (2 / (1 + math.exp(-10 * p)) - 1)
