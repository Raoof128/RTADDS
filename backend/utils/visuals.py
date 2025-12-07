"""Visualization helpers for spectrograms and signals."""
from __future__ import annotations

import base64
import io
from typing import Dict

import librosa
import librosa.display
import matplotlib

# Use non-interactive backend for server environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


def waveform_plot(waveform: np.ndarray, sample_rate: int) -> str:
    """Generate a waveform plot and return as base64 string."""
    fig, ax = plt.subplots(figsize=(6, 2))
    times = np.arange(len(waveform)) / sample_rate
    ax.plot(times, waveform, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def spectrogram_plot(waveform: np.ndarray, sample_rate: int) -> str:
    """Generate a mel spectrogram plot as base64 string."""
    S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def jitter_plot(jitter_series: np.ndarray) -> str:
    """Plot jitter-related series as base64 string."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(jitter_series, linewidth=0.8)
    ax.set_title("Micro-Jitter Proxy")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Jitter")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def artifact_heatmap(artifact_map: np.ndarray) -> str:
    """Render artifact map heatmap."""
    fig, ax = plt.subplots(figsize=(6, 3))
    cax = ax.imshow(artifact_map, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Vocoder Artifact Map")
    fig.colorbar(cax, ax=ax)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()
