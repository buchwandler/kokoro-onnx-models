#!/usr/bin/env python3
"""CPU runtime adapter for the pinned Thai Wayu split ONNX bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HARMONICS = 9
UPSAMPLE_SCALE = 300
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 600
VOICED_THRESHOLD = 10.0
SINE_AMP = 0.1
NOISE_STD = 0.003
N_FFT = 20
HOP = 5


class ThaiWayuRuntime:
    def __init__(self, asset_dir: Path) -> None:
        import onnxruntime as ort

        self.asset_dir = asset_dir
        manifest = json.loads(
            (asset_dir / "onnx_manifest.json").read_text(encoding="utf-8")
        )
        self.vocab: dict[str, int] = manifest["vocab"]
        self.max_tokens = 510
        self.sessions = {
            name: ort.InferenceSession(
                str(asset_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"]
            )
            for name in ("prosody", "curves", "decoder")
        }
        source = np.load(asset_dir / "source_params.npz")
        self.source_weight = source["weight"]
        self.source_bias = source["bias"]
        self.window = source["window"]
        with np.load(asset_dir / "voices.npz", allow_pickle=False) as voices:
            self.voices = {name: voices[name] for name in voices.files}

    def phonemize(self, text: str) -> str:
        import kokorog2p

        result = kokorog2p.phonemize(text, language="th", return_ids=False)
        phonemes = result.phonemes
        if not isinstance(phonemes, str):
            phonemes = "".join(phonemes)
        return phonemes

    def synthesize(
        self, text: str, voice: str, *, speed: float = 1.0, seed: int = 1234
    ) -> np.ndarray:
        phonemes = self.phonemize(text)
        ids = [self.vocab[p] for p in phonemes if p in self.vocab]
        if not ids:
            raise RuntimeError("Thai frontend produced no vocabulary symbols")
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"Thai frontend produced {len(ids)} tokens, maximum is {self.max_tokens}"
            )
        if voice not in self.voices:
            raise KeyError(f"Unknown Thai voice: {voice}")

        pack = self.voices[voice]
        style = np.asarray(
            pack[min(max(len(ids) - 1, 0), pack.shape[0] - 1)], dtype=np.float32
        )
        style = style.reshape(1, 256)
        style_dur, style_acou = style[:, 128:], style[:, :128]
        pred_dur, d, t_en = self.sessions["prosody"].run(
            None,
            {
                "input_ids": np.asarray([[0, *ids, 0]], dtype=np.int64),
                "style_dur": style_dur,
                "speed": np.asarray([speed], dtype=np.float32),
            },
        )
        index = np.repeat(np.arange(pred_dur.shape[0], dtype=np.int64), pred_dur)
        en = np.ascontiguousarray(d.transpose(0, 2, 1)[:, :, index])
        asr = np.ascontiguousarray(t_en[:, :, index])
        f0_curve, n_curve = self.sessions["curves"].run(
            None, {"en": en, "style_dur": style_dur}
        )
        har = self._harmonic_source(f0_curve, seed)
        (audio,) = self.sessions["decoder"].run(
            None,
            {
                "asr": asr,
                "f0_curve": f0_curve,
                "n_curve": n_curve,
                "style_acou": style_acou,
                "har": har,
            },
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _harmonic_source(self, f0_curve: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        f0 = np.repeat(np.asarray(f0_curve, dtype=np.float64), UPSAMPLE_SCALE, axis=1)[
            ..., None
        ]
        rad = f0 * np.arange(1, HARMONICS + 1, dtype=np.float64) / SAMPLE_RATE
        rad -= np.floor(rad)
        rand_ini = rng.random((1, HARMONICS))
        rand_ini[:, 0] = 0.0
        rad[:, 0, :] += rand_ini
        phase = np.cumsum(self._resample(rad, 1 / UPSAMPLE_SCALE), axis=1) * 2 * np.pi
        phase = self._resample(phase * UPSAMPLE_SCALE, UPSAMPLE_SCALE)
        voiced = (f0 > VOICED_THRESHOLD).astype(np.float64)
        amplitude = voiced * NOISE_STD + (1 - voiced) * SINE_AMP / 3
        noise = rng.standard_normal((1, f0.shape[1], HARMONICS))
        waves = np.sin(phase) * SINE_AMP * voiced + amplitude * noise
        merged = np.tanh(waves @ self.source_weight.T + self.source_bias)
        return self._stft(merged[0, :, 0].astype(np.float32))

    @staticmethod
    def _resample(values: np.ndarray, scale: float) -> np.ndarray:
        length = values.shape[1]
        output_length = int(length * scale)
        source = (np.arange(output_length, dtype=np.float64) + 0.5) / scale - 0.5
        np.clip(source, 0, length - 1, out=source)
        lower = np.floor(source).astype(np.int64)
        upper = np.minimum(lower + 1, length - 1)
        weight = (source - lower)[None, :, None]
        return values[:, lower] * (1 - weight) + values[:, upper] * weight

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        padded = np.pad(audio, N_FFT // 2, mode="reflect")
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=((len(padded) - N_FFT) // HOP + 1, N_FFT),
            strides=(padded.strides[0] * HOP, padded.strides[0]),
        )
        spectrum = np.fft.rfft(frames * self.window, axis=1).T[None]
        return np.concatenate([np.abs(spectrum), np.angle(spectrum)], axis=1).astype(
            np.float32
        )
