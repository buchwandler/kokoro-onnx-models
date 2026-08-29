# Thorsten waveform-health calibration

This report records the deterministic calibration fixtures used by the absolute
waveform-health gate. The incident WAV is not committed to the repository.

## Detector

The stationary broadband-noise detector requires all of the following for at
least one second:

- zero-crossing rate above `0.45`;
- spectral centroid above `0.23 * sample_rate`;
- spectral-centroid CV below `0.05`;
- energy above 4 kHz greater than `0.65` of spectral energy;
- frame-RMS CV below `0.08`;
- normalized spectral flux below `0.05`.

Metrics use 50 ms Hann-windowed frames with a 25 ms hop and pure NumPy
operations. The existing peak, RMS, DC, frame-envelope, and stationary-tone
checks remain enabled.

## Deterministic fixtures

| Fixture                                          |     Seed | Duration | Detector result | Other gate                                   |
| ------------------------------------------------ | -------: | -------: | --------------- | -------------------------------------------- |
| high-frequency differenced noise, incident-like  | 20260828 |      2 s | reject          | legacy metrics pass with incident thresholds |
| modulated harmonic speech-like signal plus noise |     1234 |      2 s | accept          | legacy metrics pass                          |

The incident-like fixture is generated in `tests/test_export_validation.py`:

```python
rng = np.random.default_rng(20260828)
noise = rng.standard_normal(48_000)
noise = noise - np.roll(noise, 1)
noise = 0.055 * noise / np.std(noise) - 0.015
noise = np.clip(noise, -0.32, 0.32).astype(np.float32)
```

The unit test first disables only the new detector and verifies that the old
metrics accept the fixture. It then verifies that the detector and the complete
health function reject it. The speech-like fixture verifies that the compound
rule does not reject a nonstationary native-speech-shaped waveform.

## Runtime calibration status

Native checkpoint calibration requires the pinned Thorsten checkpoint, voice,
Torch, and build dependencies. The release workflow must run every configured
checkpoint case over the configured repeated seeds and retain the resulting
JSON metrics before promoting the thresholds to a release artifact. A failed
native speech case blocks promotion rather than being hidden by threshold
changes.
