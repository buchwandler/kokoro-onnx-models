# Checkpoint export audit

Date: 2026-08-28

The exporter profile inventory identifies five checkpoint-derived profiles that use the shared `scripts/build_kokoro.py` checkpoint path:

- `de-thorsten` using `model_ep5.pth`
- `sv-joakim` using `kokoro_sv.pth`
- `kk-anuarsv` using `kokoro_kazakh.pth`
- `vi-contextbox` using `kokoro_vi.pth`
- `vi-anphunl` using `kokoro_vi.pth`

These profiles are subject to the repaired exporter on their next build. They are suspect until rebuilt and validated with native stochastic export, ONNX Runtime execution, duration checks, waveform-health checks, and representative listening.

The profiles `ar-nabra`, `de-crane`, and `he-hebrew-nc` use prebuilt or mirrored ONNX assets and are not changed by this exporter repair.

## Available graph inspection

The available workspace graphs were inspected with ONNX using the accepted random-source operator set:

| Build profile | Accepted random-source operators |
| --- | --- |
| `de-thorsten` | `RandomNormalLike`, `RandomUniformLike` |
| `sv-joakim` | `RandomNormalLike`, `RandomUniformLike` |
| `ar-nabra` | `RandomNormalLike`, `RandomUniformLike` |

The regenerated `build/de-thorsten` bundle now contains exporter provenance, both frozen waveform-validation cases, and the accepted random operators. The pre-existing `build/sv-joakim` bundle has no exporter provenance, so its graph presence does not prove that it was produced by the repaired exporter. The corrected Thorsten build measured DC offset, frame-RMS variation, and stationary-tone ratio across native PyTorch and ONNX Runtime outputs; the profile thresholds were calibrated to the observed stochastic output while still rejecting the stationary-tone regression fixture.

No changes were made to mirrored or prebuilt profiles. Other checkpoint profiles require fresh builds and profile-specific audio validation before publication; this task does not publish unvalidated releases.
