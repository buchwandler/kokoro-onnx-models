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

| Build profile | Accepted random-source operators        |
| ------------- | --------------------------------------- |
| `de-thorsten` | `RandomNormalLike`, `RandomUniformLike` |
| `sv-joakim`   | `RandomNormalLike`, `RandomUniformLike` |
| `ar-nabra`    | `RandomNormalLike`, `RandomUniformLike` |

The pre-existing `model-files-german-thorsten-v1.1.1` release passed structural and weak waveform validation but produced DC-biased noise in pykokoro. It is retired and must not be selected or overwritten.

The v1.1.2 candidate uses the repository-owned `exact-convtranspose-istft-v1` decoder reconstruction and native `torch.istft` reference validation. A local candidate was built and verified with both frozen cases, exact duration parity, native-versus-patched reconstruction error below `1e-4`, ONNX Runtime waveform metrics, and retained random-source graph checks. The candidate still requires the documented manual listening gate before publication.

The corrected candidate measured low DC offset and DC-to-RMS ratio across the frozen cases. The profile thresholds were calibrated to measured seeded native and stochastic ONNX outputs while rejecting the deterministic DC-biased noise fixture.

No changes were made to mirrored or prebuilt profiles. Other checkpoint profiles require fresh builds and profile-specific audio validation before publication.
