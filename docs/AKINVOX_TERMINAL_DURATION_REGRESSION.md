# AkinVox terminal-duration regression

The `terminal-sounds-regression` case freezes the supported PyKokoro frontend output for “It was just as uncomfortable as it sounds.”, language `en-US`, voice `AV_brody`, and speed `1.0`. It records 46 frontend token IDs and 48 model input IDs including boundary tokens. The terminal word `sounds` maps to model-token span `[40, 46]` with phonemes `sˈWndz`.

The diagnostic compares native PyTorch, patched PyTorch, and ONNX Runtime through the existing AkinVox parity path. It verifies matching duration arrays and exact duration-derived sample counts. It does not pad, trim, or otherwise modify audio.

## Local reproduction

Run the named case with the project build dependencies:

```bash
uv run --python 3.12 --extra build --extra test \
  python local_test/compare_akinvox_multispeaker_onnx.py \
  --case terminal-sounds-regression --keep-build
```

This builds both language graphs, runs the selected English case and the existing German parity cases, then writes `.local-test/compare/akinvox-multispeaker-v1/report.json`. To inspect an already built full parity bundle, use the workflow style:

```bash
uv run --python 3.12 --extra build --extra test \
  python local_test/compare_akinvox_multispeaker_onnx.py \
  --build-root build --no-build --case terminal-sounds-regression
```

The existing release workflow builds this profile and runs the comparison report step, so the frozen case participates in the same native, patched, and ONNX validation phase. The local command does not write WAV files.

## Measured result

With seed `20261007`, native, patched, and ONNX duration arrays matched. Each path produced 103 frames and 61,800 samples, exactly `103 * 600`. The terminal `sounds` span had durations `[2, 5, 3, 1, 1, 3]`, totaling 15 frames or 375 ms.

The native and ONNX tail metrics were close. Last-50-ms RMS was approximately `4.50e-8` and `5.32e-8`; last-100-ms RMS was `4.64e-8` and `4.79e-8`; last-200-ms RMS was `0.002618` and `0.002595`, respectively. Both reported the last sample above the explicitly recorded float32-epsilon threshold at index 61,799. Terminal peaks were `0.02534` at sample 57,054 for native and `0.02605` at sample 57,057 for ONNX.

These waveform-tail values are diagnostic only. They are not pass/fail criteria and do not establish perceptual quality. This measurement shows no duration or output-length divergence that would justify an exporter-side waveform change.
