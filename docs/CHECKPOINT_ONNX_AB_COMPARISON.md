# Checkpoint ONNX A/B comparison

This repository's checkpoint export gate compares the upstream-native PyTorch
model with the exported ONNX model using the same frozen model inputs. It is a
conversion-fidelity check and is separate from the final-consumer pykokoro
smoke test.

## Three execution stages

1. **Native Torch** is a `kokoro.KModel` created with `disable_complex=False`.
   It is captured before export-specific changes are installed.
2. **Patched Torch** is a separate model instance with the export-compatible
   decoder replacement. It is compared strictly with the native reference when
   the replacement promises equivalent semantics.
3. **ONNX Runtime** executes the serialized export, normally with
   `CPUExecutionProvider`.

The term native Torch must not describe the export-compatible model. This
separation prevents a defect shared by the export path and its PyTorch check
from being mistaken for conversion parity.

## Frozen inputs

Cases live in `scripts/kokoro_profiles.json` under `export_validation.cases`.
Each case pins phonemes, token IDs, speed, and human-readable source text. The
tokens, selected voice style row, and speed are built once and passed unchanged
to all three stages. The comparison does not run separate G2P paths.

Checkpoint profiles need at least two representative cases, including a short
case and language or model-specific difficult phonemes. Prebuilt ONNX profiles
cannot be compared directly because they do not provide a real pinned native
checkpoint reference.

## Running the comparison

Install the build dependencies, then run:

```bash
uv run --extra build python local_test/compare_checkpoint_onnx.py de-thorsten
```

Useful options are:

```text
--case NAME          select one case; repeat for more than one
--voice NAME         select a voice explicitly
--runs N             number of ONNX samples, default 3
--seed INTEGER       override the profile export seed
--provider NAME      ONNX Runtime provider, default CPUExecutionProvider
--build-root PATH    temporary export directory
--output-dir PATH    comparison artifact directory
--keep-build        retain the temporary export directory
--no-wav            calculate metrics without writing WAV files
```

The default output is `.local-test/compare/<profile>`. It contains
`report.json`, `report.md`, and one directory per case with
`torch-native.wav`, `torch-patched.wav`, and repeated `onnx-01.wav` files.
Files are PCM16 conversions of the original values. They are clipped only for
PCM encoding and are never independently normalized, so amplitude differences
remain audible evidence.

## Automated checks and metrics

Duration is a hard contract. The number of duration entries, integer frame
values, minimum frame size, audio length, and native versus ONNX duration values
are checked. Waveform health checks finite, non-empty audio, peak, RMS, DC
offset, DC-to-RMS ratio, frame-RMS variation, and stationary-tone ratio.

Stochastic full-model audio is not required to be sample-identical. Each ONNX
run is measured separately. Structural comparisons report envelope correlation,
frame-RMS coefficient-of-variation ratio, absolute DC difference, and RMS
ratio. These metrics detect conversion failures without treating random samples
as deterministic byte streams.

`report.json` records source revision, selected voice, seed, runtime provenance,
case timing, metrics, structural comparisons, artifact paths, and
`listening.status`. Automation always writes `listening.status` as
`not-recorded`.

## Human listening gate

After automated checks pass, listen to the native and at least two ONNX samples
for every representative case. Check intelligibility, pronunciation, timing,
prosody, noise, clipping, silence, and obvious differences in loudness or
artifacts. Recording WAV files does not pass this gate. The report must not be
changed to claim a listening pass without a reviewer recording that evidence.

## Relationship to release smoke testing

The export A/B gate is:

```text
native checkpoint -> patched Torch -> ONNX Runtime
```

It tests conversion fidelity and uses frozen token/style/speed inputs. The
pykokoro release smoke gate is:

```text
published or staged ONNX + voices + config -> pykokoro
```

It tests the final packaged consumer, frontend routing, voice assets, config,
input names, and release format. Both gates are required for a checkpoint
derived release, and neither substitutes for the other.

## Native metric distributions

The comparison captures at least three native references for each frozen input. ONNX-safe PyTorch and repeated ONNX Runtime outputs are checked against a native-derived envelope using the median plus or minus the greater of three MAD values and a metric-specific floor. The report retains the native distribution and envelope for review.

## Prepared artifact gate

Release automation runs the consumer smoke test against the prepared release directory. When `release-manifest.json` is present, `local_test/common.py` verifies every prepared asset's size and SHA-256 before pykokoro starts. Checkpoint-derived publication also requires a comparison report whose listening status is explicitly recorded as `pass`.
