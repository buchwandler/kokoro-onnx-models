# Model and release artifact licenses

The repository's `LICENSE` covers the build/release tooling and documentation. It
does **not** automatically relicense third-party model weights or voice packs.
Before publishing a release, preserve upstream notices and verify that the source
license allows redistribution.

| Release/profile | Upstream | Status used by this repository |
|---|---|---|
| Kokoro v1.0 | `hexgrad/Kokoro-82M`; existing ONNX release from `thewh1teagle/kokoro-onnx` | Apache-2.0 upstream model; mirror provenance is recorded in the manifest |
| Kokoro v1.1 Chinese | `hexgrad/Kokoro-82M-v1.1-zh`; existing ONNX release from `thewh1teagle/kokoro-onnx` | Apache-2.0 upstream model |
| German v1.1 Eva/Bernd | existing release from `holgern/kokoro-onnx-model` | Preserve upstream release attribution; source model repository may no longer be available |
| German Martin v1.2 | existing release from `holgern/kokoro-onnx-model` | Verify the source model terms before first Buchwandler publication |
| Vietnamese | `contextboxai/Kokoro-Vietnamese` / `anphunl/Kokoro-Vietnamese` | Apache-2.0 |
| Arabic Nabra | `oddadmix/Nabra-82M-v0.1` | Apache-2.0 |
| German Kerstin | `crane-local-ai/Kokoro-82M-v1.0-German-ONNX` | Apache-2.0; publisher states Kerstin 1.0 training data is CC0-1.0 |
| Hebrew NC | `thewh1teagle/kokoro-hebrew-nc` | **Restricted/non-commercial**; publication is disabled by default |

## Danny-Dasilva/inflect-kokoro-voices

Do not add these `model.pth` files to Kokoro `voices.bin`. They are complete
Inflect-Micro-v2/VITS-family TTS checkpoints trained on Kokoro-generated audio,
not Kokoro style-vector tables. Supporting them requires a separate inference
backend.

## Release rule

Every published release should contain `release-manifest.json` with source
repository/revision, declared license, file sizes, and SHA-256 hashes. Where an
upstream project ships a NOTICE or attribution file, include it unchanged in the
release as well.
