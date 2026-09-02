# Model and release artifact licenses

The repository's `LICENSE` covers the build/release tooling and documentation. It
does **not** automatically relicense third-party model weights or voice packs.
Before publishing a release, preserve upstream notices and verify that the source
license allows redistribution.

| Release/profile             | Upstream                                                                                                        | Status used by this repository                                                                                                                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kokoro v1.0                 | `onnx-community/Kokoro-82M-v1.0-ONNX-timestamped` at pinned revision `dd4401a9add81ac692d20e240d22ec9dda82cc29` | Apache-2.0 upstream model; timestamped ONNX files are mirrored unchanged and voices are deterministically repacked with source provenance                                                                                                            |
| Kokoro v1.1 Chinese         | `onnx-community/Kokoro-82M-v1.1-zh-ONNX` at pinned revision `6cc0f0d2ebe369a68b0df87c2b65c1af8c0ac3e3`          | Apache-2.0 upstream model; ONNX files are mirrored unchanged and voices are deterministically repacked with source provenance                                                                                                                        |
| German Martin               | `Godelaune/Kokoro-82M-ONNX-German-Martin`                                                                       | Apache-2.0 as declared by the Hugging Face model repository; the requested `kokoro-martin.onnx` and `voices-martin.npz` files are mirrored unchanged and provenance is recorded in `release-manifest.json`                                           |
| Vietnamese                  | `contextboxai/Kokoro-Vietnamese` / `anphunl/Kokoro-Vietnamese`                                                  | Apache-2.0                                                                                                                                                                                                                                           |
| Vietnamese Ngọc Huyền       | `dinhthuan/kokoro-vi-ngoc-huyen`                                                                                | Apache-2.0 as declared upstream; the pinned checkpoint and Ngọc Huyền voice are converted to timestamped ONNX release assets.                                                                                                                        |
| Arabic Nabra                | `marwanelamami/Nabra-82M-v0.1-ONNX` / base `oddadmix/Nabra-82M-v0.1`                                            | Apache-2.0 as declared by the ONNX packaging repository; it identifies the oddadmix fine-tune as base-model provenance. Preserve both conversion-source and base-model provenance in release metadata/documentation                                  |
| German Kerstin              | `crane-local-ai/Kokoro-82M-v1.0-German-ONNX`                                                                    | Apache-2.0; publisher states Kerstin 1.0 training data is CC0-1.0                                                                                                                                                                                    |
| Hebrew NC                   | `thewh1teagle/kokoro-hebrew-nc`                                                                                 | **Restricted/non-commercial**; publication is disabled by default                                                                                                                                                                                    |
| Swedish Joakim              | `Joakim/kokoro-sv-voices`                                                                                       | Apache-2.0 as declared by the Hugging Face repository. The model and 10 voice packs are rebuilt into repository release artifacts from the pinned upstream revision. Preserve upstream provenance and training-data attributions.                    |
| German Thorsten             | `Thorsten-Voice/Kokoro`                                                                                         | Apache-2.0 as declared upstream; the Thorsten-Voice dataset is described upstream as CC0/public domain. The default epoch-5 `model.pth` and matching `voices/thorsten.pt` are converted/repacked.                                                    |
| European Portuguese logus2k | `logus2k/kokoro_tts_eu_pt`                                                                                      | Apache-2.0 as declared by the upstream model card. The `tuga_kokoro.pth` checkpoint and `tuga_voicepack.pt` voicepack are converted into timestamped ONNX release assets. Preserve the upstream model-card attribution and `tts_eu_pt` notices.      |
| Kazakh AnuarSv              | `AnuarSv/kokoro-tts-kazakh`                                                                                     | Apache-2.0. The pinned `kokoro_kazakh.pth` checkpoint is converted to ONNX and the matching `km_m1.pt` voice is repacked. Preserve the upstream `LICENSE` and model-card attribution in the release.                                                 |
| Russian Zaakirio            | `zaakirio/kokoro-ru`                                                                                            | The upstream model card declares OpenRAIL model weights and Apache-2.0 code. The base and Dima checkpoints are converted into separate timestamped ONNX releases, with matching voice packs, pinned source provenance, and upstream terms preserved. |
| Thai Wayu                   | `kunato/wayu-kokoro-thai-v1`                                                                                    | Apache-2.0 as declared upstream. Mirror the pinned ONNX serving bundle unchanged and preserve its split-graph/runtime provenance.                                                                                                                    |

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
