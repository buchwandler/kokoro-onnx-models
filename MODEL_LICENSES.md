# Model and release artifact licenses

The repository's `LICENSE` covers the build/release tooling and documentation. It
does **not** automatically relicense third-party model weights or voice packs.
Before publishing a release, preserve upstream notices and verify that the source
license allows redistribution.

| Release/profile                 | Upstream                                                                                                        | Status used by this repository                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Kokoro v1.0                     | `onnx-community/Kokoro-82M-v1.0-ONNX-timestamped` at pinned revision `dd4401a9add81ac692d20e240d22ec9dda82cc29` | Apache-2.0 upstream model; timestamped ONNX files are mirrored unchanged and voices are deterministically repacked with source provenance                                                                                                                                                                                                                                                                                                        |
| Kokoro v1.1 Chinese             | `onnx-community/Kokoro-82M-v1.1-zh-ONNX` at pinned revision `6cc0f0d2ebe369a68b0df87c2b65c1af8c0ac3e3`          | Apache-2.0 upstream model; ONNX files are mirrored unchanged and voices are deterministically repacked with source provenance                                                                                                                                                                                                                                                                                                                    |
| German Martin                   | `Godelaune/Kokoro-82M-ONNX-German-Martin`                                                                       | Apache-2.0 as declared by the Hugging Face model repository; the requested `kokoro-martin.onnx` and `voices-martin.npz` files are mirrored unchanged and provenance is recorded in `release-manifest.json`                                                                                                                                                                                                                                       |
| Vietnamese                      | `contextboxai/Kokoro-Vietnamese` / `anphunl/Kokoro-Vietnamese`                                                  | Apache-2.0                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Vietnamese Ngọc Huyền           | `dinhthuan/kokoro-vi-ngoc-huyen`                                                                                | Apache-2.0 as declared upstream; the pinned checkpoint and Ngọc Huyền voice are converted to timestamped ONNX release assets.                                                                                                                                                                                                                                                                                                                    |
| Arabic Nabra                    | `marwanelamami/Nabra-82M-v0.1-ONNX` / base `oddadmix/Nabra-82M-v0.1`                                            | Apache-2.0 as declared by the ONNX packaging repository; it identifies the oddadmix fine-tune as base-model provenance. Preserve both conversion-source and base-model provenance in release metadata/documentation                                                                                                                                                                                                                              |
| German Kerstin                  | `crane-local-ai/Kokoro-82M-v1.0-German-ONNX`                                                                    | Apache-2.0; publisher states Kerstin 1.0 training data is CC0-1.0                                                                                                                                                                                                                                                                                                                                                                                |
| Hebrew NC                       | `thewh1teagle/kokoro-hebrew-nc`                                                                                 | **Restricted/non-commercial**; publication is disabled by default                                                                                                                                                                                                                                                                                                                                                                                |
| Swedish Joakim                  | `Joakim/kokoro-sv-voices`                                                                                       | Apache-2.0 as declared by the Hugging Face repository. The model and 10 voice packs are rebuilt into repository release artifacts from the pinned upstream revision. Preserve upstream provenance and training-data attributions.                                                                                                                                                                                                                |
| German Thorsten                 | `Thorsten-Voice/Kokoro`                                                                                         | Apache-2.0 as declared upstream; the Thorsten-Voice dataset is described upstream as CC0/public domain. The default epoch-5 `model.pth` and matching `voices/thorsten.pt` are converted/repacked.                                                                                                                                                                                                                                                |
| European Portuguese logus2k     | `logus2k/kokoro_tts_eu_pt`                                                                                      | Apache-2.0 as declared by the upstream model card. The `tuga_kokoro.pth` checkpoint and `tuga_voicepack.pt` voicepack are converted into timestamped ONNX release assets. Preserve the upstream model-card attribution and `tts_eu_pt` notices.                                                                                                                                                                                                  |
| Kazakh AnuarSv                  | `AnuarSv/kokoro-tts-kazakh`                                                                                     | Apache-2.0. The pinned `kokoro_kazakh.pth` checkpoint is converted to ONNX and the matching `km_m1.pt` voice is repacked. Preserve the upstream `LICENSE` and model-card attribution in the release.                                                                                                                                                                                                                                             |
| Russian Zaakirio                | `zaakirio/kokoro-ru`                                                                                            | The upstream model card declares OpenRAIL model weights and Apache-2.0 code. The base and Dima checkpoints are converted into separate timestamped ONNX releases, with matching voice packs, pinned source provenance, and upstream terms preserved.                                                                                                                                                                                             |
| Software Mansion German Anna    | `software-mansion/react-native-executorch-kokoro` at `9a8b5878012e01a26dad2618068dc61215994785`                 | Apache-2.0 as declared by the upstream model card; the pinned converted German checkpoint and `voices/df_anna.bin` are repacked into staged ONNX/voice candidates. Preserve both per-file hashes and the Phonemis frontend attribution. Do not use the upstream XNNPACK manifest as the ONNX architecture config.                                                                                                                                |
| Software Mansion Polish Mateusz | `software-mansion/react-native-executorch-kokoro` at `9a8b5878012e01a26dad2618068dc61215994785`                 | Apache-2.0 as declared by the upstream model card; the pinned converted Polish checkpoint and final `voices/pm_mateusz.bin` are repacked into staged ONNX/voice candidates. Preserve both per-file hashes and the Phonemis frontend attribution. Polish remains unpublished until a compatible consumer frontend exists.                                                                                                                         |
| Kokoro English 7M Distill       | `oddadmix/Kokoro-7M-Distill` at pinned revision `2ef3dfc29bb8db3d796b98a5cff0dbaef76f54cf`                      | Apache-2.0 as declared upstream; the pinned 7M distilled checkpoint and `af_msa` style pack are converted and repacked. `af_msa` is the distillation style and the only published voice.                                                                                                                                                                                                                                                         |
| Thai Wayu                       | `kunato/wayu-kokoro-thai-v1`                                                                                    | Apache-2.0 as declared upstream. Mirror the pinned ONNX serving bundle unchanged and preserve its split-graph/runtime provenance.                                                                                                                                                                                                                                                                                                                |
| Kokoro v1.0 Remsky voices       | `remsky/kokoro-inno-clone-tuner` at pinned revision `b8fc665a0a110d663b2cc9e313ec28bcf6bbbbdb`                  | Apache-2.0 is declared by the Hugging Face repository. These seven generated Kokoro-compatible style packs are voice assets only; preserve upstream attribution and the exact revision in release provenance. Names referring to Amelia Earhart, Vincent Price, Jane Goodall, and David Attenborough require maintainer legal and policy review for name, likeness, publicity, personality, and voice rights before redistribution.              |
| Kokoro v1.0 Inno voicepack tuner | `remsky/kokoro-inno-clone-tuner` v0.2.0 at pinned revision `429617d18ce4d637acea948bdff4cce3ec6cf167` (code `remsky/inno-kokoro` @ `892ef184bc932aa3ff9d72c1509d5b81ff6941e6`) | Mixed licensing: Inno code/adapter Apache-2.0, WeSpeaker code Apache-2.0, VoxCeleb initialization CC BY 4.0, and the UniSpeech-SAT teacher/derived speaker encoder CC BY-SA 3.0. Do **not** describe `inno-voicepack-v0.2.onnx` as simply Apache-2.0. See the "Inno v0.2 voicepack tuner third-party notices" section below. |
| AkinVox Kokoro Cloning v1       | `AKinvox/kokoro-cloning-v1` at pinned revision `0094666f0a9038ce49789446eb8dd3ddfc848b43` (release `v1.0.1`)    | Component-specific terms. AkinVox code and adapter weights declare Apache-2.0, but that applies only to those contributions. Kokoro is Apache-2.0 and StyleTTS2/iSTFTNet is MIT. The pinned WavLM model card links to Microsoft UniSpeech CC BY-SA 3.0, while AkinVox's upstream third-party notice labels WavLM MIT. This conflict is preserved and publication is disabled pending maintainer review. See the bundled license assets below. No top-level AkinVox license relicenses dependencies. |

## AkinVox Kokoro Cloning v1 third-party notices

`en-akinvox-cloning-v1` combines several upstream projects under their own terms.
The generated release must ship the exact upstream notice files for each:

| Component                                                        | Upstream at pinned revision                                                 | Notice to preserve                             |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| AkinVox cloning adapter, reference mapper and reference encoders | `AKinvox/kokoro-cloning-v1` @ `0094666f0a9038ce49789446eb8dd3ddfc848b43`    | `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md` |
| Kokoro v1.0 base model and architecture                          | `hexgrad/Kokoro-82M` @ `f3ff3571791e39611d31c381e3a41a3af07b4987`           | `licenses/KOKORO_LICENSE`                      |
| StyleTTS2 / iSTFTNet decoder and harmonic-source code            | shipped with the AkinVox release                                            | `licenses/STYLE_TTS2_LICENSE`                  |
| WavLM speaker-verification encoder                              | `microsoft/wavlm-base-plus-sv` @ `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` | Pinned model card and CC BY-SA 3.0 license from `microsoft/UniSpeech` @ `6112826ac13a4327f4c9a7afa2a505e35b763514`; the AkinVox upstream notice calls it MIT, so publication is blocked pending review |

The reference-cloning candidate includes the upstream AkinVox `LICENSE`, `NOTICE`, and `THIRD_PARTY_LICENSES.md`, the Kokoro and StyleTTS2 license files, the pinned WavLM model card, its model-card-linked CC BY-SA 3.0 license text, and a project-authored `LICENSE_NOTICES.md` summary. These are separate license and attribution assets. The candidate is not publishable until the WavLM license discrepancy is resolved.
Boundary rules for this release:

- Build inputs (`adapter.pt`, `reference_mapper.pt`, `reference_encoders.pt`,
  `kokoro-v1_0.pth`, `pytorch_model.bin`) are build-time only. The published
  bundle is ONNX graphs, JSON metadata and one pickle-free NPZ archive.
- The optional Silero VAD used by the upstream enrollment path is not part of the
  exported runtime bundle, and its license is therefore not shipped here.
- Do not imply that the bundle as a whole is relicensed solely by the top-level
  AkinVox Apache-2.0 declaration.
- Record the exact source revisions above in `bundle.json`,
  `release-manifest.json` and the release notes.

## Inno v0.2 voicepack tuner third-party notices

The optional v1.0 enrollment artifacts `inno-voicepack-v0.2.onnx`,
`inno-tuner-v0.2.npz` and `inno-tuner-v0.2.json` are built from
`remsky/kokoro-inno-clone-tuner` v0.2.0 (`model.safetensors`, `config.json`) and
the `remsky/inno-kokoro` v0.2.0 source. The complete tuner artifact is **not**
Apache-2.0; it combines several upstream components under their own terms:

| Component                                                          | Upstream at pinned revision                                          | License to preserve |
| ------------------------------------------------------------------ | -------------------------------------------------------------------- | ------------------- |
| Inno code/adapter, style head and prosody head                     | `remsky/inno-kokoro` @ `892ef184bc932aa3ff9d72c1509d5b81ff6941e6`    | Apache-2.0          |
| WeSpeaker ResNet34 speaker-encoder code (`models/resnet.py`, TSTP) | vendored in the Inno source                                          | Apache-2.0          |
| VoxCeleb-trained ResNet34-LM initialization                        | VoxCeleb-derived ResNet34-LM weights                                 | CC BY 4.0           |
| UniSpeech-SAT teacher/derived speaker encoder weights (`enc.*`)    | distilled from `microsoft/unispeech-sat-base-plus-sv`                | CC BY-SA 3.0        |

Boundary rules for this release:

- `model.safetensors` and `config.json` are build-time inputs only. The
  published artifacts are one ONNX graph, a pickle-free NPZ tuner metadata
  archive, a JSON sidecar, provenance and checksums.
- The baked speaker encoder weights make the tuner more restrictive than the
  seven pre-generated Remsky voice assets; do not relicense them under the
  top-level Apache-2.0 declaration.
- The CC BY-SA 3.0 speaker encoder requires an explicit maintainer license
  review before the tuner is redistributed.
- Record the exact source revisions and per-file SHA-256 values in
  `bundle.json`, `release-manifest.json` and the release notes.

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
