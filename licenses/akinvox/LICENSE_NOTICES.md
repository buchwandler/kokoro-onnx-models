# AkinVox cloning bundle license notices

This bundle contains independently licensed components. The AkinVox Apache-2.0 declaration applies to AkinVox contributions only and does not relicense Kokoro, StyleTTS2, WavLM, or frontend dependencies.

- The AkinVox source repository's `LICENSE`, `NOTICE`, and `THIRD_PARTY_LICENSES.md` are included unchanged.
- Kokoro's upstream license is included as `KOKORO_LICENSE.txt`.
- StyleTTS2/iSTFTNet's upstream license is included as `STYLE_TTS2_LICENSE.txt`.
- The pinned `microsoft/wavlm-base-plus-sv` model card at revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` links to Microsoft UniSpeech's license. The license file included here is from `microsoft/UniSpeech` commit `6112826ac13a4327f4c9a7afa2a505e35b763514` and identifies Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0).
- AkinVox's upstream `THIRD_PARTY_LICENSES.md` describes WavLM as MIT, which conflicts with the license linked by the pinned model card. Both the upstream notice and the model-card-linked license are preserved. This discrepancy requires maintainer review; this profile is not publishable until resolved.
- The AkinVox English frontend identifies Misaki 0.9.4 under Apache-2.0. eSpeak NG and phonemizer are separate GPL-3.0 runtime dependencies where used. These frontend components are not included in the ONNX model assets and retain their own terms.

The complete upstream notices and license texts are provided as separate assets. No single top-level license applies to the entire bundle. This summary is informational and is not legal advice.
