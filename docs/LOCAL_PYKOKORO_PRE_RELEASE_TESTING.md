# Local pykokoro pre-release testing for `kokoro-onnx-models`

## Goal

Before publishing a GitHub Release, exercise the exact local ONNX model and voice
artifacts with the current `pykokoro` checkout. The test data must stay local:
model weights, voice packs, temporary compatibility archives, Hugging Face cache
data, and generated WAV files must never be committed.

The tracked test harness is:

```text
local_test/
├── common.py
├── prepare_local_assets.py
├── smoke_v1_0.py
├── smoke_v1_1_zh.py
├── smoke_de_martin.py
├── smoke_vi_contextbox.py
├── smoke_vi_anphunl.py
├── smoke_ar_nabra.py
├── smoke_de_crane.py
└── smoke_he_hebrew_nc.py
```

All large/runtime artifacts go here:

```text
.local-test/
├── assets/<catalog-key>/
├── build/
├── downloads/
├── compat/
└── wav/<catalog-key>/
```

`.local-test/` and `*.wav` are ignored by git.

## Important findings from the current repositories

### 1. Current `pykokoro` expects named NumPy voice archives

`pykokoro.voice_manager.VoiceManager` treats a `.bin` file as a NumPy `.npz`
archive and loads it with `numpy.load()`. This is true even though the extension
is `.bin`.

The builder now emits two explicit voice artifacts for build profiles:
`voices.npz` is a named NumPy archive for pykokoro and `voices.raw.bin` is the
headerless little-endian float32 speaker table for sherpa-onnx. The release
preparation step preserves both formats and records their formats in the manifest.

The harness prefers the named NumPy archive for an exact voice-package test. If
only a raw archive is staged, it can create a local-only `.npz` compatibility
archive from the raw bytes and `bundle.json`, but that is not a release-format pass.

The strict gate rejects raw-only staging:

```bash
python local_test/smoke_vi_contextbox.py --strict-release-format
```

Do not solve this by silently renaming a raw file to `.npz`; the bytes must
actually be a NumPy archive.

### 2. New profiles and explicit local configuration

`pykokoro.config_types.ModelVariant` now includes:

```text
v1.0
v1.1-zh
v1.2-de-martin
vi-contextbox
vi-anphunl
ar-nabra
de-crane
he-hebrew-nc
```

The new profiles have release metadata and can be selected with explicit local
model, named voice archive, and config or release-manifest paths. They remain
experimental until their required frontends and golden phoneme fixtures pass.
The local scripts continue to use `v1.0` as an integration shim for unsupported
frontends, so those runs are not evidence of automatic profile compatibility.

### 3. Frontend compatibility is separate from ONNX compatibility

The repository profile metadata already records this, and the local tests must
not erase that distinction.

| Profile          | Required/expected frontend                                                       | Current test meaning                                       |
| ---------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `v1.0`           | current pykokoro/kokorog2p frontend                                              | real integration smoke test                                |
| `v1.1-zh`        | current pykokoro Chinese frontend                                                | real integration smoke test                                |
| `v1.2-de-martin` | current pykokoro German frontend                                                 | real integration smoke test                                |
| `vi-contextbox`  | `vig2p`                                                                          | experimental until pykokoro has matching frontend          |
| `vi-anphunl`     | `vig2p`                                                                          | experimental until pykokoro has matching frontend          |
| `ar-nabra`       | normalize → diacritize → Arabic espeak → Nabra cleanup, plus Nabra symbols/vocab | experimental until implemented                             |
| `de-crane`       | German IPA; training data used `espeak-ng` German IPA                            | useful smoke test, but model contract must also be checked |
| `he-hebrew-nc`   | Hebrew-specific G2P/config                                                       | experimental and restricted/non-commercial                 |

For unsupported frontends the harness refuses to synthesize ordinary text by
default. `--allow-frontend-mismatch` enables an **experimental** espeak-backed
smoke test. A generated WAV in that mode does not prove pronunciation quality or
training-time frontend parity.

### 4. ONNX input typing is model-driven

The Crane Kerstin upstream ONNX uses `input_ids`, `style`, and float `speed`.
`pykokoro.audio_generator.AudioGenerator` now inspects ONNX input metadata and
constructs each input using the declared dtype. This avoids inferring a tensor type
from `model_source` and is covered by unit tests for float and integer speed inputs.

### 5. Custom vocabulary/config needs an explicit local path or embedded-model path

Nabra and Hebrew have model-specific vocabulary requirements. `PipelineConfig` now
accepts `model_config_path` or `release_manifest_path`; the manifest resolver
selects the local ONNX, named voice archive, and config without triggering unrelated
upstream downloads. Profiles that require downloaded config metadata still need an
explicit config file when running from local release assets.

Do not test a custom-vocabulary model with the stock v1.0 vocabulary and call it
compatible merely because ONNX inference returned samples.

## Environment

Use the included `pykokoro` checkout so the smoke test matches the code that will
consume the releases.

Example with `uv`:

```bash
cd /path/to/pykokoro
uv sync --extra cpu
```

or an editable install:

```bash
python -m pip install -e "/path/to/pykokoro[cpu]"
```

Then run the test commands from the `kokoro-onnx-models` repository.

Build profiles need the model repository build dependencies:

```bash
python -m pip install -e ".[build]"
```

If `pykokoro` and `kokoro-onnx-models` use separate virtual environments, use a
single test environment containing both sets of dependencies.

## Populate local assets

### Automated local preparation using repository source-of-truth

The helper reuses `catalog/releases.json`, `scripts/mirror_release.py`, and
`scripts/build_kokoro.py`. Nothing is written to a tracked model directory.

Prepare one profile:

```bash
python local_test/prepare_local_assets.py v1.2-de-martin
python local_test/prepare_local_assets.py vi-contextbox
```

The Martin preparation command follows the catalog's Hugging Face mirror entry and downloads the exact
Godelaune `kokoro-martin.onnx` and `voices-martin.npz` source files.

Prepare all catalog entries:

```bash
python local_test/prepare_local_assets.py all
```

This creates normalized local directories such as:

```text
.local-test/assets/v1.2-de-martin/
├── model.onnx
├── voices.bin
└── release-manifest.json

.local-test/assets/vi-contextbox/
├── model.onnx
├── voices.bin
├── bundle.json
└── config.json
```

### Manually downloaded files

Manual staging is also supported. Put the files into the same normalized layout:

```text
.local-test/assets/<catalog-key>/model.onnx
.local-test/assets/<catalog-key>/voices.bin
```

For raw builder voice packs also copy:

```text
.local-test/assets/<catalog-key>/bundle.json
```

`bundle.json` supplies the speaker names/order needed for the local raw-to-npz
compatibility conversion.

For single-speaker raw files, the harness also knows the expected speaker names
for Nabra, Kerstin, and Hebrew.

## Run the supported release smoke tests

```bash
python local_test/smoke_v1_0.py
python local_test/smoke_v1_1_zh.py
python local_test/smoke_de_martin.py
```

Each script:

1. opens the local voice archive;
2. enumerates **every voice**;
3. selects a language-appropriate sentence;
4. reuses one local ONNX backend while iterating voices where pykokoro permits;
5. checks for non-empty finite audio;
6. prints the resulting phoneme stream;
7. writes one WAV per voice below `.local-test/wav/`.

For a real pre-release gate, use:

```bash
python local_test/smoke_de_martin.py --strict-release-format
```

The strict flag prevents any compatibility repacking.

Test one voice only:

```bash
python local_test/smoke_v1_0.py --voice af_heart
```

Do not write WAV:

```bash
python local_test/smoke_v1_0.py --no-write
```

## Run the new-profile diagnostic tests

### Vietnamese ContextBoxAI

```bash
python local_test/prepare_local_assets.py vi-contextbox
python local_test/smoke_vi_contextbox.py --strict-release-format
```

The strict run currently catches the raw-voice-package mismatch.

For a local-only acoustic experiment:

```bash
python local_test/smoke_vi_contextbox.py --allow-frontend-mismatch
```

Expected voices include:

```text
diem_trinh
hung_thinh
mai_linh
mai_loan
manh_dung
my_yen
ngoc_huyen
phat_tai
thanh_dat
thuc_trinh
tuan_ngoc
storyvert
duc_an
duc_duy
```

Upstream Vietnamese inference uses `vig2p`; the espeak experiment is not a
frontend-parity test.

### Vietnamese anphunl mirror

```bash
python local_test/prepare_local_assets.py vi-anphunl
python local_test/smoke_vi_anphunl.py --strict-release-format
python local_test/smoke_vi_anphunl.py --allow-frontend-mismatch
```

This should be compared against the ContextBoxAI result because the repository
describes anphunl as a mirror.

### Arabic Nabra

```bash
python local_test/prepare_local_assets.py ar-nabra
python local_test/smoke_ar_nabra.py --strict-release-format
```

The native-language sample is already diacritized to remove one variable from the
smoke test, but upstream Nabra still requires its Arabic phoneme cleanup and
custom symbols. Do not publish Nabra as pykokoro-compatible until those are
represented in the pykokoro profile/frontend.

Experimental only:

```bash
python local_test/smoke_ar_nabra.py --allow-frontend-mismatch
```

### German Kerstin / Crane

```bash
python local_test/prepare_local_assets.py de-crane
python local_test/smoke_de_crane.py --strict-release-format
```

A raw-only manually staged voice file is expected to fail the exact current-pykokoro
gate. The builder's named `voices.npz` output passes the voice-package check, while
the separate `voices.raw.bin` remains available for sherpa consumers.

The ONNX input dtype is now read from model metadata, including the float `speed`
contract used by Crane.

### Hebrew NC

```bash
python local_test/prepare_local_assets.py he-hebrew-nc
python local_test/smoke_he_hebrew_nc.py --strict-release-format
```

Do not publish this profile by default. Its non-commercial/restricted upstream
terms remain the governing constraint.

Experimental only:

```bash
python local_test/smoke_he_hebrew_nc.py --allow-frontend-mismatch
```

## Recommended coding-agent changes before new releases

### A. Decide one canonical pykokoro voice artifact format

Recommended for this repository/consumer pairing: publish a named `.npz`
container (the filename may remain `voices-*.bin` for backward compatibility if
desired, but the bytes must be npz), because current pykokoro needs speaker names.

If sherpa-onnx also requires raw packed float32, publish **two explicit assets**,
for example:

```text
voices-<profile>.npz
voices-<profile>.raw.bin
```

Do not overload one filename with two incompatible byte formats.

Update the release manifest with an explicit format field, e.g.:

```json
{
  "name": "voices-vietnamese-v1.0.npz",
  "role": "voices",
  "format": "numpy-npz",
  "speaker_names": ["diem_trinh", "hung_thinh"]
}
```

### B. Add model profiles to pykokoro

Add first-class pykokoro profiles for:

```text
vi-contextbox
vi-anphunl
ar-nabra
de-crane
he-hebrew-nc
```

Each profile should describe at least:

```text
release repository
release tag/revision
model filename
voice filename
SHA-256 and size
language tags
speaker names/default speaker
sample rate
tokenizer vocabulary source
frontend/G2P strategy
ONNX input contract
suggested speed
license/restriction flags
```

Longer term, prefer reading this from `release-manifest.json` instead of
duplicating constants in pykokoro.

### C. Make ONNX input typing model-driven

Do not infer speed dtype from `model_source`.

At session initialization inspect:

```python
for input_meta in session.get_inputs():
    print(input_meta.name, input_meta.type, input_meta.shape)
```

Build `tokens`/`input_ids`, `style`, and `speed` according to the actual ONNX
metadata. Add tests for:

```text
tokens + float speed
input_ids + float speed
input_ids + int32 speed
```

### D. Add arbitrary local config/vocabulary support

For local pre-release testing, pykokoro should accept something like:

```python
PipelineConfig(
    model_path=...,
    voices_path=...,
    model_config_path=...,
)
```

or a resolved release manifest path. Explicit local model paths should never
silently trigger unrelated upstream config downloads.

### E. Implement frontend adapters before enabling automatic profiles

Required minimum:

```text
Vietnamese: vig2p-compatible adapter
Arabic Nabra: normalization + diacritization + Arabic phonemization + Nabra cleanup
German Kerstin: verified German IPA path
Hebrew: model-specific Hebrew G2P/config
```

Add golden phoneme fixtures from each upstream implementation. Audio-only tests
are insufficient because a wrong frontend can still produce non-empty audio.

## Release gate

A profile is ready for publication to be consumed by pykokoro only when all of
the following are true:

- repository tests pass;
- ONNX checker/contract validation passes;
- local pykokoro test uses **explicit local paths** and performs no model download;
- `--strict-release-format` passes;
- every advertised speaker synthesizes non-empty finite audio;
- language-appropriate text produces expected/golden phonemes;
- custom vocabulary is loaded from the intended release metadata/config;
- ONNX input dtypes are derived correctly;
- sample rate is 24 kHz as expected;
- release SHA-256/size values match the tested bytes;
- generated WAV/model/voice files are absent from `git status`;
- license/redistribution review is complete;
- Hebrew remains disabled unless its upstream restrictions have been independently cleared.

## Useful git safety checks

Before committing:

```bash
git status --short
git check-ignore -v .local-test/wav/v1.2-de-martin/martin.wav
git check-ignore -v .local-test/assets/v1.2-de-martin/model.onnx
```

Search for accidentally staged binary artifacts:

```bash
git diff --cached --name-only -- \
  '*.onnx' '*.pt' '*.pth' '*.bin' '*.npz' '*.wav'
```

The output should be empty.

## Current upstream observations (verified 2026-08-26)

These are outside the repository snapshot and should be rechecked when changing
profiles:

- `contextboxai/Kokoro-Vietnamese` currently publishes `kokoro_vi.onnx` directly,
  in addition to the checkpoint, and documents ONNX inference with `vig2p`.
  The build profile can potentially mirror/copy that ONNX instead of exporting
  the checkpoint again, after byte/contract validation.
- `anphunl/Kokoro-Vietnamese` documents the same 14 Vietnamese voices and `vig2p`.
- `oddadmix/Nabra-82M-v0.1` explicitly requires diacritized MSA, Arabic espeak,
  Nabra-specific phoneme cleanup, and dedicated `ʕ`/`ħ` vocabulary entries.
- `crane-local-ai/Kokoro-82M-v1.0-German-ONNX` documents `df_kerstin` as a
  headerless raw float32 voice file and an ONNX contract with float speed.
- `thewh1teagle/kokoro-hebrew-nc` remains non-commercial/restricted and ships
  `he_shaul` plus a Hebrew config.

Re-verify upstream revisions and licenses before pinning or publishing.
