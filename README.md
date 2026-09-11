# kokoro-onnx-models

Canonical build and release repository for Kokoro-family ONNX model assets used by
[`buchwandler/pykokoro`](https://github.com/buchwandler/pykokoro).

The canonical client inventory is `catalog/models.json`. It describes immutable runtime distributions, exact artifact URLs, hashes, formats, voices, frontends, and provenance.

GitHub Releases are an optional mirror provider for redistribution-compatible models:
```text
https://github.com/buchwandler/kokoro-onnx-models/releases/download/<tag>/<asset>
```

Direct upstream distributions use pinned HTTPS URLs and standard-library downloads; `pykokoro` does not require `huggingface_hub` to consume registry artifacts. The Zaakirio Russian runtime distributions are served from this repository's checkpoint-built GitHub releases.

## What belongs here

- scripts that convert compatible Kokoro checkpoints to ONNX;
- scripts that pack Kokoro `.pt` style tables into deterministic `voices.bin`;
- release manifests with SHA-256 hashes and provenance;
- migration tooling to mirror the release assets already consumed by `pykokoro`;
- documentation of model/front-end compatibility and third-party licenses.

The runtime registry is the source of truth for usable models. Build-only checkpoints and `.pt` voice sources stay in `scripts/kokoro_profiles.json` as conversion recipes and are not exposed as client runtime artifacts. A registry distribution is selected atomically, so its model, voices, config, and split components cannot be mixed with another provider.

`mirror_policy` distinguishes runtime availability from redistribution: `required`, `preferred`, `optional`, and `forbidden`. Russian Zaakirio is built into separate GitHub releases for its base and Dima checkpoints.

Large model binaries should live in **GitHub Releases**, not in git history.

## Existing `pykokoro` releases to mirror

```bash
python scripts/mirror_release.py v1.0
python scripts/mirror_release.py v1.1-zh
python scripts/mirror_release.py v1.2-de-martin
```

Each command creates `dist/<release-tag>/` and a `release-manifest.json`. Upload
those files to a release with the same tag in this repository. The v1.0 mirror uses
the new immutable `model-files-v1.0-timestamped-r3` revision; the historical
`model-files-v1.0-timestamped` and `model-files-v1.0-timestamped-r2` releases remain unchanged.

The `v1.2-de-martin` command mirrors the exact Apache-2.0 German Martin files from
`Godelaune/Kokoro-82M-ONNX-German-Martin` at `main`: `kokoro-martin.onnx` and
`voices-martin.npz`. The files are copied unchanged, then renamed to the established
release filenames.
## New Kokoro-family profiles

```bash
uv run scripts/build_kokoro.py list
uv run scripts/build_kokoro.py build vi-contextbox --out build
uv run scripts/prepare_release.py vi-contextbox
```

Supported build profiles:

| Profile | Language | Source | Publish by default? |
|---|---|---|---|
| `vi-contextbox` | Vietnamese | `contextboxai/Kokoro-Vietnamese` | yes |
| `vi-anphunl` | Vietnamese | `anphunl/Kokoro-Vietnamese` | yes |
| `vi-ngoc-huyen` | Vietnamese | `dinhthuan/kokoro-vi-ngoc-huyen` | yes |
| `ar-nabra` | Arabic | `marwanelamami/Nabra-82M-v0.1-ONNX` | yes |
| `de-crane` | German | `crane-local-ai/Kokoro-82M-v1.0-German-ONNX` | yes |
| `he-hebrew-nc` | Hebrew | `thewh1teagle/kokoro-hebrew-nc` | **no** — restricted/non-commercial |
| `sv-joakim` | Swedish | `Joakim/kokoro-sv-voices` | yes |
| `de-thorsten` | German | `Thorsten-Voice/Kokoro` | yes |
| `de-anna` | German | `software-mansion/react-native-executorch-kokoro` | yes |
| `pl-mateusz` | Polish | `software-mansion/react-native-executorch-kokoro` | yes |
| `pt-eu-logus2k` | European Portuguese | `logus2k/kokoro_tts_eu_pt` | yes |
| `th-wayu` | Thai | `kunato/wayu-kokoro-thai-v1` | yes, mirror/split ONNX |
| `kk-anuarsv` | Kazakh | `AnuarSv/kokoro-tts-kazakh` | yes |
| `ru-zaakirio-base` | Russian | `zaakirio/kokoro-ru` | yes, checkpoint build |
| `ru-zaakirio-dima` | Russian | `zaakirio/kokoro-ru` | yes, checkpoint build |

The Swedish source revision is pinned after the upstream stock-Kokoro checkpoint format fix. Its optional upstream post-processing recommendation uses notch filters at 2400, 4800, 7200, and 9600 Hz with Q=35; those filters are not baked into the ONNX graph.

The Thorsten release converts upstream's default epoch-5 `model.pth` and matching `voices/thorsten.pt`. Its German frontend requires the training-time `ʏ -> y` normalization.
The European Portuguese profile uses the upstream `tts_eu_pt` TugaPhone pt-PT Lisbon frontend and the Kokoro base config pinned from `hexgrad/Kokoro-82M`. Its ONNX release exposes per-token frame durations for timestamp-aware consumers.
The Software Mansion Anna and Mateusz profiles use the pinned converted
checkpoints at `9a8b5878012e01a26dad2618068dc61215994785`, the stock
`hexgrad/Kokoro-82M` architecture config, and their matching style tables.

`de-anna` is runtime-ready through pykokoro's `german-ipa-v1` frontend using
kokorog2p and the `df_anna` voice.

`pl-mateusz` remains staged because the current runtime frontend stack does not
provide the required Polish text frontend.

Thai Wayu is a split ONNX serving bundle (prosody + curves + decoder), not a single KModelForONNX graph. The release contains all graph components, source parameters, the upstream ONNX manifest, and voice/style archives.

Nabra uses the upstream pre-exported FP32 ONNX model. This repository repackages
the `af_msa` voice table and retains the model-specific `vocab.json`; it does not
re-export `kokoro_arabic.pth`. The original `oddadmix/Nabra-82M-v0.1` fine-tune
remains part of the model lineage.


The Zaakirio Russian source has two acoustic checkpoints. Sveta and Masha use the
base checkpoint, while Dima uses a dedicated checkpoint. The repository builds and
releases them as separate profiles so each release has a single unambiguous
model-to-voice mapping with validated token durations.

Kazakh uses language code `kk`. It is not published under `ka`, which is the
Georgian language code.
`build_kokoro.py` validates voice tensor shape, writes raw little-endian float32
speaker tables, validates the ONNX I/O types, preserves source ONNX metadata, and
adds speaker/source/frontend metadata.

## Frontends matter

An acoustic model being Kokoro-compatible does not mean the same text frontend
works for every language. The profile records the expected G2P path:

- Vietnamese requires `vig2p`-compatible preprocessing.
- `vi-ngoc-huyen` uses the upstream `vig2p==0.1.0` frontend and the single Ngọc Huyền voice.
- Nabra Arabic requires diacritized MSA plus its Arabic phoneme cleanup.
- German Kerstin was trained on German IPA; verify `pykokoro`/`kokorog2p` output
  against the model's expected phoneme distribution.
- Software Mansion Anna is runtime-ready through pykokoro's `german-ipa-v1`
  frontend using kokorog2p and the `df_anna` voice.
- Software Mansion Mateusz uses the upstream Phonemis Polish frontend as its reference;
  current `kokorog2p` rejects `pl`, so no generic fallback is permitted.
- Hebrew uses a Hebrew-specific frontend and has restricted terms.

- Zaakirio Russian requires stress-aware `ru_g2p.py` behavior, including `ё`,
  homographs, vowel reduction, and Russian orthoepy.
- AnuarSv Kazakh uses `kk` espeak-ng IPA through the Kazakh frontend.
These requirements should ultimately be consumed by `pykokoro` from release
metadata rather than scattered hard-coded branches.

## Release workflow

The runtime inventory is `catalog/models.json`. Kokoro v1.0 is mirrored from the timestamped ONNX Community distribution (`onnx-community/Kokoro-82M-v1.0-ONNX-timestamped`) so its native token durations are available; v1.1-zh is mirrored from its pinned ONNX Community revision. Each is published as a separate immutable profile release.
Kokoro v1.0 r3 contains the existing 54 stock voices plus seven externally tuned,
Kokoro-compatible Remsky voice assets. These are voice assets only and do not replace
or modify the v1.0 timestamped ONNX model.
The Remsky repository declares Apache-2.0 and its exact pinned revision is recorded in
`source-assets.json`. Four names refer to real people, so maintainers must complete legal
and policy review of name, likeness, publicity, personality, and voice rights before
public redistribution.

For a local candidate build and verification, see [`docs/LOCAL_RELEASE_TESTING.md`](docs/LOCAL_RELEASE_TESTING.md). Maintainers can build every publishable catalog entry through GitHub Actions using **Actions > release-all > Run workflow**. The workflow builds the complete candidate matrix before its protected `publish-all` job publishes independent release tags.
The Software Mansion Anna profile is runtime-ready through pykokoro's
`german-ipa-v1` frontend and its published release is synchronized into the
runtime registry. The Polish Mateusz profile remains staged because its required
text frontend is not available; synthetic-token export success alone is not runtime
readiness.
For a single local mirror candidate:

```bash
python scripts/mirror_release.py v1.0
python scripts/verify_candidate.py dist/model-files-v1.0-timestamped-r3 \
  --expected-tag model-files-v1.0-timestamped-r3
  --expected-profile v1.0
```

The normal workflow never deletes or overwrites an existing release. A missing tag is
published, an equivalent tag is skipped, and a differing tag fails. The v1.0 r3 release
is a new immutable tag, while `model-files-v1.0-timestamped` and
`model-files-v1.0-timestamped-r2` remain historical releases.
After publishing a release manually, validate its assets and synchronize the runtime catalog with the published bytes:

```bash
python scripts/sync_registry_from_release.py de-thorsten
git diff -- catalog/models.json
git add catalog/models.json
git commit -m "chore: synchronize model registry from release"
```

The sync command downloads every manifest asset, checks its size and SHA-256, validates the candidate, then updates `catalog/models.json`.

Review `MODEL_LICENSES.md` before every first publication of an upstream model. Do not use `--allow-restricted` for Hebrew unless you have independently verified that redistribution is permitted by the original gated model and dataset terms.

## Licensing

Repository-authored tooling and documentation are Apache-2.0. Model and voice
artifacts keep their upstream terms; see [`MODEL_LICENSES.md`](MODEL_LICENSES.md)
and per-release manifests.

## Not a Kokoro voice pack: Inflect voices

`Danny-Dasilva/inflect-kokoro-voices` contains complete Inflect/VITS-family model
checkpoints, not Kokoro 256-dimensional style tables. They must not be concatenated
into Kokoro `voices.bin` and are intentionally excluded from this repository's
Kokoro release builder.
