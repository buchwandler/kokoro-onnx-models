# kokoro-onnx-models

Canonical build and release repository for Kokoro-family ONNX model assets used by
[`buchwandler/pykokoro`](https://github.com/buchwandler/pykokoro).

The goal is to give `pykokoro` one stable GitHub release origin:

```text
https://github.com/buchwandler/kokoro-onnx-models/releases/download/<tag>/<asset>
```

instead of hard-coding multiple upstream GitHub and Hugging Face repositories in
the runtime library.

## What belongs here

- scripts that convert compatible Kokoro checkpoints to ONNX;
- scripts that pack Kokoro `.pt` style tables into deterministic `voices.bin`;
- release manifests with SHA-256 hashes and provenance;
- migration tooling to mirror the release assets already consumed by `pykokoro`;
- documentation of model/front-end compatibility and third-party licenses.

Large model binaries should live in **GitHub Releases**, not in git history.

## Existing `pykokoro` releases to mirror

```bash
python scripts/mirror_release.py v1.0
python scripts/mirror_release.py v1.1-zh
python scripts/mirror_release.py v1.2-de-martin
```

Each command creates `dist/<release-tag>/` and a `release-manifest.json`. Upload
those files to a release with the same tag in this repository. Keeping the tags
and filenames unchanged makes the initial `pykokoro` migration small.

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
| `ar-nabra` | Arabic | `oddadmix/Nabra-82M-v0.1` | yes |
| `de-crane` | German | `crane-local-ai/Kokoro-82M-v1.0-German-ONNX` | yes |
| `he-hebrew-nc` | Hebrew | `thewh1teagle/kokoro-hebrew-nc` | **no** — restricted/non-commercial |

`build_kokoro.py` validates voice tensor shape, writes raw little-endian float32
speaker tables, validates the ONNX I/O types, preserves source ONNX metadata, and
adds speaker/source/frontend metadata.

## Frontends matter

An acoustic model being Kokoro-compatible does not mean the same text frontend
works for every language. The profile records the expected G2P path:

- Vietnamese requires `vig2p`-compatible preprocessing.
- Nabra Arabic requires diacritized MSA plus its Arabic phoneme cleanup.
- German Kerstin was trained on German IPA; verify `pykokoro`/`kokorog2p` output
  against the model's expected phoneme distribution.
- Hebrew uses a Hebrew-specific frontend and has restricted terms.

These requirements should ultimately be consumed by `pykokoro` from release
metadata rather than scattered hard-coded branches.

## Release workflow

For a mirrored release:

```bash
python scripts/mirror_release.py v1.0
gh release create model-files-v1.0 dist/model-files-v1.0/* \
  --repo buchwandler/kokoro-onnx-models \
  --title "Kokoro model files v1.0"
```

For a newly built model:

```bash
uv run scripts/build_kokoro.py build ar-nabra --out build
uv run scripts/prepare_release.py ar-nabra
gh release create model-files-arabic-nabra-v0.1 \
  dist/model-files-arabic-nabra-v0.1/* \
  --repo buchwandler/kokoro-onnx-models \
  --title "Kokoro Arabic Nabra v0.1"
```

Review `MODEL_LICENSES.md` before every first publication of an upstream model.
Do not use `--allow-restricted` for Hebrew unless you have independently verified
that redistribution is permitted by the original gated model/dataset terms.

## Licensing

Repository-authored tooling and documentation are Apache-2.0. Model and voice
artifacts keep their upstream terms; see [`MODEL_LICENSES.md`](MODEL_LICENSES.md)
and per-release manifests.

## Not a Kokoro voice pack: Inflect voices

`Danny-Dasilva/inflect-kokoro-voices` contains complete Inflect/VITS-family model
checkpoints, not Kokoro 256-dimensional style tables. They must not be concatenated
into Kokoro `voices.bin` and are intentionally excluded from this repository's
Kokoro release builder.
