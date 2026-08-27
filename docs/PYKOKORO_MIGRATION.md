# Migrating `buchwandler/pykokoro`

Target all GitHub model downloads at one repository:

```python
MODEL_RELEASE_REPOSITORY = "buchwandler/kokoro-onnx-models"
```

Existing tags and filenames are intentionally preserved for the three retained

| pykokoro variant | Release tag                      |
| ---------------- | -------------------------------- |
| `v1.0`           | `model-files-v1.0`               |
| `v1.1-zh`        | `model-files-v1.1`               |
| `v1.2-de-martin` | `model-files-german-martin-v1.2` |

The runtime source of truth is `catalog/models.json`, not the GitHub release catalog. PyKokoro selects one complete distribution and downloads every artifact from that distribution using plain HTTPS with size and SHA-256 validation. GitHub Releases are the preferred provider for v1.0 and v1.1-zh, while pinned direct Hugging Face distributions remain available for upstream-only models such as Russian Zaakirio.
Zaakirio Russian is upstream-only: `ru-zaakirio-base` contains `sveta` and `masha`, while `ru-zaakirio-dima` contains `dima`. Both use the pinned `zaakirio/kokoro-ru` revision and are forbidden from GitHub mirroring.

New planned tags:

- `model-files-vietnamese-v1.0`
- `model-files-vietnamese-anphunl-v1.0`
- `model-files-arabic-nabra-v0.1`
- `model-files-german-kerstin-v1.0`

- `model-files-swedish-v1.0`
- `model-files-german-thorsten-v1.0`
- `model-files-thai-wayu-v1`
- `model-files-kazakh-anuarsv-v1`

Planned first-class `pykokoro` variants:
`ru-zaakirio-base`, `ru-zaakirio-dima`, and `kk-anuarsv`. Route voices as follows:
`sveta` and `masha` to `ru-zaakirio-base`, `dima` to `ru-zaakirio-dima`, and
`km_m1` to `kk-anuarsv`. Russian acoustic model selection must not use only
`language == "ru"`, because the two Russian voices use different checkpoints.
Until this relationship is data-driven from the release catalog or manifest, keep
the explicit voice-to-profile mapping in the integration layer. Kazakh routing
must use canonical language code `kk`, never `ka`.

Swedish and Thorsten use the standard model plus named NPZ voice pattern. Thai explicitly exposes `runtime.layout` as `split-onnx-v1` and must be dispatched to a Wayu split-runtime adapter. `pykokoro` must not infer that layout from filenames: `single-onnx-v1` uses the existing AudioGenerator path, while `split-onnx-v1` uses the Wayu prosody, curves, host alignment/source preparation, and decoder path.
The Nabra release additionally contains `vocab-arabic-nabra-v0.1.json` alongside
the ONNX model and both named NPZ/raw voice assets. Its runtime contract is
`input_ids` (int64), `ref_s` (float32), and `speed` (float32), with a 510-token
maximum per inference call. The active artifact source is the pinned
`marwanelamami/Nabra-82M-v0.1-ONNX` repository; `oddadmix/Nabra-82M-v0.1` is
retained only as base-model provenance.
Hebrew is not enabled for publication by default because its upstream model is
non-commercial/restricted.

Longer term, `pykokoro` should read `release-manifest.json` so filenames,
checksums, voice names, and frontend requirements are data rather than duplicated
constants in the Python package.
