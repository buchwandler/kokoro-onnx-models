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

New planned tags:

- `model-files-vietnamese-v1.0`
- `model-files-vietnamese-anphunl-v1.0`
- `model-files-arabic-nabra-v0.1`
- `model-files-german-kerstin-v1.0`

- `model-files-swedish-v1.0`
- `model-files-german-thorsten-v1.0`
- `model-files-thai-wayu-v1`

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
