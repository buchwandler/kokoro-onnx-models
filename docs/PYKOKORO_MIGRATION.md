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

Hebrew is not enabled for publication by default because its upstream model is
non-commercial/restricted.

Longer term, `pykokoro` should read `release-manifest.json` so filenames,
checksums, voice names, and frontend requirements are data rather than duplicated
constants in the Python package.
