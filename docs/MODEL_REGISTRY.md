# Runtime model registry

`catalog/models.json` is the canonical inventory of runtime-ready Kokoro model profiles. Each profile identifies its language and machine frontend, runtime layout, voices, ONNX contract, license provenance, and one or more complete distributions.

## Atomic distributions

Select one distribution before downloading. A distribution contains the compatible model, voice, config, vocabulary, and split-graph components. Do not combine artifacts from different distributions.

Every artifact has an immutable HTTPS URL, local filename, format, byte size, and SHA-256 digest. Hugging Face URLs use a pinned commit rather than `main`, and registry downloads use ordinary HTTPS without `huggingface_hub`.

## Mirrors and upstream sources

`mirror_policy` describes redistribution separately from runtime availability:

- `required`: this repository must build the runtime distribution.
- `preferred`: a GitHub mirror is normally provided.
- `optional`: either provider is acceptable.
- `forbidden`: runtime bytes must remain upstream-only.

`catalog/releases.json` describes GitHub publication jobs. `scripts/kokoro_profiles.json` describes build and repack recipes, including checkpoint-only sources that are not direct client distributions.

Russian Zaakirio is an upstream-only example. The base profile uses `sveta` and `masha` with `onnx/model.onnx`; the Dima profile uses `dima` with `onnx/model_dima.onnx`. Both use revision `d649c57b239b18c4c384378127cbf01dba039bc1` and raw float32 voice tables.

## Tools

Check committed metadata without downloading large files:

```bash
python scripts/collect_runtime_metadata.py --check
python scripts/verify_model_registry.py
```

Collect metadata from an exact upstream distribution when intentionally updating it:

```bash
python scripts/collect_runtime_metadata.py ru-zaakirio-base --update
```

The collector refuses changed bytes for already populated metadata and never changes a pinned revision.
