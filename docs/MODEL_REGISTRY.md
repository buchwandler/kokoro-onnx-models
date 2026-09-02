# Runtime model registry

`catalog/models.json` is the canonical inventory of runtime-ready Kokoro model profiles. Each profile identifies its language and machine frontend, runtime layout, voices, ONNX contract, license provenance, and one or more complete distributions.

## Atomic distributions

Select one distribution before downloading. A distribution contains the compatible model, voice, config, vocabulary, and split-graph components. Do not combine artifacts from different distributions.

Every artifact has an immutable HTTPS URL, local filename, format, byte size, and SHA-256 digest. Hugging Face URLs use a pinned commit rather than `main`, and registry downloads use ordinary HTTPS without `huggingface_hub`.

## Immutable release tags

A release tag is a content identity. Once published, the artifact names, URLs, formats, sizes, SHA-256 values, handling metadata, and bytes under that tag are immutable. If model bytes or any artifact identity changes, publish a new release tag and update the registry to the new distribution instead of rewriting metadata under the old tag.

`scripts/update_registry_from_release.py --update` enforces this contract and rejects changed artifact identity for an existing release tag. The checks workflow also compares the catalog with the pull-request base or push parent to reject manual same-tag edits.

## Mirrors and upstream sources

`mirror_policy` describes redistribution separately from runtime availability:

- `required`: this repository must build the runtime distribution.
- `preferred`: a GitHub mirror is normally provided.
- `optional`: either provider is acceptable.
- `forbidden`: runtime bytes must remain upstream-only.

`catalog/releases.json` describes GitHub publication jobs. `scripts/kokoro_profiles.json` describes build and repack recipes, including checkpoint-only sources that are not direct client distributions.

Russian Zaakirio uses two checkpoint-built GitHub distributions. The base profile builds `kokoro-ru-v2-base.pth` for `sveta` and `masha`; the Dima profile builds `kokoro-ru-v2-dima.pth` for `dima`. Both use revision `d649c57b239b18c4c384378127cbf01dba039bc1`, named NumPy voice archives, raw float32 voice tables, and validated token durations.

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
