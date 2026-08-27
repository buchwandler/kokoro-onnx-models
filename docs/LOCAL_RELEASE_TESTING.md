# Local release testing

Use `scripts/test_release.py` to run the candidate workflow locally without starting
GitHub Actions or publishing a release.

## Test a mirrored release

```bash
uv run python scripts/test_release.py v1.0
```

The script:

1. Reads the release definition from `catalog/releases.json`.
2. Mirrors GitHub or Hugging Face assets, or builds a local profile.
3. Writes the candidate below `.local-test/release/`.
4. Runs `scripts/verify_candidate.py` with the catalog tag and profile.

A successful run prints the verified candidate path. The default output directory is
ignored by Git. Use another directory with `--dist` when needed:

```bash
uv run python scripts/test_release.py v1.0 --dist /tmp/kokoro-release-test
```

The mirror step reuses assets that already match their configured size and SHA-256
hash, so rerunning the command does not redownload valid files. It does not upload or
publish anything.

## Test a restricted or build profile

Restricted releases require an explicit opt-in, matching the release workflow:

```bash
uv run python scripts/test_release.py he-hebrew-nc --allow-restricted
```

Build profiles use the same command and build into a temporary sibling directory below
the selected distribution directory:

```bash
uv run python scripts/test_release.py sv-joakim
```

## Automated checks

Run the script regression test and the full test suite with:

```bash
uv run pytest -q tests/test_release_test.py
uv run pytest -q
```
