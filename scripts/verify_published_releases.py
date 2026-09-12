#!/usr/bin/env python3
"""Verify that runtime-ready GitHub distributions are publicly published."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "catalog" / "models.json"
RELEASES = ROOT / "catalog" / "releases.json"
DEFAULT_REPOSITORY = "buchwandler/kokoro-onnx-models"


class PublicationVerificationError(ValueError):
    """Raised when a runtime-ready GitHub distribution is not published correctly."""


class GitHubReleaseClient:
    """Small GitHub Releases API client with an injectable opener for tests."""

    def __init__(
        self,
        repository: str,
        *,
        token: str | None = None,
        opener: Callable[..., Any] = urlopen,
        timeout: int = 30,
    ) -> None:
        self.repository = repository
        self.token = token
        self.opener = opener
        self.timeout = timeout

    def get_release(self, tag: str) -> dict[str, Any] | None:
        url = (
            f"https://api.github.com/repos/{self.repository}/releases/tags/"
            f"{quote(tag, safe='')}"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "kokoro-onnx-models-publication-check",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, headers=headers)
        try:
            with self.opener(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise PublicationVerificationError(
                f"GitHub API request failed for release {tag!r}: HTTP {exc.code}"
            ) from exc
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicationVerificationError(
                f"GitHub API request failed for release {tag!r}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise PublicationVerificationError(
                f"GitHub API returned a non-object for release {tag!r}"
            )
        return value


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationVerificationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationVerificationError(f"{path} must contain an object")
    return value


def _validate_artifact_url(
    model_id: str, release_tag: str, artifact: dict[str, Any]
) -> None:
    url = str(artifact.get("url", ""))
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    expected_name = str(artifact.get("local_name", ""))
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or len(parts) < 5
        or parts[-3] != "download"
        or parts[-2] != release_tag
        or parts[-1] != expected_name
    ):
        raise PublicationVerificationError(
            f"{model_id}/{artifact.get('id')}: URL does not match release tag "
            f"{release_tag!r} and asset {expected_name!r}: {url}"
        )


def _asset_index(release: dict[str, Any], model_id: str, tag: str) -> dict[str, dict[str, Any]]:
    if release.get("tag_name") != tag:
        raise PublicationVerificationError(
            f"{model_id}: GitHub release tag is {release.get('tag_name')!r}, expected {tag!r}"
        )
    if release.get("draft") is True:
        raise PublicationVerificationError(f"{model_id}: GitHub release {tag!r} is a draft")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise PublicationVerificationError(f"{model_id}: GitHub release {tag!r} has no asset list")
    result: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if isinstance(asset, dict) and isinstance(asset.get("name"), str):
            result[asset["name"]] = asset
    return result


def verify_distribution(
    model_id: str,
    distribution: dict[str, Any],
    client: GitHubReleaseClient,
    check_digests: bool = False,
) -> None:
    tag = distribution.get("release_tag")
    if not isinstance(tag, str) or not tag:
        raise PublicationVerificationError(f"{model_id}: GitHub distribution has no release_tag")
    for artifact in distribution.get("artifacts", []):
        _validate_artifact_url(model_id, tag, artifact)
    release = client.get_release(tag)
    if release is None:
        raise PublicationVerificationError(
            f"{model_id}: GitHub release {tag!r} does not exist"
        )
    assets = _asset_index(release, model_id, tag)
    for artifact in distribution.get("artifacts", []):
        name = str(artifact["local_name"])
        asset = assets.get(name)
        if asset is None:
            raise PublicationVerificationError(
                f"{model_id}/{artifact['id']}: release {tag!r} is missing asset {name!r}"
            )
        expected_size = artifact.get("size")
        actual_size = asset.get("size")
        if actual_size != expected_size:
            raise PublicationVerificationError(
                f"{model_id}/{artifact['id']}: asset {name!r} size is {actual_size}, "
                f"expected {expected_size}"
            )
        digest = asset.get("digest")
        expected_sha256 = artifact.get("sha256")
        if check_digests and (
            isinstance(digest, str)
            and digest.startswith("sha256:")
            and expected_sha256
        ):
            if digest.removeprefix("sha256:") != expected_sha256:
                raise PublicationVerificationError(
                    f"{model_id}/{artifact['id']}: asset {name!r} SHA-256 digest differs"
                )


def verify_publications(
    registry: dict[str, Any],
    client: GitHubReleaseClient,
    *,
    profile: str | None = None,
    check_digests: bool = False,
) -> int:
    models = registry.get("models")
    if not isinstance(models, dict):
        raise PublicationVerificationError("Registry must contain a models object")
    if profile is not None and profile not in models:
        raise PublicationVerificationError(f"Unknown model profile: {profile}")
    selected = [profile] if profile else list(models)
    checked = 0
    for model_id in selected:
        model = models[model_id]
        for distribution in model.get("distributions", []):
            if (
                distribution.get("provider") != "github-release"
                or distribution.get("runtime_ready") is not True
            ):
                continue
            verify_distribution(model_id, distribution, client, check_digests=check_digests)
            checked += 1
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="check only one model profile")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument(
        "--check-digests",
        action="store_true",
        help="compare GitHub-provided SHA-256 digests when available",
    )
    args = parser.parse_args(argv)
    try:
        registry = _load(args.registry)
        client = GitHubReleaseClient(args.repository, token=args.token)
        checked = verify_publications(
            registry, client, profile=args.profile, check_digests=args.check_digests
        )
    except PublicationVerificationError as exc:
        print(f"published release verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {checked} runtime-ready GitHub distributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
