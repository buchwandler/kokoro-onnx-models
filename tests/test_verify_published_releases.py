from __future__ import annotations

import importlib

import pytest

verify = importlib.import_module("scripts.verify_published_releases")

TAG = "model-files-test-v1"
NAME = "model.onnx"
URL = (
    f"https://github.com/buchwandler/kokoro-onnx-models/releases/download/{TAG}/{NAME}"
)


class FakeClient:
    def __init__(self, release):
        self.release = release
        self.tags: list[str] = []

    def get_release(self, tag: str):
        self.tags.append(tag)
        return self.release


def distribution(
    *, provider="github-release", runtime_ready=True, url=URL, size=12, sha256="a" * 64
):
    return {
        "id": "github-test",
        "provider": provider,
        "release_tag": TAG,
        "runtime_ready": runtime_ready,
        "artifacts": [
            {
                "id": "model-model",
                "url": url,
                "local_name": NAME,
                "size": size,
                "sha256": sha256,
            }
        ],
    }


def release(*, draft=False, assets=None):
    return {
        "tag_name": TAG,
        "draft": draft,
        "assets": assets if assets is not None else [{"name": NAME, "size": 12}],
    }


def test_valid_release_and_assets_pass() -> None:
    client = FakeClient(release())

    verify.verify_distribution("test", distribution(), client)

    assert client.tags == [TAG]


def test_missing_release_fails() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="does not exist"):
        verify.verify_distribution("test", distribution(), FakeClient(None))


def test_missing_asset_fails() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="missing asset"):
        verify.verify_distribution(
            "test", distribution(), FakeClient(release(assets=[]))
        )


def test_wrong_asset_size_fails() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="size is 13"):
        verify.verify_distribution(
            "test",
            distribution(),
            FakeClient(release(assets=[{"name": NAME, "size": 13}])),
        )


def test_wrong_digest_fails_when_digest_check_enabled() -> None:
    published = release(
        assets=[{"name": NAME, "size": 12, "digest": "sha256:" + "b" * 64}]
    )
    with pytest.raises(verify.PublicationVerificationError, match="digest differs"):
        verify.verify_distribution(
            "test", distribution(), FakeClient(published), check_digests=True
        )


def test_missing_digest_fails_when_digest_check_enabled() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="no SHA-256 digest"):
        verify.verify_distribution(
            "test", distribution(), FakeClient(release()), check_digests=True
        )


def test_digest_is_not_required_without_digest_check() -> None:
    verify.verify_distribution("test", distribution(), FakeClient(release()))


def test_swedish_bundle_identity_mismatch_fails() -> None:
    published_tag = "model-files-swedish-v1.1"
    expected_size = 70395
    expected_sha = "153a68523a8e5f2c01843d2fd2d8a40f2b94288b167bd595865762253ca610c3"
    actual_size = 70344
    actual_sha = "a2069566ff1933263f03a24aadb68e6bf247562293633ca5998867d01543dd39"
    swedish = distribution(
        url=f"https://github.com/buchwandler/kokoro-onnx-models/releases/download/{published_tag}/bundle.json",
        size=expected_size,
        sha256=expected_sha,
    )
    swedish["release_tag"] = published_tag
    swedish["artifacts"][0].update({"id": "bundle-bundle", "local_name": "bundle.json"})
    published = release(
        assets=[
            {
                "name": "bundle.json",
                "size": actual_size,
                "digest": "sha256:" + actual_sha,
            }
        ]
    )
    published["tag_name"] = published_tag

    with pytest.raises(verify.PublicationVerificationError, match="size is 70344"):
        verify.verify_distribution(
            "sv-joakim", swedish, FakeClient(published), check_digests=True
        )


def test_draft_release_fails() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="is a draft"):
        verify.verify_distribution(
            "test", distribution(), FakeClient(release(draft=True))
        )


def test_url_tag_mismatch_fails() -> None:
    wrong_url = URL.replace(TAG, "model-files-other-v1")
    with pytest.raises(
        verify.PublicationVerificationError, match="does not match release tag"
    ):
        verify.verify_distribution(
            "test", distribution(url=wrong_url), FakeClient(release())
        )


def test_url_basename_mismatch_fails() -> None:
    wrong_url = URL.replace(NAME, "other.onnx")
    with pytest.raises(
        verify.PublicationVerificationError, match="does not match release tag"
    ):
        verify.verify_distribution(
            "test", distribution(url=wrong_url), FakeClient(release())
        )


def test_non_github_distribution_is_ignored() -> None:
    registry = {
        "models": {"test": {"distributions": [distribution(provider="huggingface")]}}
    }

    assert verify.verify_publications(registry, FakeClient(None)) == 0


def test_non_runtime_ready_distribution_is_ignored() -> None:
    registry = {
        "models": {"test": {"distributions": [distribution(runtime_ready=False)]}}
    }

    assert verify.verify_publications(registry, FakeClient(None)) == 0


def test_github_client_returns_none_for_404() -> None:
    class NotFound:
        code = 404

    def opener(*args, **kwargs):
        raise verify.HTTPError("https://api.github.com", 404, "not found", {}, None)

    client = verify.GitHubReleaseClient("owner/repo", opener=opener)

    assert client.get_release(TAG) is None
