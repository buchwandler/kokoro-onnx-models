from __future__ import annotations

import importlib

import pytest


verify = importlib.import_module("scripts.verify_published_releases")

TAG = "model-files-test-v1"
NAME = "model.onnx"
URL = f"https://github.com/buchwandler/kokoro-onnx-models/releases/download/{TAG}/{NAME}"


class FakeClient:
    def __init__(self, release):
        self.release = release
        self.tags: list[str] = []

    def get_release(self, tag: str):
        self.tags.append(tag)
        return self.release


def distribution(*, provider="github-release", runtime_ready=True, url=URL):
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
                "size": 12,
                "sha256": "a" * 64,
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
        verify.verify_distribution("test", distribution(), FakeClient(release(assets=[])))


def test_wrong_asset_size_fails() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="size is 13"):
        verify.verify_distribution(
            "test", distribution(), FakeClient(release(assets=[{"name": NAME, "size": 13}]))
        )


def test_draft_release_fails() -> None:
    with pytest.raises(verify.PublicationVerificationError, match="is a draft"):
        verify.verify_distribution("test", distribution(), FakeClient(release(draft=True)))


def test_url_tag_mismatch_fails() -> None:
    wrong_url = URL.replace(TAG, "model-files-other-v1")
    with pytest.raises(verify.PublicationVerificationError, match="does not match release tag"):
        verify.verify_distribution("test", distribution(url=wrong_url), FakeClient(release()))


def test_url_basename_mismatch_fails() -> None:
    wrong_url = URL.replace(NAME, "other.onnx")
    with pytest.raises(verify.PublicationVerificationError, match="does not match release tag"):
        verify.verify_distribution("test", distribution(url=wrong_url), FakeClient(release()))


def test_non_github_distribution_is_ignored() -> None:
    registry = {"models": {"test": {"distributions": [distribution(provider="huggingface")]}}}

    assert verify.verify_publications(registry, FakeClient(None)) == 0


def test_non_runtime_ready_distribution_is_ignored() -> None:
    registry = {"models": {"test": {"distributions": [distribution(runtime_ready=False)]}}}

    assert verify.verify_publications(registry, FakeClient(None)) == 0


def test_github_client_returns_none_for_404() -> None:
    class NotFound:
        code = 404

    def opener(*args, **kwargs):
        raise verify.HTTPError("https://api.github.com", 404, "not found", {}, None)

    client = verify.GitHubReleaseClient("owner/repo", opener=opener)

    assert client.get_release(TAG) is None
