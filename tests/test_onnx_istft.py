from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.onnx_istft import ExactOnnxISTFT


@pytest.mark.parametrize(
    ("n_fft", "hop", "win_length"),
    [(20, 5, 20), (512, 128, 512), (800, 200, 800), (1024, 256, 1024)],
)
@pytest.mark.parametrize("frames", [10, 40, 73])
def test_exact_onnx_istft_matches_torch_istft(
    n_fft: int, hop: int, win_length: int, frames: int
) -> None:
    torch.manual_seed(7)
    magnitude = torch.rand(2, n_fft // 2 + 1, frames) + 0.1
    phase = (torch.rand_like(magnitude) - 0.5) * 2.0 * torch.pi
    window = torch.hann_window(win_length, periodic=True)
    expected = torch.istft(
        torch.polar(magnitude, phase),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_length,
        window=window,
        center=True,
    )

    actual = ExactOnnxISTFT(
        filter_length=n_fft,
        hop_length=hop,
        win_length=win_length,
    )(magnitude, phase).squeeze(1)

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_exact_onnx_istft_applies_one_sided_interior_bin_scaling() -> None:
    n_fft, hop, frames = 20, 5, 20
    magnitude = torch.zeros(1, n_fft // 2 + 1, frames)
    magnitude[:, 1, :] = 1.0
    phase = torch.zeros_like(magnitude)
    expected = torch.istft(
        torch.polar(magnitude, phase),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        center=True,
    )
    actual = ExactOnnxISTFT(filter_length=n_fft, hop_length=hop, win_length=n_fft)(
        magnitude, phase
    ).squeeze(1)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_exact_onnx_istft_exports_dynamic_frames_to_onnx(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    module = ExactOnnxISTFT(filter_length=20, hop_length=5, win_length=20).eval()
    magnitude = torch.rand(1, 11, 10) + 0.1
    phase = torch.rand_like(magnitude)
    path = tmp_path / "istft.onnx"
    torch.onnx.export(
        module,
        (magnitude, phase),
        path,
        input_names=["magnitude", "phase"],
        output_names=["audio"],
        dynamic_axes={
            "magnitude": {2: "frames"},
            "phase": {2: "frames"},
            "audio": {2: "samples"},
        },
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for frames in (10, 23):
        magnitude_np = (torch.rand(1, 11, frames) + 0.1).numpy()
        phase_np = torch.rand(1, 11, frames).numpy()
        actual = session.run(["audio"], {"magnitude": magnitude_np, "phase": phase_np})[
            0
        ]
        expected = torch.istft(
            torch.polar(torch.from_numpy(magnitude_np), torch.from_numpy(phase_np)),
            n_fft=20,
            hop_length=5,
            win_length=20,
            window=torch.hann_window(20),
            center=True,
        ).numpy()[:, None, :]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_exact_onnx_istft_uses_registered_kernels() -> None:
    module = ExactOnnxISTFT(filter_length=20, hop_length=5, win_length=20)
    names = dict(module.named_buffers())
    assert set(names) == {
        "weight_backward_real",
        "weight_backward_imag",
        "weight_forward_real",
        "weight_forward_imag",
        "window_envelope_kernel",
    }
