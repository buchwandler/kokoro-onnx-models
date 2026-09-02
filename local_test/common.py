#!/usr/bin/env python3
"""Shared local-only pre-release smoke test runner for pykokoro.

The tracked scripts live in ``local_test/``. Large models, voice archives,
compatibility archives, and WAV output live below ``.local-test/`` and are
intentionally ignored by git.

Two voice-package modes are supported:

* exact: a release voice file is already a NumPy .npz archive (even if its
  extension is .bin), which is what current pykokoro expects;
* compatibility: a headerless raw float32 speaker table is converted to a
  local-only .npz archive before pykokoro is invoked. This proves that the
  acoustic model/voice vectors can be exercised, but it does NOT prove the
  release voice artifact is directly consumable by current pykokoro.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.tokenizer import TokenizerConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.export_validation import (
        validate_waveform_health as _validate_waveform_health,
    )
except ModuleNotFoundError:
    from export_validation import (
        validate_waveform_health as _validate_waveform_health,
    )

LOCAL_ROOT = ROOT / ".local-test"
ASSET_ROOT = LOCAL_ROOT / "assets"
OUTPUT_ROOT = LOCAL_ROOT / "wav"
COMPAT_ROOT = LOCAL_ROOT / "compat"

ModelSource = Literal["github", "huggingface"]

SENTENCES = {
    "en-us": "Hello. This is a local pre-release voice test for the Kokoro model.",
    "en-gb": "Good afternoon. This is a local pre-release voice test for the Kokoro model.",
    "es": "Hola. Esta es una prueba local de voz antes de publicar el modelo.",
    "fr-fr": "Bonjour. Ceci est un test vocal local avant la publication du modèle.",
    "de": "Hallo. Die Brücke ist schön, und dies ist ein lokaler Sprachtest vor der Veröffentlichung des Modells.",
    "it": "Ciao. Questa è una prova vocale locale prima della pubblicazione del modello.",
    "pt": "Olá. Este é um teste local de voz antes da publicação do modelo.",
    "ja": "こんにちは。これはモデル公開前のローカル音声テストです。",
    "zh": "你好。这是模型发布之前的本地语音测试。",
    "hi": "नमस्ते। यह मॉडल जारी करने से पहले एक स्थानीय आवाज़ परीक्षण है।",
    "vi": "Xin chào. Đây là bài kiểm tra giọng nói cục bộ trước khi phát hành mô hình.",
    "ar": "مَرْحَبًا بِكَ. هَذَا اخْتِبَارٌ مَحَلِّيٌّ لِلصَّوْتِ قَبْلَ إِصْدَارِ النَّمُوذَجِ.",
    "sv": "Hej. Det här är ett lokalt rösttest före publiceringen av modellen.",
    "th": "สวัสดี นี่คือการทดสอบเสียงแบบภายในก่อนเผยแพร่โมเดล",
    "he": "שלום. זוהי בדיקת קול מקומית לפני פרסום המודל.",
    "ru": "Здравствуйте! Это локальная проверка русского голоса перед публикацией модели.",
    "kk": "Сәлем! Бұл модельді жарияламас бұрын қазақ тіліндегі жергілікті дауыс сынағы.",
}

VOICE_PREFIX_LANGUAGE = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt",
    "z": "zh",
}


@dataclass(frozen=True)
class LocalTestSpec:
    key: str
    display_name: str
    language: str
    model_source: ModelSource
    model_variant: str
    speed: float = 1.0
    voice_prefix_languages: bool = False
    expected_speakers: tuple[str, ...] = ()
    frontend: str = "pykokoro-native"
    exact_pykokoro_expected: bool = True
    required_files: tuple[str, ...] = ()
    notes: str = ""


SPECS: dict[str, LocalTestSpec] = {
    "v1.0": LocalTestSpec(
        key="v1.0",
        display_name="Kokoro v1.0",
        language="en-us",
        model_source="github",
        model_variant="v1.0",
        voice_prefix_languages=True,
        frontend="pykokoro-native / per-voice language",
    ),
    "v1.1-zh": LocalTestSpec(
        key="v1.1-zh",
        display_name="Kokoro v1.1 Chinese",
        language="zh",
        model_source="github",
        model_variant="v1.1-zh",
        voice_prefix_languages=True,
        frontend="pykokoro-native",
    ),
    "v1.2-de-martin": LocalTestSpec(
        key="v1.2-de-martin",
        display_name="Kokoro German Martin v1.2",
        language="de",
        model_source="github",
        model_variant="v1.2-de-martin",
        speed=1.125,
        expected_speakers=("martin",),
        frontend="pykokoro-native German",
    ),
    "vi-contextbox": LocalTestSpec(
        key="vi-contextbox",
        display_name="Kokoro Vietnamese (ContextBoxAI)",
        language="vi",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=(
            "diem_trinh",
            "hung_thinh",
            "mai_linh",
            "mai_loan",
            "manh_dung",
            "my_yen",
            "ngoc_huyen",
            "phat_tai",
            "thanh_dat",
            "thuc_trinh",
            "tuan_ngoc",
            "storyvert",
            "duc_an",
            "duc_duy",
        ),
        frontend="vig2p required upstream; espeak here is experimental only",
        exact_pykokoro_expected=False,
        notes="Current pykokoro has no Vietnamese model profile or vig2p frontend.",
    ),
    "vi-anphunl": LocalTestSpec(
        key="vi-anphunl",
        display_name="Kokoro Vietnamese (anphunl mirror)",
        language="vi",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=(
            "diem_trinh",
            "hung_thinh",
            "mai_linh",
            "mai_loan",
            "manh_dung",
            "my_yen",
            "ngoc_huyen",
            "phat_tai",
            "thanh_dat",
            "thuc_trinh",
            "tuan_ngoc",
            "storyvert",
            "duc_an",
            "duc_duy",
        ),
        frontend="vig2p required upstream; espeak here is experimental only",
        exact_pykokoro_expected=False,
        notes="Current pykokoro has no Vietnamese model profile or vig2p frontend.",
    ),
    "vi-ngoc-huyen": LocalTestSpec(
        key="vi-ngoc-huyen",
        display_name="Kokoro Vietnamese (Ngọc Huyền)",
        language="vi",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=("ngoc_huyen",),
        frontend="vig2p Vietnamese; current pykokoro frontend is experimental",
        exact_pykokoro_expected=False,
        notes="The release exposes native token durations; use vig2p==0.1.0 for text frontend parity.",
    ),
    "ar-nabra": LocalTestSpec(
        key="ar-nabra",
        display_name="Nabra-82M Arabic",
        language="ar",
        model_source="github",
        model_variant="ar-nabra",
        expected_speakers=("af_msa",),
        frontend="diacritizer + Arabic espeak + Nabra cleanup required upstream",
        exact_pykokoro_expected=True,
        required_files=("vocab.json",),
        notes=(
            "Nabra uses the upstream pre-exported FP32 ONNX model and requires "
            "its dedicated vocab.json before inference. The sample text is already "
            "diacritized; this does not alone establish frontend parity."
        ),
    ),
    "de-crane": LocalTestSpec(
        key="de-crane",
        display_name="Kokoro German Kerstin (Crane)",
        language="de",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=("df_kerstin",),
        frontend="German IPA; training used espeak-ng German IPA",
        exact_pykokoro_expected=False,
        notes=(
            "The upstream voice file is raw float32 and current pykokoro expects "
            "a named npz voice archive. The upstream ONNX also uses input_ids with "
            "float speed, while current pykokoro's GitHub input_ids path uses int32 speed."
        ),
    ),
    "he-hebrew-nc": LocalTestSpec(
        key="he-hebrew-nc",
        display_name="Kokoro Hebrew NC",
        language="he",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=("he_shaul",),
        frontend="Hebrew-specific G2P/config required upstream",
        exact_pykokoro_expected=False,
        notes=(
            "Non-commercial/restricted upstream terms apply. Current pykokoro has "
            "no Hebrew model profile/frontend and cannot consume the custom config directly."
        ),
    ),
    "sv-joakim": LocalTestSpec(
        key="sv-joakim",
        display_name="Kokoro Swedish (Joakim)",
        language="sv",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=(
            "Alice",
            "Anton",
            "Björn",
            "Ebba",
            "Elsa",
            "Greta",
            "Lars",
            "Nils",
            "Oskar",
            "Stina",
        ),
        frontend="kokorog2p Swedish; validate parity with Joakim Swedish frontend",
        exact_pykokoro_expected=False,
        notes="Uses the current pykokoro integration shim until sv-joakim is first-class.",
    ),
    "de-thorsten": LocalTestSpec(
        key="de-thorsten",
        display_name="Kokoro German Thorsten",
        language="de",
        model_source="github",
        model_variant="de-thorsten",
        expected_speakers=("thorsten",),
        required_files=("config.json",),
        frontend="German Kokoro G2P with Thorsten ʏ -> y cleanup",
        exact_pykokoro_expected=False,
        notes="Uses the v1.1.4 exact-iSTFT candidate; listen before publication.",
    ),
    "ru-zaakirio-base": LocalTestSpec(
        key="ru-zaakirio-base",
        display_name="Kokoro Russian (Zaakirio base)",
        language="ru",
        model_source="huggingface",
        model_variant="ru-zaakirio-base",
        expected_speakers=("sveta", "masha"),
        required_files=("config.json",),
        frontend="kokorog2p Russian; preserve upstream stress and orthoepy behavior",
        exact_pykokoro_expected=False,
        notes=(
            "Uses pykokoro's Russian profile with the release-provided config. "
            "The actual ONNX contract is input_ids/style/speed -> waveform."
        ),
    ),
    "ru-zaakirio-dima": LocalTestSpec(
        key="ru-zaakirio-dima",
        display_name="Kokoro Russian (Zaakirio Dima)",
        language="ru",
        model_source="huggingface",
        model_variant="ru-zaakirio-dima",
        expected_speakers=("dima",),
        required_files=("config.json",),
        frontend="kokorog2p Russian; preserve upstream stress and orthoepy behavior",
        exact_pykokoro_expected=False,
        notes=(
            "Dima requires the dedicated model_dima checkpoint. Never test "
            "dima with the base Russian model."
        ),
    ),
    "kk-anuarsv": LocalTestSpec(
        key="kk-anuarsv",
        display_name="Kokoro Kazakh (AnuarSv km_m1)",
        language="kk",
        model_source="github",
        model_variant="v1.0",
        expected_speakers=("km_m1",),
        frontend="kokorog2p Kazakh (kk); parity target is misaki espeak kk",
        exact_pykokoro_expected=False,
        notes=(
            "Uses the v1.0 local integration shim until pykokoro has a "
            "first-class kk-anuarsv profile."
        ),
    ),
}


def validate_local_waveform_health(
    spec_key: str, audio: np.ndarray, sample_rate: int
) -> dict[str, float] | None:
    if spec_key != "de-thorsten":
        return None

    profiles = json.loads(
        (ROOT / "scripts" / "kokoro_profiles.json").read_text(encoding="utf-8")
    )
    validation = profiles[spec_key].get("export_validation") or {}
    metric_kwargs = {
        "min_audio_rms": 0.0,
        "max_audio_abs": float("inf"),
        "max_abs_dc": float("inf"),
        "min_frame_rms_cv": 0.0,
        "max_dc_to_rms_ratio": float("inf"),
        "max_stationary_tone_ratio": 1.0,
    }
    metrics = _validate_waveform_health(audio, sample_rate, **metric_kwargs)
    _validate_waveform_health(
        audio,
        sample_rate,
        min_audio_rms=float(validation.get("min_audio_rms", 0.0005)),
        max_audio_abs=float(validation.get("max_audio_abs", 1.0)),
        max_dc_to_rms_ratio=float(validation.get("max_dc_to_rms_ratio", float("inf"))),
        max_abs_dc=float(validation.get("max_abs_dc", 0.05)),
        min_frame_rms_cv=float(validation.get("min_frame_rms_cv", 0.03)),
        max_stationary_tone_ratio=float(
            validation.get("max_stationary_tone_ratio", 0.35)
        ),
    )
    return metrics


def _verify_prepared_manifest(asset_dir: Path) -> dict[str, object] | None:
    manifest_path = asset_dir / "release-manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("Prepared release manifest has no assets")
    for asset in assets:
        if not isinstance(asset, dict):
            raise TypeError("Prepared release manifest contains an invalid asset")
        name = asset.get("name")
        if not isinstance(name, str):
            raise TypeError("Prepared release manifest has an invalid asset name")
        path = asset_dir / name
        if not path.is_file():
            raise RuntimeError(f"Prepared release is missing asset {name!r}")
        actual_size = path.stat().st_size
        if actual_size != asset.get("size"):
            raise RuntimeError(
                f"Prepared asset {name!r} size mismatch: "
                f"{actual_size} != {asset.get('size')}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != asset.get("sha256"):
            raise RuntimeError(
                f"Prepared asset {name!r} SHA-256 mismatch: "
                f"{digest} != {asset.get('sha256')}"
            )
    return manifest


def _prepared_asset_path(
    asset_dir: Path, manifest: dict[str, object], role: str
) -> Path:
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise TypeError("Prepared release manifest has no asset list")
    matches = [
        asset["name"]
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("role") == role
        and isinstance(asset.get("name"), str)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Prepared release must contain exactly one {role} asset; "
            f"found {len(matches)}"
        )
    return asset_dir / matches[0]


def _parser(spec: LocalTestSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Local pykokoro pre-release smoke test: {spec.display_name}"
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ASSET_ROOT / spec.key,
        help="Directory containing model.onnx and voices.bin/voices.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / spec.key,
        help="WAV output directory (ignored by git)",
    )
    parser.add_argument("--provider", default="cpu", help="pykokoro ONNX provider")
    parser.add_argument("--speed", type=float, default=spec.speed)
    parser.add_argument(
        "--strict-release-format",
        action="store_true",
        help=(
            "Refuse local raw->npz voice conversion. Use this as the real release "
            "compatibility gate for current pykokoro."
        ),
    )
    parser.add_argument(
        "--allow-frontend-mismatch",
        action="store_true",
        help=(
            "For profiles not implemented by pykokoro, allow an experimental "
            "espeak-based text smoke test. Audio generation is not proof of correct G2P."
        ),
    )
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help="Test only this voice; repeat for multiple voices. Default: every voice.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run synthesis without writing WAV files.",
    )
    return parser


def _find_one(asset_dir: Path, patterns: tuple[str, ...], label: str) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(asset_dir.glob(pattern)))
    deduped = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if not deduped:
        raise FileNotFoundError(
            f"No {label} found in {asset_dir}. Expected one of: {', '.join(patterns)}"
        )
    if len(deduped) > 1:
        raise RuntimeError(
            f"Multiple {label} files found in {asset_dir}: "
            + ", ".join(str(p.name) for p in deduped)
        )
    return deduped[0]


def _speaker_names_from_bundle(asset_dir: Path) -> tuple[str, ...]:
    bundle = asset_dir / "bundle.json"
    if not bundle.is_file():
        return ()
    data = json.loads(bundle.read_text(encoding="utf-8"))
    speakers = data.get("speakers") or []
    return tuple(
        str(item["name"])
        for item in speakers
        if isinstance(item, dict) and item.get("name")
    )


def _raw_to_local_npz(
    raw_path: Path,
    *,
    spec: LocalTestSpec,
    asset_dir: Path,
) -> Path:
    names = _speaker_names_from_bundle(asset_dir) or spec.expected_speakers
    if not names:
        raise RuntimeError(
            f"{raw_path} is raw float32 but speaker names are unknown. "
            "Provide bundle.json beside it or add expected_speakers to the local test spec."
        )

    raw = np.fromfile(raw_path, dtype="<f4")
    per_row = len(names) * 256
    if raw.size == 0 or raw.size % per_row:
        raise RuntimeError(
            f"Cannot split {raw_path.name}: {raw.size} float32 values are not divisible "
            f"by {len(names)} speaker(s) × 256."
        )
    rows = raw.size // per_row
    packed = raw.reshape(len(names), rows, 1, 256)

    compat_dir = COMPAT_ROOT / spec.key
    compat_dir.mkdir(parents=True, exist_ok=True)
    target = compat_dir / "voices-pykokoro.npz"
    np.savez(target, **{name: packed[index] for index, name in enumerate(names)})
    return target


def _prepare_voice_archive(
    voices_path: Path,
    *,
    spec: LocalTestSpec,
    asset_dir: Path,
    strict_release_format: bool,
) -> tuple[Path, bool]:
    if zipfile.is_zipfile(voices_path):
        return voices_path, True
    if strict_release_format:
        raise RuntimeError(
            f"{voices_path.name} is not a NumPy npz archive. Current pykokoro "
            "loads .bin voice files with numpy.load(), so this exact release artifact "
            "is not directly compatible."
        )
    compat = _raw_to_local_npz(voices_path, spec=spec, asset_dir=asset_dir)
    print(
        f"WARNING: {voices_path.name} is raw float32. Using local-only compatibility "
        f"archive {compat}. This is NOT an exact release-format pass.",
        file=sys.stderr,
    )
    return compat, False


def _voices_in_archive(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as archive:
        return sorted(archive.files)


def _language_for_voice(spec: LocalTestSpec, voice: str) -> str:
    if not spec.voice_prefix_languages:
        return spec.language
    prefix = voice.split("_", 1)[0]
    if not prefix:
        return spec.language
    return VOICE_PREFIX_LANGUAGE.get(prefix[0].lower(), spec.language)


def _tokenizer_for(
    spec: LocalTestSpec, lang: str, allow_frontend_mismatch: bool
) -> TokenizerConfig:
    native = {
        "en-us",
        "en-gb",
        "es",
        "fr-fr",
        "de",
        "it",
        "pt",
        "ja",
        "zh",
        "ar",
        "ru",
        "kk",
    }
    if lang in native:
        return TokenizerConfig()
    if not allow_frontend_mismatch:
        raise RuntimeError(
            f"{spec.key}: language {lang!r} is not a native current-pykokoro frontend "
            f"for this model. Required frontend: {spec.frontend}. "
            "Pass --allow-frontend-mismatch only for an experimental acoustic smoke test."
        )
    return TokenizerConfig(backend="espeak", load_gold=False, load_silver=False)


def _print_header(spec: LocalTestSpec, model: Path, voices: Path, exact: bool) -> None:
    print("=" * 80)
    print(spec.display_name)
    print(f"model:   {model}")
    print(f"voices:  {voices}")
    print(f"frontend:{spec.frontend}")
    print(
        f"exact voice-package compatibility: {'YES' if exact else 'NO (local conversion)'}"
    )
    if spec.notes:
        print(f"note:    {spec.notes}")
    print("=" * 80)


def run_cli(spec_key: str, argv: list[str] | None = None) -> int:
    spec = SPECS[spec_key]
    args = _parser(spec).parse_args(argv)
    asset_dir = args.asset_dir.resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"Asset directory does not exist: {asset_dir}\n"
            f"Run: python local_test/prepare_local_assets.py {spec.key}"
        )

    prepared_manifest = _verify_prepared_manifest(asset_dir)
    if prepared_manifest is not None:
        print(
            f"verified prepared release manifest: {asset_dir / 'release-manifest.json'}"
        )
    required_paths = {
        filename: asset_dir / filename for filename in spec.required_files
    }
    if prepared_manifest is not None and "config.json" in spec.required_files:
        required_paths["config.json"] = _prepared_asset_path(
            asset_dir, prepared_manifest, "config"
        )
    missing_required = [
        filename
        for filename in spec.required_files
        if not required_paths[filename].is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            f"{spec.key}: missing required runtime asset(s): "
            + ", ".join(missing_required)
        )
    model_path = _find_one(asset_dir, ("model.onnx", "*.onnx"), "ONNX model")
    preferred_voices = asset_dir / "voices.npz"
    if preferred_voices.is_file():
        original_voices = preferred_voices
    else:
        npz_candidates = sorted(asset_dir.glob("*.npz"))
        if len(npz_candidates) == 1:
            original_voices = npz_candidates[0]
        else:
            original_voices = _find_one(
                asset_dir, ("voices.bin", "*.bin"), "voice archive"
            )
    voices_path, exact_voice_format = _prepare_voice_archive(
        original_voices,
        spec=spec,
        asset_dir=asset_dir,
        strict_release_format=args.strict_release_format,
    )
    _print_header(spec, model_path, voices_path, exact_voice_format)

    all_voices = _voices_in_archive(voices_path)
    if spec.expected_speakers:
        missing = sorted(set(spec.expected_speakers) - set(all_voices))
        if missing:
            raise RuntimeError(
                "Voice archive is missing expected speaker(s): " + ", ".join(missing)
            )

    selected = args.voice or all_voices
    unknown = sorted(set(selected) - set(all_voices))
    if unknown:
        raise RuntimeError(
            "Unknown requested voice(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(all_voices)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failures: list[tuple[str, str]] = []

    base_config = PipelineConfig(
        voice=selected[0],
        model_path=model_path,
        voices_path=voices_path,
        model_config_path=(
            asset_dir / "vocab.json"
            if "vocab.json" in spec.required_files
            else (
                required_paths["config.json"]
                if "config.json" in spec.required_files
                else None
            )
        ),
        model_source=spec.model_source,
        model_variant=spec.model_variant,  # type: ignore[arg-type]
        model_quality="fp32",
        provider=args.provider,
        generation=GenerationConfig(
            lang=_language_for_voice(spec, selected[0]), speed=args.speed
        ),
        return_trace=True,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )
    with KokoroPipeline(base_config) as pipeline:
        for index, voice in enumerate(selected, start=1):
            lang = _language_for_voice(spec, voice)
            sentence = SENTENCES[lang]
            try:
                tokenizer_config = _tokenizer_for(
                    spec, lang, args.allow_frontend_mismatch
                )
                generation = GenerationConfig(lang=lang, speed=args.speed)
                print(f"[{index}/{len(selected)}] {voice} ({lang}): {sentence}")
                result = pipeline.run(
                    sentence,
                    voice=voice,
                    generation=generation,
                    tokenizer_config=tokenizer_config,
                )
                if result.audio.size == 0:
                    raise RuntimeError("synthesis returned an empty waveform")
                if not np.isfinite(result.audio).all():
                    raise RuntimeError("synthesis returned NaN/Inf samples")
                health_error: Exception | None = None
                if spec.key == "de-thorsten":
                    try:
                        metrics = _validate_waveform_health(
                            result.audio,
                            result.sample_rate,
                            min_audio_rms=0.0,
                            max_audio_abs=float("inf"),
                            max_abs_dc=float("inf"),
                            min_frame_rms_cv=0.0,
                            max_stationary_tone_ratio=1.0,
                        )
                        for name in (
                            "peak",
                            "rms",
                            "dc",
                            "dc_to_rms_ratio",
                            "frame_rms_cv",
                            "stationary_tone_ratio",
                        ):
                            print(f"    {name}: {metrics[name]:.6f}")
                        validate_local_waveform_health(
                            spec.key, result.audio, result.sample_rate
                        )
                    except ValueError as exc:
                        health_error = exc
                        print(f"    waveform health FAILED: {exc}")

                duration = len(result.audio) / result.sample_rate
                print(
                    f"    ok: {len(result.audio)} samples, "
                    f"{result.sample_rate} Hz, {duration:.2f} s"
                )
                if result.phoneme_segments:
                    phonemes = " ".join(seg.phonemes for seg in result.phoneme_segments)
                    print(f"    phonemes: {phonemes}")
                if not args.no_write:
                    output = args.output_dir / f"{voice}.wav"
                    sf.write(output, result.audio, result.sample_rate)
                    print(f"    wrote: {output}")
                if health_error is not None:
                    raise RuntimeError(str(health_error))
                success += 1
            except Exception as exc:  # noqa: BLE001
                failures.append((voice, f"{type(exc).__name__}: {exc}"))
                print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("-" * 80)
    print(f"voices passed: {success}/{len(selected)}")
    if failures:
        for voice, message in failures:
            print(f"  FAIL {voice}: {message}", file=sys.stderr)
        return 1
    if args.strict_release_format and not spec.exact_pykokoro_expected:
        print(
            "NOTE: this profile is not yet declared first-class by current pykokoro; "
            "review the coding-agent guide before release.",
            file=sys.stderr,
        )
    return 0
