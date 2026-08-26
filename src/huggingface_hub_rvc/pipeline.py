from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import HfApi, snapshot_download

from huggingface_hub_rvc._runtime import (
    DEFAULT_ASSET_REPO,
    FEATURES_NPY_NAME,
    FEATURES_SAFETENSORS_NAME,
    INDEX_NAME,
    MODEL_PTH_NAME,
    MODEL_SAFETENSORS_NAME,
    SimpleRVCArtifact,
    SimpleRVCConverter,
    SimpleRVCTrainer,
    _load_model_payload,
    export_webui_artifact,
)

CONFIG_NAME = "config.json"
MANIFEST_NAME = "manifest.json"
MODEL_CARD_NAME = "README.md"
VOICE_TRANSFORM_DIR = "voice_transform"
FORMAT_VERSION = 1
DEFAULT_MODEL_TYPE = "rvc"
DEFAULT_MODEL_NAME = "RVC Voice Model"

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class RVCConfig:
    model_name: str = DEFAULT_MODEL_NAME
    model_type: str = DEFAULT_MODEL_TYPE
    format_version: int = FORMAT_VERSION
    architecture: str = "rvc-v2-f0-48k"
    sample_rate: int = 48000
    f0: bool = True
    model_file: str = MODEL_SAFETENSORS_NAME
    index_file: str | None = INDEX_NAME
    features_file: str | None = FEATURES_SAFETENSORS_NAME
    asset_hub_model_id: str = DEFAULT_ASSET_REPO
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RVCConfig":
        components = data.get("components") or {}
        metadata = dict(data.get("metadata") or {})
        return cls(
            model_name=_coerce_model_name(data.get("model_name", data.get("name", metadata.get("model_name")))) or DEFAULT_MODEL_NAME,
            model_type=str(data.get("model_type", data.get("arch_type", DEFAULT_MODEL_TYPE))),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
            architecture=str(data.get("architecture", data.get("arch_version", "rvc-v2-f0-48k"))),
            sample_rate=int(data.get("sample_rate", 48000)),
            f0=bool(data.get("f0", True)),
            model_file=str(data.get("model_file", components.get("model", components.get("pth", MODEL_SAFETENSORS_NAME)))),
            index_file=data.get("index_file", components.get("index", INDEX_NAME)),
            features_file=data.get("features_file", components.get("features", FEATURES_SAFETENSORS_NAME)),
            asset_hub_model_id=str(data.get("asset_hub_model_id", DEFAULT_ASSET_REPO)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "format_version": self.format_version,
            "architecture": self.architecture,
            "sample_rate": self.sample_rate,
            "f0": self.f0,
            "model_file": self.model_file,
            "index_file": self.index_file,
            "features_file": self.features_file,
            "asset_hub_model_id": self.asset_hub_model_id,
            "components": {
                key: value
                for key, value in {
                    "model": self.model_file,
                    "index": self.index_file,
                    "features": self.features_file,
                }.items()
                if value is not None
            },
            "metadata": self.metadata,
        }


class _ProgressLogger:
    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self.callback = callback

    def event(self, transform_id: str, event: str, **payload: Any) -> None:
        if self.callback is not None:
            self.callback({"transform_id": transform_id, "event": event, **payload})


class RVCPipeline:
    config: RVCConfig
    artifact: SimpleRVCArtifact

    def __init__(
        self,
        artifact: SimpleRVCArtifact,
        config: RVCConfig | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.artifact = artifact
        self.config = config or _config_from_manifest(artifact.manifest)
        self.model_config = {
            "model_name": self.config.model_name,
            "asset_hub_model_id": self.config.asset_hub_model_id,
            "sample_rate": self.config.sample_rate,
            **(model_config or {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        subfolder: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> "RVCPipeline":
        source = Path(pretrained_model_name_or_path).expanduser()
        model_path_override = None
        if source.exists():
            if source.is_file():
                if source.suffix.lower() not in {".pth", ".safetensors"}:
                    raise ValueError("Local RVC model files must use .pth or .safetensors.")
                model_path_override = source
                root = source.parent
            else:
                root = source / subfolder if subfolder else source
        else:
            root = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token=token,
                    allow_patterns=[
                        CONFIG_NAME,
                        f"{VOICE_TRANSFORM_DIR}/*",
                        "*.safetensors",
                        "*.pth",
                        "*.index",
                        "*.npy",
                        "*.json",
                    ],
                )
            )
            if subfolder:
                root = root / subfolder
        artifact, config = _load_artifact(
            root,
            model_name_fallback=source.stem if model_path_override is not None else _name_from_repo_id(pretrained_model_name_or_path),
            model_path_override=model_path_override,
        )
        return cls(artifact=artifact, config=config, model_config=model_config)

    @classmethod
    def train(
        cls,
        identity_dir: str | Path,
        output_dir: str | Path,
        *,
        training_steps: int = 1000,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        max_seconds_per_file: float = 180.0,
        identity_audio_mode: str = "separate",
        device: str | None = None,
        demucs_device: str | None = None,
        asset_hub_model_id: str = DEFAULT_ASSET_REPO,
        model_name: str | None = None,
        build_index: bool = True,
        progress_callback: ProgressCallback | None = None,
        transform_id: str = "rvc-train",
        **model_config: Any,
    ) -> "RVCPipeline":
        output_root = Path(output_dir).expanduser()
        cache_dir = output_root / VOICE_TRANSFORM_DIR
        resolved_model_name = model_name or output_root.name or DEFAULT_MODEL_NAME
        transform_config = {
            "id": transform_id,
            "method": "rvc",
            "model": {
                "model_name": resolved_model_name,
                "identity_data_dir": str(Path(identity_dir).expanduser()),
                "cache_dir": str(cache_dir),
                "asset_hub_model_id": asset_hub_model_id,
                "sample_rate": 48000,
                "identity_audio_mode": identity_audio_mode,
                "training_steps": training_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_seconds_per_file": max_seconds_per_file,
                "build_index": build_index,
                **model_config,
            },
        }
        if device is not None:
            transform_config["model"]["device"] = device
        if demucs_device is not None:
            transform_config["model"]["demucs_device"] = demucs_device
        manifest_base = _manifest_base("rvc-train", transform_id, model_name=resolved_model_name)
        artifact = SimpleRVCTrainer().train(
            source_backend_config={"id": "identity", "dataset_type": "audio"},
            transform_config=transform_config,
            cache_dir=cache_dir,
            fingerprint="",
            manifest_base=manifest_base,
            run_logger=_ProgressLogger(progress_callback),
        )
        pipeline = cls(
            artifact=artifact,
            config=_config_from_manifest(artifact.manifest),
            model_config=transform_config["model"],
        )
        pipeline.save_pretrained(output_root)
        return pipeline

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        model_name: str | None = None,
        push_to_hub: bool = False,
        repo_id: str | None = None,
        token: bool | str | None = None,
        private: bool | None = None,
        commit_message: str = "Upload RVC pipeline",
    ) -> Path:
        save_root = Path(save_directory).expanduser()
        voice_root = save_root / VOICE_TRANSFORM_DIR
        voice_root.mkdir(parents=True, exist_ok=True)
        config = self._resolved_config()
        config.model_name = (
            model_name
            or (config.model_name if config.model_name != DEFAULT_MODEL_NAME else None)
            or _name_from_repo_id(repo_id)
            or save_root.name
            or DEFAULT_MODEL_NAME
        )
        _copy_if_exists(self.artifact.model_path, voice_root / config.model_file)
        if self.artifact.index_path and config.index_file:
            _copy_if_exists(self.artifact.index_path, voice_root / config.index_file)
        for feature_name in (FEATURES_SAFETENSORS_NAME, FEATURES_NPY_NAME):
            feature_path = self.artifact.cache_dir / feature_name
            if feature_path.exists():
                _copy_if_exists(feature_path, voice_root / feature_name)
        manifest = dict(self.artifact.manifest)
        manifest.setdefault("voice_model", {})
        manifest["voice_model"].update(
            {
                "model_file": config.model_file,
                "index_file": config.index_file,
                "features_file": config.features_file,
                "model_name": config.model_name,
            }
        )
        if config.metadata.get("speaker_info"):
            manifest["voice_model"]["speaker_info"] = config.metadata["speaker_info"]
        manifest["model_name"] = config.model_name
        (voice_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (save_root / CONFIG_NAME).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (save_root / MODEL_CARD_NAME).write_text(_model_card_text(config, manifest), encoding="utf-8")
        if push_to_hub:
            if repo_id is None:
                raise ValueError("repo_id is required when push_to_hub=True.")
            self.push_to_hub(repo_id=repo_id, folder_path=save_root, token=token, private=private, commit_message=commit_message)
        return save_root

    def push_to_hub(
        self,
        repo_id: str,
        *,
        folder_path: str | Path | None = None,
        token: bool | str | None = None,
        private: bool | None = None,
        commit_message: str = "Upload RVC pipeline",
    ) -> Any:
        folder = Path(folder_path).expanduser() if folder_path is not None else self.save_pretrained(Path.cwd() / "rvc-pipeline")
        if not (folder / MODEL_CARD_NAME).exists():
            folder = self.save_pretrained(folder)
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        return api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message=commit_message,
        )

    def export_webui(
        self,
        save_directory: str | Path,
        *,
        model_name: str | None = None,
        training_steps: int | None = None,
    ) -> Path:
        return export_webui_artifact(
            self.artifact,
            Path(save_directory).expanduser(),
            model_name=model_name or self.config.model_name,
            training_steps=training_steps,
        )

    def convert_directory(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        *,
        audio_mode: str = "separate_convert_remix",
        device: str | None = None,
        demucs_device: str | None = None,
        retrieval_strength: float = 0.75,
        pitch_shift: float = 0.0,
        f0_method: str = "rmvpe",
        protect: float = 0.33,
        rms_mix_rate: float = 1.0,
        speaker_id: int = 0,
        output_sample_rate: int | None = None,
        separation_method: str = "pymss",
        use_cuda_graph: bool = False,
        torch_retrieval: bool = True,
        progress_callback: ProgressCallback | None = None,
        **conversion_config: Any,
    ) -> list[Path]:
        source_root = Path(source_dir).expanduser()
        output_root = Path(output_dir).expanduser()
        input_paths = sorted(path for path in source_root.rglob("*") if path.suffix.lower() in {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac", ".opus"})
        if not input_paths:
            raise ValueError(f"No audio files found under {source_root}.")
        self._convert_paths(
            source_root=source_root,
            output_root=output_root,
            input_paths=input_paths,
            audio_mode=audio_mode,
            device=device,
            demucs_device=demucs_device,
            retrieval_strength=retrieval_strength,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            speaker_id=speaker_id,
            output_sample_rate=output_sample_rate,
            separation_method=separation_method,
            use_cuda_graph=use_cuda_graph,
            torch_retrieval=torch_retrieval,
            progress_callback=progress_callback,
            conversion_config=conversion_config,
        )
        return sorted(output_root.rglob("*.wav"))

    def convert_file(self, source: str | Path, output: str | Path, **kwargs: Any) -> Path:
        source_path = Path(source).expanduser()
        output_path = Path(output).expanduser()
        with _SingleFileOutput(output_path) as output_root:
            self._convert_paths(
                source_root=source_path.parent,
                output_root=output_root,
                input_paths=[source_path],
                **kwargs,
            )
            generated = output_root / source_path.with_suffix(".wav").name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated), output_path)
        return output_path

    def prepare_datasets(self, source_dir: str | Path, generated_dir: str | Path, **kwargs: Any) -> list[Path]:
        return self.convert_directory(source_dir, generated_dir, **kwargs)

    def _convert_paths(
        self,
        *,
        source_root: Path,
        output_root: Path,
        input_paths: Iterable[Path],
        audio_mode: str = "separate_convert_remix",
        device: str | None = None,
        demucs_device: str | None = None,
        retrieval_strength: float = 0.75,
        pitch_shift: float = 0.0,
        f0_method: str = "rmvpe",
        protect: float = 0.33,
        rms_mix_rate: float = 1.0,
        speaker_id: int = 0,
        output_sample_rate: int | None = None,
        separation_method: str = "pymss",
        use_cuda_graph: bool = False,
        torch_retrieval: bool = True,
        progress_callback: ProgressCallback | None = None,
        conversion_config: dict[str, Any] | None = None,
        **extra_conversion_config: Any,
    ) -> None:
        conversion = {
            "audio_mode": audio_mode,
            "separation_method": separation_method,
            "retrieval_strength": retrieval_strength,
            "pitch_shift": pitch_shift,
            "f0_method": f0_method,
            "protect": protect,
            "rms_mix_rate": rms_mix_rate,
            "speaker_id": speaker_id,
            "use_cuda_graph": use_cuda_graph,
            "torch_retrieval": torch_retrieval,
            **(conversion_config or {}),
            **extra_conversion_config,
        }
        if output_sample_rate is not None:
            conversion["output_sample_rate"] = output_sample_rate
        if device is not None:
            conversion["device"] = device
        if demucs_device is not None:
            conversion["demucs_device"] = demucs_device
        SimpleRVCConverter().convert(
            source_backend_config={"id": "source", "instance_data_dir": str(source_root), "dataset_type": "audio"},
            target_backend_config={"id": "generated", "instance_data_dir": str(output_root), "dataset_type": "audio"},
            transform_config={"id": "rvc-convert", "method": "rvc", "model": self.model_config, "conversion": conversion},
            artifact=self.artifact,
            input_paths=[str(path) for path in input_paths],
            accelerator=None,
            run_logger=_ProgressLogger(progress_callback),
        )

    def _resolved_config(self) -> RVCConfig:
        model_file = self.artifact.model_path.name
        index_file = self.artifact.index_path.name if self.artifact.index_path else None
        features_file = FEATURES_SAFETENSORS_NAME if (self.artifact.cache_dir / FEATURES_SAFETENSORS_NAME).exists() else None
        if features_file is None and (self.artifact.cache_dir / FEATURES_NPY_NAME).exists():
            features_file = FEATURES_NPY_NAME
        return RVCConfig(
            model_name=self.config.model_name,
            architecture=self.config.architecture,
            sample_rate=self.config.sample_rate,
            f0=self.config.f0,
            model_file=model_file,
            index_file=index_file,
            features_file=features_file,
            asset_hub_model_id=self.config.asset_hub_model_id,
            metadata=self.config.metadata,
        )


class _SingleFileOutput:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.root = output_path.parent / f".{output_path.stem}.rvc-tmp"

    def __enter__(self) -> Path:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        return self.root

    def __exit__(self, *_exc: Any) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)


def _load_artifact(
    root: Path,
    model_name_fallback: str | None = None,
    model_path_override: Path | None = None,
) -> tuple[SimpleRVCArtifact, RVCConfig]:
    config_path = root / CONFIG_NAME
    if (root / VOICE_TRANSFORM_DIR).exists():
        voice_root = root / VOICE_TRANSFORM_DIR
    elif (root / "cache" / "rvc_model").exists():
        voice_root = root / "cache" / "rvc_model"
    else:
        voice_root = root
    manifest_path = voice_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else _manifest_base("", "rvc")
    if config_path.exists():
        config = RVCConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    else:
        config = _config_from_manifest(manifest)
    if config.model_name == DEFAULT_MODEL_NAME:
        config.model_name = _name_from_manifest(manifest) or model_name_fallback or root.name or DEFAULT_MODEL_NAME
    model_path = model_path_override or _first_existing(
        voice_root,
        [config.model_file, MODEL_SAFETENSORS_NAME, MODEL_PTH_NAME, "*.safetensors", "*.pth"],
    )
    index_path = _optional_existing(voice_root, [config.index_file, INDEX_NAME, "*.index"])
    payload = _load_model_payload(model_path)
    config.architecture = f"rvc-{payload.get('version', 'v2')}-{'f0' if payload.get('f0', True) else 'no-f0'}-{payload.get('sample_rate', 48000)}"
    config.sample_rate = int(payload.get("sample_rate", config.sample_rate))
    config.f0 = bool(payload.get("f0", config.f0))
    config.model_file = model_path.name
    if payload.get("speaker_info"):
        config.metadata["speaker_info"] = payload["speaker_info"]
    speaker_embedding = payload.get("generator_state_dict", {}).get("emb_g.weight")
    if hasattr(speaker_embedding, "shape"):
        config.metadata["speaker_count"] = int(speaker_embedding.shape[0])
    return SimpleRVCArtifact(voice_root, manifest_path, model_path, index_path, manifest), config


def _config_from_manifest(manifest: dict[str, Any]) -> RVCConfig:
    voice_model = manifest.get("voice_model") or {}
    sample_rate = int(voice_model.get("sample_rate", 48000))
    version = str(voice_model.get("version", "v2"))
    f0 = bool(voice_model.get("f0", True))
    return RVCConfig(
        model_name=_name_from_manifest(manifest) or DEFAULT_MODEL_NAME,
        architecture=f"rvc-{version}-{'f0' if f0 else 'no-f0'}-{sample_rate}",
        sample_rate=sample_rate,
        f0=f0,
        model_file=str(voice_model.get("model_file", MODEL_SAFETENSORS_NAME)),
        index_file=voice_model.get("index_file", INDEX_NAME),
        features_file=voice_model.get("features_file", FEATURES_SAFETENSORS_NAME),
        metadata={"speaker_info": voice_model["speaker_info"]} if voice_model.get("speaker_info") else {},
    )


def _manifest_base(fingerprint: str, transform_id: str, model_name: str | None = None) -> dict[str, Any]:
    manifest = {
        "format": "huggingface-hub-rvc",
        "format_version": FORMAT_VERSION,
        "task": "identity_transfer",
        "method": "rvc",
        "fingerprint": fingerprint,
        "transform_id": transform_id,
    }
    if model_name:
        manifest["model_name"] = model_name
    return manifest


def _name_from_manifest(manifest: dict[str, Any]) -> str | None:
    voice_model = manifest.get("voice_model") or {}
    name = manifest.get("model_name") or voice_model.get("model_name") or voice_model.get("name")
    return _coerce_model_name(name)


def _coerce_model_name(name: Any) -> str | None:
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def _name_from_repo_id(value: str | Path | None) -> str | None:
    if not value:
        return None
    text = str(value).rstrip("/")
    if not text:
        return None
    return text.rsplit("/", 1)[-1] or None


def _model_card_text(config: RVCConfig, manifest: dict[str, Any]) -> str:
    voice_model = manifest.get("voice_model") or {}
    lines = [
        "---",
        "library_name: huggingface-hub-rvc",
        "pipeline_tag: audio-to-audio",
        "tags:",
        "- rvc",
        "- voice-conversion",
        "- safetensors",
        "---",
        "",
        f"# {config.model_name}",
        "",
        "This repository contains an RVC voice-conversion artifact saved with `huggingface-hub-rvc`.",
        "",
        "## Artifact Layout",
        "",
        "```text",
        "config.json",
        "voice_transform/",
        "  manifest.json",
        f"  {config.model_file}",
    ]
    if config.features_file:
        lines.append(f"  {config.features_file}")
    if config.index_file:
        lines.append(f"  {config.index_file}")
    lines.extend(
        [
            "```",
            "",
            "## Model Details",
            "",
            f"- Model name: `{config.model_name}`",
            f"- Architecture: `{config.architecture}`",
            f"- Sample rate: `{config.sample_rate}`",
            f"- F0 conditioning: `{config.f0}`",
            f"- Format version: `{config.format_version}`",
            "",
            "## Voice Transform",
            "",
            f"- Task: `{manifest.get('task', 'identity_transfer')}`",
            f"- Method: `{manifest.get('method', 'rvc')}`",
        ]
    )
    if voice_model.get("input_count") is not None:
        lines.append(f"- Identity input files: `{voice_model['input_count']}`")
    if voice_model.get("frame_count") is not None:
        lines.append(f"- Indexed/training frames: `{voice_model['frame_count']}`")
    if voice_model.get("steps") is not None:
        lines.append(f"- RVC training steps: `{voice_model['steps']}`")
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```python",
            "from huggingface_hub_rvc import RVCPipeline",
            "",
            "pipe = RVCPipeline.from_pretrained(\"org/model-id\")",
            "pipe.convert_file(\"input.wav\", \"output.wav\")",
            "```",
            "",
            "## Safety And Rights",
            "",
            "Voice-conversion artifacts can encode a singer or speaker identity. Only use or publish this model where you have the required rights and consent.",
            "",
        ]
    )
    return "\n".join(lines)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _first_existing(root: Path, candidates: list[str | None]) -> Path:
    path = _optional_existing(root, candidates)
    if path is None:
        raise FileNotFoundError(f"No RVC model file found under {root}.")
    return path


def _optional_existing(root: Path, candidates: list[str | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        if "*" in candidate:
            matches = sorted(root.glob(candidate))
            if matches:
                return matches[0]
            continue
        path = root / candidate
        if path.exists():
            return path
    return None
