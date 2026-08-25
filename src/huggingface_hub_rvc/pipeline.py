from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
)

CONFIG_NAME = "config.json"
MANIFEST_NAME = "manifest.json"
VOICE_TRANSFORM_DIR = "voice_transform"
FORMAT_VERSION = 1
DEFAULT_MODEL_TYPE = "rvc"

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class RVCConfig:
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
        return cls(
            model_type=str(data.get("model_type", data.get("arch_type", DEFAULT_MODEL_TYPE))),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
            architecture=str(data.get("architecture", data.get("arch_version", "rvc-v2-f0-48k"))),
            sample_rate=int(data.get("sample_rate", 48000)),
            f0=bool(data.get("f0", True)),
            model_file=str(data.get("model_file", components.get("model", components.get("pth", MODEL_SAFETENSORS_NAME)))),
            index_file=data.get("index_file", components.get("index", INDEX_NAME)),
            features_file=data.get("features_file", components.get("features", FEATURES_SAFETENSORS_NAME)),
            asset_hub_model_id=str(data.get("asset_hub_model_id", DEFAULT_ASSET_REPO)),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
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
        if source.exists():
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
                        "*.json",
                    ],
                )
            )
            if subfolder:
                root = root / subfolder
        artifact, config = _load_artifact(root)
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
        build_index: bool = True,
        progress_callback: ProgressCallback | None = None,
        transform_id: str = "rvc-train",
        **model_config: Any,
    ) -> "RVCPipeline":
        output_root = Path(output_dir).expanduser()
        cache_dir = output_root / VOICE_TRANSFORM_DIR
        transform_config = {
            "id": transform_id,
            "method": "rvc",
            "model": {
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
        manifest_base = _manifest_base("rvc-train", transform_id)
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
            }
        )
        (voice_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (save_root / CONFIG_NAME).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
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
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        return api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message=commit_message,
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
        timbre_strength: float = 1.0,
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
            timbre_strength=timbre_strength,
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
        timbre_strength: float = 1.0,
        torch_retrieval: bool = True,
        progress_callback: ProgressCallback | None = None,
        conversion_config: dict[str, Any] | None = None,
    ) -> None:
        conversion = {
            "audio_mode": audio_mode,
            "separation_method": "demucs",
            "retrieval_strength": retrieval_strength,
            "timbre_strength": timbre_strength,
            "torch_retrieval": torch_retrieval,
            **(conversion_config or {}),
        }
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


def _load_artifact(root: Path) -> tuple[SimpleRVCArtifact, RVCConfig]:
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
    model_path = _first_existing(voice_root, [config.model_file, MODEL_SAFETENSORS_NAME, MODEL_PTH_NAME])
    index_path = _optional_existing(voice_root, [config.index_file, INDEX_NAME, "*.index"])
    return SimpleRVCArtifact(voice_root, manifest_path, model_path, index_path, manifest), config


def _config_from_manifest(manifest: dict[str, Any]) -> RVCConfig:
    voice_model = manifest.get("voice_model") or {}
    return RVCConfig(
        sample_rate=int(voice_model.get("sample_rate", 48000)),
        f0=bool(voice_model.get("f0", True)),
        model_file=str(voice_model.get("model_file", MODEL_SAFETENSORS_NAME)),
        index_file=voice_model.get("index_file", INDEX_NAME),
        features_file=voice_model.get("features_file", FEATURES_SAFETENSORS_NAME),
    )


def _manifest_base(fingerprint: str, transform_id: str) -> dict[str, Any]:
    return {
        "format": "huggingface-hub-rvc",
        "format_version": FORMAT_VERSION,
        "task": "identity_transfer",
        "method": "rvc",
        "fingerprint": fingerprint,
        "transform_id": transform_id,
    }


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
