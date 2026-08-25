"""Hugging Face Hub RVC RVC training and conversion.

The lifecycle is RVC artifact, but the model architecture, losses, F0
handling, and retrieval behavior are adapted from the MIT-licensed RVC project.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub_rvc.core import commons
from huggingface_hub_rvc.core.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from huggingface_hub_rvc.core.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from huggingface_hub_rvc.core.models import MultiPeriodDiscriminatorV2, SynthesizerTrnMs768NSFsid
from huggingface_hub_rvc.core.rmvpe import RMVPE

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}
DEFAULT_ASSET_REPO = "lj1995/VoiceConversionWebUI"
DEFAULT_VERSION = "v2"
EPS = 1e-6
MODEL_SAFETENSORS_NAME = "model.safetensors"
MODEL_PTH_NAME = "model.pth"
INDEX_NAME = "index.index"
FEATURES_SAFETENSORS_NAME = "features.safetensors"
FEATURES_NPY_NAME = "features.npy"
MODEL_KIND = "rvc-v2-f0"
LEGACY_MODEL_KIND = "simpletuner-rvc-v2-f0"
FEATURES_KIND = "rvc-features-v1"

RVC_48K_CONFIG: Dict[str, Any] = {
    "train": {
        "learning_rate": 1e-4,
        "betas": (0.8, 0.99),
        "eps": 1e-9,
        "batch_size": 4,
        "segment_size": 17280,
        "c_mel": 45.0,
        "c_kl": 1.0,
    },
    "data": {
        "max_wav_value": 32768.0,
        "sampling_rate": 48000,
        "filter_length": 2048,
        "hop_length": 480,
        "win_length": 2048,
        "n_mel_channels": 128,
        "mel_fmin": 0.0,
        "mel_fmax": None,
    },
    "model": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [12, 10, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [24, 20, 4, 4],
        "use_spectral_norm": False,
        "gin_channels": 256,
        "spk_embed_dim": 109,
    },
}


@dataclass(frozen=True)
class SimpleRVCArtifact:
    cache_dir: Path
    manifest_path: Path
    model_path: Path
    index_path: Optional[Path]
    manifest: Dict[str, Any]


@dataclass
class RVCRecord:
    phone: torch.Tensor
    pitch: torch.Tensor
    pitchf: torch.Tensor
    spec: torch.Tensor
    wave: torch.Tensor


class NoVoicedFramesError(ValueError):
    pass


def _hubert_base_class():
    from transformers import HubertModel

    return HubertModel


class _HubertModelWithFinalProj(_hubert_base_class()):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)


def _get_device(accelerator: Any = None, requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if accelerator is not None and getattr(accelerator, "device", None) is not None:
        return torch.device(getattr(accelerator, "device"))
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_arg(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"


def _audio_paths(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def _load_audio(path: Path, sample_rate: int, mono: bool = True) -> torch.Tensor:
    import torchaudio

    try:
        waveform, source_rate = torchaudio.load(str(path))
    except Exception as torchaudio_error:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1" if mono else "2",
                "-ar",
                str(sample_rate),
                str(tmp_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip()) from torchaudio_error
            waveform, source_rate = torchaudio.load(str(tmp_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if mono and waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(source_rate), sample_rate)
    if mono:
        return waveform.squeeze(0).to(torch.float32).contiguous()
    return waveform.to(torch.float32).contiguous()


def _save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu().to(torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    peak = waveform.abs().max().clamp_min(1.0)
    torchaudio.save(str(path), (waveform / peak).clamp(-0.99, 0.99), sample_rate)


def _copy_sidecars(source: Path, destination: Path) -> None:
    for suffix in (".txt", ".lyrics"):
        sidecar = source.with_suffix(suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, destination.with_suffix(suffix))


def _save_model_payload(path: Path, state_dict: Dict[str, torch.Tensor], training: Dict[str, float], model_name: Optional[str] = None) -> None:
    metadata = {
        "kind": MODEL_KIND,
        "version": DEFAULT_VERSION,
        "f0": json.dumps(True),
        "sample_rate": str(RVC_48K_CONFIG["data"]["sampling_rate"]),
        "config_json": json.dumps(RVC_48K_CONFIG, sort_keys=True),
        "training_json": json.dumps(training, sort_keys=True),
    }
    if model_name:
        metadata["model_name"] = model_name
    tensors = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
    save_safetensors_file(tensors, str(path), metadata=metadata)


def _load_model_payload(path: Path) -> Dict[str, Any]:
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return {
            "kind": metadata.get("kind"),
            "version": metadata.get("version"),
            "f0": json.loads(metadata.get("f0", "false")),
            "sample_rate": int(metadata.get("sample_rate", "0")),
            "config": json.loads(metadata["config_json"]) if metadata.get("config_json") else None,
            "generator_state_dict": load_safetensors_file(str(path), device="cpu"),
            "training": json.loads(metadata["training_json"]) if metadata.get("training_json") else {},
        }
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_feature_vectors(cache_dir: Path, features: np.ndarray) -> Path:
    path = cache_dir / FEATURES_SAFETENSORS_NAME
    save_safetensors_file({"features": torch.from_numpy(features)}, str(path), metadata={"kind": FEATURES_KIND})
    return path


def _load_feature_vectors(index_path: Path) -> Optional[np.ndarray]:
    safetensors_path = index_path.with_name(FEATURES_SAFETENSORS_NAME)
    if safetensors_path.exists():
        return load_safetensors_file(str(safetensors_path), device="cpu")["features"].numpy().astype("float32", copy=False)
    npy_path = index_path.with_name(FEATURES_NPY_NAME)
    if npy_path.exists():
        return np.load(npy_path).astype("float32", copy=False)
    return None


def _relative_output_path(source_root: Path, input_path: Path, output_root: Path) -> Path:
    relative = input_path.relative_to(source_root) if input_path.is_relative_to(source_root) else Path(input_path.name)
    return (output_root / relative).with_suffix(".wav")


def _run_demucs_two_stem(input_path: Path, output_dir: Path, device: str, model_name: str = "htdemucs") -> Path:
    try:
        import demucs  # noqa: F401
    except ImportError as exc:
        raise ImportError("demucs is required for identity_transfer Demucs separation.") from exc

    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        model_name,
        "-o",
        str(output_dir),
        "-d",
        device,
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"demucs separation failed for {input_path}: {result.stderr.strip()}")
    return output_dir / model_name / input_path.stem


def _coarse_f0(f0: np.ndarray) -> np.ndarray:
    f0_bin = 256
    f0_min = 50.0
    f0_max = 1100.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    return np.rint(f0_mel).astype(np.int64)


def _interp_unvoiced(f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float32)
    uv = f0 <= 0
    if uv.all():
        raise NoVoicedFramesError("identity_transfer RVC F0 extraction produced no voiced frames.")
    if uv.any():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return f0


class _RVCAssets:
    def __init__(self, cache_dir: Path, model_cfg: Dict[str, Any]) -> None:
        self.cache_dir = cache_dir
        self.model_cfg = model_cfg
        self.repo_id = str(model_cfg.get("asset_hub_model_id", DEFAULT_ASSET_REPO))

    def hubert_dir(self) -> Path:
        explicit = self.model_cfg.get("hubert_model_path")
        if explicit:
            path = Path(explicit).expanduser()
            if not (path / "config.json").exists():
                raise FileNotFoundError(f"identity_transfer.model.hubert_model_path is not a Transformers model folder: {path}")
            return path
        return self._snapshot_subdir("hubert_base", ["hubert_base/*"])

    def rmvpe_path(self) -> Path:
        return self._file("rmvpe_model_path", "rmvpe.pt")

    def generator_path(self) -> Path:
        return self._file("pretrained_generator_path", "pretrained_v2/f0G48k.pth")

    def discriminator_path(self) -> Path:
        return self._file("pretrained_discriminator_path", "pretrained_v2/f0D48k.pth")

    def _snapshot_subdir(self, subdir: str, allow_patterns: List[str]) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required to download default RVC assets.") from exc
        root = Path(
            snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=allow_patterns,
                local_dir=str(self.cache_dir / "assets"),
                token=self.model_cfg.get("asset_hub_token") or self.model_cfg.get("hub_token"),
            )
        )
        return root / subdir

    def _file(self, config_key: str, filename: str) -> Path:
        explicit = self.model_cfg.get(config_key)
        if explicit:
            path = Path(explicit).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"identity_transfer.model.{config_key} does not exist: {path}")
            return path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required to download default RVC assets.") from exc
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=str(self.cache_dir / "assets"),
                token=self.model_cfg.get("asset_hub_token") or self.model_cfg.get("hub_token"),
            )
        )


class _HubertFeatureExtractor:
    def __init__(self, model_dir: Path, device: torch.device, is_half: bool) -> None:
        from transformers import AutoFeatureExtractor

        self.device = device
        self.is_half = is_half and device.type == "cuda"
        dtype = torch.float16 if self.is_half else torch.float32
        self.model = _HubertModelWithFinalProj.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval().requires_grad_(False)
        self.normalize_audio = bool(AutoFeatureExtractor.from_pretrained(str(model_dir), local_files_only=True).do_normalize)

    def extract(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        source = waveform_16k.to(device=self.device, dtype=torch.float32)
        if self.normalize_audio:
            source = F.layer_norm(source, source.shape)
        source = source.unsqueeze(0)
        padding_mask = torch.zeros_like(source, dtype=torch.bool)
        with torch.inference_mode():
            outputs = self.model(
                input_values=source.half() if self.is_half else source,
                attention_mask=None if not padding_mask.any() else (~padding_mask).long(),
                output_hidden_states=False,
                return_dict=True,
            )
        return outputs.last_hidden_state.squeeze(0).float().detach().cpu()


class _RVCDataset(Dataset):
    def __init__(self, records: List[RVCRecord], segment_frames: int) -> None:
        self.records = records
        self.segment_frames = segment_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        length = min(record.phone.shape[0], record.spec.shape[1], record.pitch.shape[0], record.pitchf.shape[0])
        phone = record.phone[:length]
        pitch = record.pitch[:length]
        pitchf = record.pitchf[:length]
        spec = record.spec[:, :length]
        wave = record.wave[:, : length * RVC_48K_CONFIG["data"]["hop_length"]]
        return phone, pitch, pitchf, spec, wave, torch.tensor(length, dtype=torch.long)


class _RVCCollate:
    def __call__(self, batch):
        batch = sorted(batch, key=lambda item: int(item[-1]), reverse=True)
        lengths = torch.LongTensor([int(item[-1]) for item in batch])
        max_phone_len = max(item[0].shape[0] for item in batch)
        max_spec_len = max(item[3].shape[1] for item in batch)
        max_wave_len = max(item[4].shape[1] for item in batch)
        phone = torch.zeros(len(batch), max_phone_len, batch[0][0].shape[1])
        pitch = torch.zeros(len(batch), max_phone_len, dtype=torch.long)
        pitchf = torch.zeros(len(batch), max_phone_len)
        spec = torch.zeros(len(batch), batch[0][3].shape[0], max_spec_len)
        wave = torch.zeros(len(batch), 1, max_wave_len)
        for idx, item in enumerate(batch):
            phone[idx, : item[0].shape[0]] = item[0]
            pitch[idx, : item[1].shape[0]] = item[1]
            pitchf[idx, : item[2].shape[0]] = item[2]
            spec[idx, :, : item[3].shape[1]] = item[3]
            wave[idx, :, : item[4].shape[1]] = item[4]
        return phone, lengths, pitch, pitchf, spec, lengths.clone(), wave, lengths * RVC_48K_CONFIG["data"]["hop_length"], torch.zeros(len(batch), dtype=torch.long)


def _load_pretrained_generator(model: torch.nn.Module, path: Path) -> None:
    target = model.module if hasattr(model, "module") else model
    saved_state = torch.load(path, map_location="cpu")["model"]
    current_state = target.state_dict()
    embedding_key = "emb_g.weight"
    if embedding_key in saved_state and embedding_key in current_state and saved_state[embedding_key].shape != current_state[embedding_key].shape:
        saved_embedding = saved_state[embedding_key]
        current_embedding = current_state[embedding_key]
        if saved_embedding.dim() == current_embedding.dim() and saved_embedding.shape[1:] == current_embedding.shape[1:]:
            expanded = current_embedding.clone()
            rows = min(saved_embedding.shape[0], current_embedding.shape[0])
            expanded[:rows].copy_(saved_embedding[:rows])
            saved_state[embedding_key] = expanded
    target.load_state_dict(saved_state)


def _load_pretrained_discriminator(model: torch.nn.Module, path: Path) -> None:
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(torch.load(path, map_location="cpu")["model"])


def _make_generator(is_half: bool) -> SynthesizerTrnMs768NSFsid:
    data_cfg = RVC_48K_CONFIG["data"]
    return SynthesizerTrnMs768NSFsid(
        data_cfg["filter_length"] // 2 + 1,
        RVC_48K_CONFIG["train"]["segment_size"] // data_cfg["hop_length"],
        **RVC_48K_CONFIG["model"],
        is_half=is_half,
        sr=data_cfg["sampling_rate"],
    )


class SimpleRVCTrainer:
    """Train a real RVC v2 F0 voice artifact from target vocal examples."""

    def train(
        self,
        source_backend_config: Dict[str, Any],
        transform_config: Dict[str, Any],
        cache_dir: Path,
        fingerprint: str,
        manifest_base: Dict[str, Any],
        accelerator: Any = None,
        run_logger: Any = None,
    ) -> SimpleRVCArtifact:
        _ = source_backend_config
        model_cfg = transform_config.get("model") or {}
        identity_dir = model_cfg.get("identity_data_dir") or model_cfg.get("voice_data_dir")
        if not identity_dir:
            raise ValueError("identity_transfer.model.identity_data_dir is required when training an RVC artifact.")
        identity_root = Path(identity_dir).expanduser()
        if not identity_root.exists():
            raise FileNotFoundError(f"identity_transfer model identity_data_dir does not exist: {identity_root}")
        input_paths = _audio_paths(identity_root)
        if not input_paths:
            raise ValueError(f"identity_transfer found no target voice audio files under {identity_root}.")
        if int(model_cfg.get("sample_rate", RVC_48K_CONFIG["data"]["sampling_rate"])) != RVC_48K_CONFIG["data"]["sampling_rate"]:
            raise ValueError("identity_transfer RVC currently supports sample_rate=48000 with v2 48k pretrained assets.")

        cache_dir.mkdir(parents=True, exist_ok=True)
        device = _get_device(accelerator, model_cfg.get("device"))
        is_half = bool(model_cfg.get("is_half", device.type == "cuda"))
        rank = int(getattr(accelerator, "process_index", 0) or 0)
        world_size = int(getattr(accelerator, "num_processes", 1) or 1)
        assets = _RVCAssets(cache_dir, model_cfg)

        records = self._preprocess_rank(
            input_paths=input_paths,
            model_cfg=model_cfg,
            cache_dir=cache_dir,
            rank=rank,
            world_size=world_size,
            device=device,
            is_half=is_half,
            assets=assets,
            transform_id=transform_config["id"],
            run_logger=run_logger,
        )
        shard_path = cache_dir / f"records.rank{rank}.pt"
        torch.save(records, shard_path)
        if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        all_records = self._load_all_records(cache_dir, world_size)
        if not all_records:
            raise ValueError("identity_transfer RVC training produced no usable voice records.")
        train_stats = self._train_rvc(
            records=all_records,
            cache_dir=cache_dir,
            assets=assets,
            model_cfg=model_cfg,
            accelerator=accelerator,
            device=device,
            is_half=is_half,
            transform_id=transform_config["id"],
            run_logger=run_logger,
        )
        if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        if rank == 0:
            index_path = self._build_index(all_records, cache_dir, model_cfg)
            manifest_path = cache_dir / "manifest.json"
            voice_model = {
                "kind": MODEL_KIND,
                "model_file": MODEL_SAFETENSORS_NAME,
                "index_file": INDEX_NAME if index_path is not None else None,
                "features_file": FEATURES_SAFETENSORS_NAME if (cache_dir / FEATURES_SAFETENSORS_NAME).exists() else None,
                "sample_rate": RVC_48K_CONFIG["data"]["sampling_rate"],
                "version": DEFAULT_VERSION,
                "f0": True,
                "input_count": len(input_paths),
                "frame_count": int(sum(record.phone.shape[0] for record in all_records)),
                **train_stats,
            }
            if model_cfg.get("model_name"):
                voice_model["model_name"] = model_cfg["model_name"]
            manifest = {
                **manifest_base,
                "fingerprint": fingerprint,
                "voice_model": voice_model,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            if run_logger is not None:
                run_logger.event(transform_config["id"], "voice_model_training_complete", **train_stats)

        if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        manifest_path = cache_dir / "manifest.json"
        model_path = cache_dir / MODEL_SAFETENSORS_NAME
        if not model_path.exists():
            model_path = cache_dir / MODEL_PTH_NAME
        index_path = cache_dir / INDEX_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SimpleRVCArtifact(
            cache_dir=cache_dir,
            manifest_path=manifest_path,
            model_path=model_path,
            index_path=index_path if index_path.exists() else None,
            manifest=manifest,
        )

    def _preprocess_rank(
        self,
        input_paths: List[Path],
        model_cfg: Dict[str, Any],
        cache_dir: Path,
        rank: int,
        world_size: int,
        device: torch.device,
        is_half: bool,
        assets: _RVCAssets,
        transform_id: str,
        run_logger: Any,
    ) -> List[RVCRecord]:
        sample_rate = RVC_48K_CONFIG["data"]["sampling_rate"]
        hop_length = RVC_48K_CONFIG["data"]["hop_length"]
        identity_audio_mode = str(model_cfg.get("identity_audio_mode", "separate"))
        max_seconds_per_file = float(model_cfg.get("max_seconds_per_file", 180.0))
        hubert = _HubertFeatureExtractor(assets.hubert_dir(), device, is_half)
        rmvpe = RMVPE(str(assets.rmvpe_path()), is_half=is_half, device=str(device))
        records: List[RVCRecord] = []
        for audio_path in [path for idx, path in enumerate(input_paths) if idx % world_size == rank]:
            waveform = self._load_identity_audio(audio_path, sample_rate, identity_audio_mode, model_cfg, device)
            max_samples = int(max_seconds_per_file * sample_rate)
            if max_samples > 0 and waveform.numel() > max_samples:
                waveform = waveform[:max_samples]
            try:
                record = self._record_from_waveform(waveform, hubert, rmvpe, sample_rate, hop_length)
            except NoVoicedFramesError:
                if run_logger is not None:
                    run_logger.event(transform_id, "voice_model_feature_skip", path=str(audio_path), reason="no_voiced_frames")
                continue
            records.append(record)
            if run_logger is not None:
                run_logger.event(transform_id, "voice_model_feature_extract", path=str(audio_path), frames=record.phone.shape[0])
        return records

    def _record_from_waveform(
        self,
        waveform: torch.Tensor,
        hubert: _HubertFeatureExtractor,
        rmvpe: RMVPE,
        sample_rate: int,
        hop_length: int,
    ) -> RVCRecord:
        import torchaudio

        wave = waveform.unsqueeze(0)
        spec = spectrogram_torch(
            wave,
            RVC_48K_CONFIG["data"]["filter_length"],
            sample_rate,
            hop_length,
            RVC_48K_CONFIG["data"]["win_length"],
            center=False,
        ).squeeze(0).cpu()
        waveform_16k = torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, 16000).squeeze(0)
        phone = hubert.extract(waveform_16k)
        phone = phone.repeat_interleave(2, dim=0)
        f0 = _interp_unvoiced(rmvpe.infer_from_audio(waveform_16k.detach().cpu().numpy(), thred=0.03))
        pitch = torch.from_numpy(_coarse_f0(f0)).long()
        pitchf = torch.from_numpy(f0.astype(np.float32))
        length = min(phone.shape[0], spec.shape[1], pitch.shape[0], pitchf.shape[0])
        return RVCRecord(
            phone=phone[:length].contiguous(),
            pitch=pitch[:length].contiguous(),
            pitchf=pitchf[:length].contiguous(),
            spec=spec[:, :length].contiguous(),
            wave=wave[:, : length * hop_length].contiguous().cpu(),
        )

    def _load_identity_audio(
        self,
        audio_path: Path,
        sample_rate: int,
        identity_audio_mode: str,
        model_cfg: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        if identity_audio_mode in {"vocal_only", "stem", "none"}:
            return _load_audio(audio_path, sample_rate)
        if identity_audio_mode != "separate":
            raise ValueError("identity_transfer.model.identity_audio_mode must be 'separate' or 'vocal_only'.")
        with tempfile.TemporaryDirectory(prefix="hfhub-rvc-identity-demucs-") as tmp_dir:
            separated_root = _run_demucs_two_stem(
                audio_path,
                Path(tmp_dir),
                str(model_cfg.get("demucs_device", _device_arg(device))),
                str(model_cfg.get("demucs_model", "htdemucs")),
            )
            return _load_audio(separated_root / "vocals.wav", sample_rate)

    def _load_all_records(self, cache_dir: Path, world_size: int) -> List[RVCRecord]:
        records: List[RVCRecord] = []
        for idx in range(world_size):
            shard = cache_dir / f"records.rank{idx}.pt"
            if shard.exists():
                records.extend(torch.load(shard, map_location="cpu", weights_only=False))
        return records

    def _train_rvc(
        self,
        records: List[RVCRecord],
        cache_dir: Path,
        assets: _RVCAssets,
        model_cfg: Dict[str, Any],
        accelerator: Any,
        device: torch.device,
        is_half: bool,
        transform_id: str,
        run_logger: Any,
    ) -> Dict[str, float]:
        steps = int(model_cfg.get("training_steps", model_cfg.get("acoustic_training_steps", 1000)))
        batch_size = int(model_cfg.get("batch_size", model_cfg.get("acoustic_batch_size", RVC_48K_CONFIG["train"]["batch_size"])))
        learning_rate = float(model_cfg.get("learning_rate", model_cfg.get("acoustic_learning_rate", RVC_48K_CONFIG["train"]["learning_rate"])))
        segment_frames = RVC_48K_CONFIG["train"]["segment_size"] // RVC_48K_CONFIG["data"]["hop_length"]
        dataset = _RVCDataset(records, segment_frames)
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(getattr(accelerator, "num_processes", 1) or 1),
            rank=int(getattr(accelerator, "process_index", 0) or 0),
            shuffle=True,
        ) if accelerator is not None and int(getattr(accelerator, "num_processes", 1) or 1) > 1 else None
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, collate_fn=_RVCCollate())
        net_g = _make_generator(is_half=is_half).to(device)
        net_d = MultiPeriodDiscriminatorV2(RVC_48K_CONFIG["model"]["use_spectral_norm"]).to(device)
        _load_pretrained_generator(net_g, assets.generator_path())
        _load_pretrained_discriminator(net_d, assets.discriminator_path())
        optim_g = torch.optim.AdamW(net_g.parameters(), learning_rate, betas=RVC_48K_CONFIG["train"]["betas"], eps=RVC_48K_CONFIG["train"]["eps"])
        optim_d = torch.optim.AdamW(net_d.parameters(), learning_rate, betas=RVC_48K_CONFIG["train"]["betas"], eps=RVC_48K_CONFIG["train"]["eps"])
        if accelerator is not None:
            net_g, net_d, optim_g, optim_d, loader = accelerator.prepare(net_g, net_d, optim_g, optim_d, loader)

        last: Dict[str, float] = {}
        global_step = 0
        while global_step < steps:
            if sampler is not None:
                sampler.set_epoch(global_step)
            for batch in loader:
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = [
                    item.to(device) if torch.is_tensor(item) else item for item in batch
                ]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_half and device.type == "cuda"):
                    y_hat, ids_slice, _, z_mask, latent = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                    z, z_p, m_p, logs_p, _, logs_q = latent
                    mel = spec_to_mel_torch(
                        spec,
                        RVC_48K_CONFIG["data"]["filter_length"],
                        RVC_48K_CONFIG["data"]["n_mel_channels"],
                        RVC_48K_CONFIG["data"]["sampling_rate"],
                        RVC_48K_CONFIG["data"]["mel_fmin"],
                        RVC_48K_CONFIG["data"]["mel_fmax"],
                    )
                    y_mel = commons.slice_segments(mel, ids_slice, segment_frames)
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        RVC_48K_CONFIG["data"]["filter_length"],
                        RVC_48K_CONFIG["data"]["n_mel_channels"],
                        RVC_48K_CONFIG["data"]["sampling_rate"],
                        RVC_48K_CONFIG["data"]["hop_length"],
                        RVC_48K_CONFIG["data"]["win_length"],
                        RVC_48K_CONFIG["data"]["mel_fmin"],
                        RVC_48K_CONFIG["data"]["mel_fmax"],
                    )
                    wave_slice = commons.slice_segments(wave, ids_slice * RVC_48K_CONFIG["data"]["hop_length"], RVC_48K_CONFIG["train"]["segment_size"])
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave_slice, y_hat.detach())
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                optim_d.zero_grad(set_to_none=True)
                if accelerator is not None:
                    accelerator.backward(loss_disc)
                else:
                    loss_disc.backward()
                optim_d.step()

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_half and device.type == "cuda"):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_slice, y_hat)
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * RVC_48K_CONFIG["train"]["c_mel"]
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * RVC_48K_CONFIG["train"]["c_kl"]
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                optim_g.zero_grad(set_to_none=True)
                if accelerator is not None:
                    accelerator.backward(loss_gen_all)
                else:
                    loss_gen_all.backward()
                optim_g.step()

                global_step += 1
                last = {
                    "steps": float(global_step),
                    "loss_disc": float(loss_disc.detach().float().cpu()),
                    "loss_gen": float(loss_gen_all.detach().float().cpu()),
                    "loss_mel": float(loss_mel.detach().float().cpu()),
                    "loss_kl": float(loss_kl.detach().float().cpu()),
                }
                if run_logger is not None and (global_step == 1 or global_step % max(1, steps // 10) == 0):
                    run_logger.event(transform_id, "voice_model_train_step", **last)
                if global_step >= steps:
                    break

        is_main = accelerator is None or bool(getattr(accelerator, "is_main_process", True))
        if is_main:
            unwrapped_g = accelerator.unwrap_model(net_g) if accelerator is not None else net_g
            _save_model_payload(cache_dir / MODEL_SAFETENSORS_NAME, unwrapped_g.cpu().state_dict(), last, model_name=model_cfg.get("model_name"))
        return last

    def _build_index(self, records: List[RVCRecord], cache_dir: Path, model_cfg: Dict[str, Any]) -> Optional[Path]:
        if not bool(model_cfg.get("build_index", True)):
            return None
        features = torch.cat([record.phone for record in records], dim=0).numpy().astype("float32")
        if features.shape[0] == 0:
            return None
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required when identity_transfer.model.build_index is true.") from exc
        flat_threshold = int(model_cfg.get("flat_index_threshold", 100000))
        if features.shape[0] < flat_threshold:
            index = faiss.IndexFlatL2(features.shape[1])
        else:
            n_ivf = max(1, min(int(16 * np.sqrt(features.shape[0])), max(1, features.shape[0] // 39)))
            index = faiss.index_factory(features.shape[1], f"IVF{n_ivf},Flat")
            faiss.extract_index_ivf(index).nprobe = 1
            index.train(features)
        for start in range(0, features.shape[0], 8192):
            index.add(features[start : start + 8192])
        index_path = cache_dir / INDEX_NAME
        faiss.write_index(index, str(index_path))
        _save_feature_vectors(cache_dir, features)
        return index_path


class SimpleRVCConverter:
    """Apply a SimpleTuner RVC artifact to audio files."""

    def convert(
        self,
        source_backend_config: Dict[str, Any],
        target_backend_config: Dict[str, Any],
        transform_config: Dict[str, Any],
        artifact: SimpleRVCArtifact,
        input_paths: Iterable[str],
        accelerator: Any = None,
        run_logger: Any = None,
    ) -> None:
        source_root = Path(source_backend_config["instance_data_dir"]).expanduser()
        output_root = Path(target_backend_config["instance_data_dir"]).expanduser()
        conversion_cfg = transform_config.get("conversion") or {}
        audio_mode = conversion_cfg.get("audio_mode", "vocal_only")
        if audio_mode not in {"vocal_only", "separate_convert_remix", "full_mix_convert"}:
            raise ValueError(f"Unsupported identity_transfer conversion.audio_mode={audio_mode!r}.")
        device = _get_device(accelerator, conversion_cfg.get("device"))
        model_payload = _load_model_payload(artifact.model_path)
        if model_payload.get("kind") not in {MODEL_KIND, LEGACY_MODEL_KIND}:
            raise ValueError("identity_transfer found an incompatible RVC artifact. Remove the old rvc_model cache and rerun.")
        model_cfg = transform_config.get("model") or {}
        assets = _RVCAssets(artifact.cache_dir, model_cfg)
        hubert = _HubertFeatureExtractor(assets.hubert_dir(), device, bool(conversion_cfg.get("is_half", device.type == "cuda")))
        rmvpe = RMVPE(str(assets.rmvpe_path()), is_half=bool(conversion_cfg.get("is_half", device.type == "cuda")), device=str(device))
        net_g = _make_generator(is_half=bool(conversion_cfg.get("is_half", device.type == "cuda"))).to(device)
        net_g.load_state_dict(model_payload["generator_state_dict"])
        net_g.eval()
        index, index_vectors = self._load_index(artifact.index_path, conversion_cfg)

        for input_name in input_paths:
            input_path = Path(input_name)
            output_path = _relative_output_path(source_root, input_path, output_root)
            if audio_mode == "separate_convert_remix":
                self._convert_with_demucs(input_path, output_path, net_g, hubert, rmvpe, index, index_vectors, conversion_cfg, device)
            else:
                waveform = _load_audio(input_path, RVC_48K_CONFIG["data"]["sampling_rate"])
                converted = self._convert_waveform(waveform, net_g, hubert, rmvpe, index, index_vectors, conversion_cfg, device)
                _save_audio(output_path, converted, RVC_48K_CONFIG["data"]["sampling_rate"])
            _copy_sidecars(input_path, output_path)
            if run_logger is not None:
                run_logger.event(transform_config["id"], "conversion_file_complete", source=str(input_path), output=str(output_path))

    def _load_index(self, index_path: Optional[Path], conversion_cfg: Dict[str, Any]):
        index_rate = float(conversion_cfg.get("retrieval_strength", conversion_cfg.get("index_rate", 0.75)))
        if index_path is None or index_rate == 0:
            return None, None
        index_vectors = _load_feature_vectors(index_path)
        if bool(conversion_cfg.get("torch_retrieval", True)) and index_vectors is not None:
            return None, index_vectors
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required when identity_transfer conversion uses retrieval_strength > 0.") from exc
        index = faiss.read_index(str(index_path))
        return index, index.reconstruct_n(0, index.ntotal)

    def _convert_with_demucs(
        self,
        input_path: Path,
        output_path: Path,
        net_g: torch.nn.Module,
        hubert: _HubertFeatureExtractor,
        rmvpe: RMVPE,
        index: Any,
        index_vectors: Any,
        conversion_cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        if conversion_cfg.get("separation_method", "demucs") != "demucs":
            raise ValueError("separate_convert_remix requires conversion.separation_method='demucs'.")
        with tempfile.TemporaryDirectory(prefix="hfhub-rvc-demucs-") as tmp_dir:
            separated_root = _run_demucs_two_stem(input_path, Path(tmp_dir), str(conversion_cfg.get("demucs_device", _device_arg(device))), str(conversion_cfg.get("demucs_model", "htdemucs")))
            converted_vocals = self._convert_waveform(
                _load_audio(separated_root / "vocals.wav", RVC_48K_CONFIG["data"]["sampling_rate"]),
                net_g,
                hubert,
                rmvpe,
                index,
                index_vectors,
                conversion_cfg,
                device,
            )
            instrumental = _load_audio(separated_root / "no_vocals.wav", RVC_48K_CONFIG["data"]["sampling_rate"], mono=False)
            length = min(converted_vocals.shape[-1], instrumental.shape[-1])
            vocals = converted_vocals[:length]
            accompaniment = instrumental[..., :length]
            if accompaniment.ndim == 2:
                vocals = vocals.unsqueeze(0).expand(accompaniment.shape[0], -1)
            mixed = vocals + accompaniment
            peak = mixed.abs().max()
            if peak > 0.99:
                mixed = mixed / peak * 0.99
            _save_audio(output_path, mixed, RVC_48K_CONFIG["data"]["sampling_rate"])

    def _convert_waveform(
        self,
        waveform: torch.Tensor,
        net_g: torch.nn.Module,
        hubert: _HubertFeatureExtractor,
        rmvpe: RMVPE,
        index: Any,
        index_vectors: Any,
        conversion_cfg: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        import torchaudio

        sr = RVC_48K_CONFIG["data"]["sampling_rate"]
        waveform_16k = torchaudio.functional.resample(waveform.unsqueeze(0), sr, 16000).squeeze(0)
        filtered = signal.filtfilt(*signal.butter(N=5, Wn=48, btype="high", fs=16000), waveform_16k.detach().cpu().numpy())
        phone = hubert.extract(torch.from_numpy(filtered.astype(np.float32))).to(device)
        retrieval_strength = float(conversion_cfg.get("retrieval_strength", conversion_cfg.get("index_rate", 0.75)))
        if index_vectors is not None and retrieval_strength > 0:
            npy = phone.detach().cpu().numpy().astype("float32")
            if index is None:
                retrieved = self._torch_retrieve(npy, index_vectors, device)
            else:
                score, ix = index.search(npy, k=8)
                score = np.maximum(score, EPS)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                retrieved = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            phone = torch.from_numpy(retrieved).to(device) * retrieval_strength + phone * (1 - retrieval_strength)
        phone = F.interpolate(phone.unsqueeze(0).permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = phone.shape[1]
        f0 = _interp_unvoiced(rmvpe.infer_from_audio(filtered.astype(np.float32), thred=0.03))
        pitch = torch.from_numpy(_coarse_f0(f0)[:p_len]).unsqueeze(0).long().to(device)
        pitchf = torch.from_numpy(f0.astype(np.float32)[:p_len]).unsqueeze(0).float().to(device)
        if pitch.shape[1] < p_len:
            p_len = pitch.shape[1]
            phone = phone[:, :p_len]
        lengths = torch.tensor([p_len], dtype=torch.long, device=device)
        sid = torch.zeros(1, dtype=torch.long, device=device)
        with torch.inference_mode():
            audio = net_g.infer(phone, lengths, pitch, pitchf, sid)[0][0, 0].detach().float().cpu()
        source_rms = waveform.pow(2).mean().sqrt().clamp_min(EPS)
        converted_rms = audio.pow(2).mean().sqrt().clamp_min(EPS)
        audio = audio * (source_rms / converted_rms)
        strength = float(conversion_cfg.get("timbre_strength", 1.0))
        if strength < 1.0:
            length = min(audio.numel(), waveform.numel())
            audio = waveform[:length] * (1.0 - strength) + audio[:length] * strength
        return audio[: waveform.numel()]

    def _torch_retrieve(self, query: np.ndarray, index_vectors: np.ndarray, device: torch.device) -> np.ndarray:
        keys = torch.from_numpy(index_vectors).to(device=device, dtype=torch.float32)
        outputs = []
        k = min(8, keys.shape[0])
        for chunk_np in np.array_split(query, max(1, math.ceil(query.shape[0] / 512))):
            chunk = torch.from_numpy(chunk_np).to(device=device, dtype=torch.float32)
            distances = torch.cdist(chunk, keys).clamp_min(EPS)
            score, indices = torch.topk(distances, k=k, largest=False)
            weights = torch.square(1.0 / score)
            weights = weights / weights.sum(dim=1, keepdim=True)
            outputs.append((keys[indices] * weights.unsqueeze(-1)).sum(dim=1).detach().cpu())
        return torch.cat(outputs, dim=0).numpy().astype("float32", copy=False)
