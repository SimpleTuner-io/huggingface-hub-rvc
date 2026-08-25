import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub_rvc import RVCConfig, RVCPipeline


def _write_artifact(root: Path) -> Path:
    voice_root = root / "voice_transform"
    voice_root.mkdir(parents=True)
    save_file(
        {"emb_g.weight": torch.randn(1, 2)},
        str(voice_root / "model.safetensors"),
        metadata={
            "kind": "rvc-v2-f0",
            "version": "v2",
            "f0": "true",
            "sample_rate": "48000",
            "config_json": "{}",
            "training_json": "{}",
        },
    )
    save_file({"features": torch.randn(3, 768)}, str(voice_root / "features.safetensors"))
    (voice_root / "index.index").write_bytes(b"index")
    (voice_root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "huggingface-hub-rvc",
                "format_version": 1,
                "method": "rvc",
                "voice_model": {
                    "kind": "rvc-v2-f0",
                    "model_file": "model.safetensors",
                    "features_file": "features.safetensors",
                    "index_file": "index.index",
                    "sample_rate": 48000,
                    "f0": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(json.dumps(RVCConfig().to_dict()), encoding="utf-8")
    return root


def test_from_pretrained_loads_local_artifact():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = _write_artifact(Path(temp_dir))

        pipe = RVCPipeline.from_pretrained(root, local_files_only=True)

        assert pipe.artifact.model_path.name == "model.safetensors"
        assert pipe.artifact.index_path.name == "index.index"
        assert pipe.config.sample_rate == 48000


def test_save_pretrained_writes_hub_layout():
    with tempfile.TemporaryDirectory() as temp_dir:
        source = _write_artifact(Path(temp_dir) / "source")
        save_root = Path(temp_dir) / "saved"
        pipe = RVCPipeline.from_pretrained(source, local_files_only=True)

        pipe.save_pretrained(save_root)

        assert (save_root / "config.json").exists()
        assert (save_root / "voice_transform" / "manifest.json").exists()
        assert (save_root / "voice_transform" / "model.safetensors").exists()
        assert (save_root / "voice_transform" / "features.safetensors").exists()
        with safe_open(str(save_root / "voice_transform" / "model.safetensors"), framework="pt", device="cpu") as handle:
            assert handle.metadata()["sample_rate"] == "48000"
