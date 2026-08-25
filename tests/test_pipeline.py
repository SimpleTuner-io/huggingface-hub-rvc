import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub_rvc import RVCConfig, RVCPipeline


def _write_artifact(root: Path, model_name: str | None = None, legacy_config: bool = False) -> Path:
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
    manifest = {
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
    if model_name:
        manifest["model_name"] = model_name
        manifest["voice_model"]["model_name"] = model_name
    (voice_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = RVCConfig(model_name=model_name or "Source Voice").to_dict()
    if legacy_config:
        config.pop("model_name")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def test_from_pretrained_loads_local_artifact():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = _write_artifact(Path(temp_dir), model_name="Test Voice")

        pipe = RVCPipeline.from_pretrained(root, local_files_only=True)

        assert pipe.artifact.model_path.name == "model.safetensors"
        assert pipe.artifact.index_path.name == "index.index"
        assert pipe.config.sample_rate == 48000
        assert pipe.config.model_name == "Test Voice"


def test_save_pretrained_writes_hub_layout():
    with tempfile.TemporaryDirectory() as temp_dir:
        source = _write_artifact(Path(temp_dir) / "source")
        save_root = Path(temp_dir) / "saved"
        pipe = RVCPipeline.from_pretrained(source, local_files_only=True)

        pipe.save_pretrained(save_root, model_name="Saved Voice")

        assert (save_root / "config.json").exists()
        assert (save_root / "README.md").exists()
        assert (save_root / "voice_transform" / "manifest.json").exists()
        assert (save_root / "voice_transform" / "model.safetensors").exists()
        assert (save_root / "voice_transform" / "features.safetensors").exists()
        config = json.loads((save_root / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((save_root / "voice_transform" / "manifest.json").read_text(encoding="utf-8"))
        assert config["model_name"] == "Saved Voice"
        assert manifest["model_name"] == "Saved Voice"
        assert manifest["voice_model"]["model_name"] == "Saved Voice"
        model_card = (save_root / "README.md").read_text(encoding="utf-8")
        assert "# Saved Voice" in model_card
        assert "library_name: huggingface-hub-rvc" in model_card
        with safe_open(str(save_root / "voice_transform" / "model.safetensors"), framework="pt", device="cpu") as handle:
            assert handle.metadata()["sample_rate"] == "48000"


def test_legacy_config_infers_model_name_from_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = _write_artifact(Path(temp_dir) / "legacy-voice", legacy_config=True)

        pipe = RVCPipeline.from_pretrained(root, local_files_only=True)

        assert pipe.config.model_name == "legacy-voice"
