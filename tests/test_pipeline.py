import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from huggingface_hub_rvc import RVCConfig, RVCPipeline
from huggingface_hub_rvc._runtime import SimpleRVCConverter, SimpleRVCTrainer, _audio_paths


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


def _webui_config(speaker_count: int = 2) -> list:
    return [
        1025,
        32,
        192,
        192,
        768,
        2,
        6,
        3,
        0,
        "1",
        [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [12, 10, 2, 2],
        512,
        [24, 20, 4, 4],
        speaker_count,
        256,
        48000,
    ]


def test_from_pretrained_inspects_safe_webui_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "two-speaker.pth"
        torch.save(
            {
                "weight": {"emb_g.weight": torch.randn(2, 256)},
                "config": _webui_config(),
                "version": "v2",
                "f0": 1,
                "sr": "48k",
                "speaker_info": [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}],
            },
            checkpoint,
        )

        pipe = RVCPipeline.from_pretrained(checkpoint, local_files_only=True)

        assert pipe.config.model_name == "two-speaker"
        assert pipe.config.architecture == "rvc-v2-f0-48000"
        assert pipe.config.sample_rate == 48000
        assert pipe.config.metadata["speaker_count"] == 2
        assert pipe.config.metadata["speaker_info"][1]["name"] == "B"


def test_export_webui_is_owned_by_pipeline():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        checkpoint = root / "voice.pth"
        torch.save(
            {
                "weight": {"emb_g.weight": torch.randn(1, 256)},
                "config": _webui_config(1),
                "version": "v2",
                "f0": 1,
                "sr": "48k",
            },
            checkpoint,
        )
        pipe = RVCPipeline.from_pretrained(checkpoint, local_files_only=True)

        output = pipe.export_webui(root / "webui", model_name="Test Voice", training_steps=50)

        exported = torch.load(output / "Test-Voice.pth", map_location="cpu", weights_only=True)
        assert exported["version"] == "v2"
        assert exported["f0"] == 1
        assert exported["info"] == "50 training steps"


def test_pitch_shift_is_applied_in_semitones():
    class FakeRMVPE:
        @staticmethod
        def infer_from_audio(audio, thred):
            del thred
            return np.full(audio.shape[0] // 160, 100.0, dtype=np.float32)

    converter = SimpleRVCConverter()
    _, continuous = converter._get_f0(
        np.zeros(1600, dtype=np.float32),
        10,
        {"f0_method": "rmvpe", "pitch_shift": 12},
        torch.device("cpu"),
        FakeRMVPE(),
    )

    assert np.allclose(continuous, 200.0)


def test_audio_paths_ignore_macos_sidecars():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "voice.wav").write_bytes(b"wav")
        (root / "._voice.wav").write_bytes(b"sidecar")

        paths = _audio_paths(root)

        assert paths == [root / "voice.wav"]


def test_convert_directory_defaults_to_demucs_and_ignores_sidecars(monkeypatch):
    captured = {}

    def fake_convert(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(SimpleRVCConverter, "convert", fake_convert)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        artifact_root = _write_artifact(root / "artifact")
        source = root / "source"
        source.mkdir()
        (source / "voice.wav").write_bytes(b"wav")
        (source / "._voice.wav").write_bytes(b"sidecar")

        pipe = RVCPipeline.from_pretrained(artifact_root, local_files_only=True)
        pipe.convert_directory(source, root / "generated")

    assert captured["transform_config"]["conversion"]["separation_method"] == "demucs"
    assert [Path(path).name for path in captured["input_paths"]] == ["voice.wav"]


def test_pipeline_train_defaults_to_demucs():
    import inspect

    assert inspect.signature(RVCPipeline.train).parameters["separation_method"].default == "demucs"


def test_identity_separation_logs_and_preserves_vocal_stem(monkeypatch):
    sample_rate = 48000
    events = []

    class RunLogger:
        @staticmethod
        def event(transform_id, event, **payload):
            events.append({"transform_id": transform_id, "event": event, **payload})

    def fake_separate(input_path, output_dir, **_kwargs):
        stem_root = output_dir / input_path.stem
        stem_root.mkdir(parents=True)
        t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
        vocals = np.sin(2 * np.pi * 220 * t).astype(np.float32) * 0.25
        accompaniment = np.sin(2 * np.pi * 80 * t).astype(np.float32) * 0.05
        sf.write(stem_root / "vocals.wav", vocals, sample_rate)
        sf.write(stem_root / "no_vocals.wav", accompaniment, sample_rate)
        return stem_root

    monkeypatch.setattr("huggingface_hub_rvc._runtime._separate_two_stem", fake_separate)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_path = root / "mixed.wav"
        sf.write(input_path, np.zeros(sample_rate, dtype=np.float32), sample_rate)
        debug_dir = root / "debug"

        waveform = SimpleRVCTrainer()._load_identity_audio(
            input_path,
            sample_rate,
            "separate",
            {"separation_method": "demucs", "identity_stem_debug_dir": str(debug_dir)},
            torch.device("cpu"),
            run_logger=RunLogger(),
            transform_id="identity-test",
        )

        assert waveform.numel() == sample_rate
        assert (debug_dir / "mixed" / "vocals.wav").exists()
        assert (debug_dir / "mixed" / "no_vocals.wav").exists()

    assert events[0]["event"] == "identity_separation_complete"
    assert events[0]["separation_method"] == "demucs"
    assert events[0]["vocals_rms"] > events[0]["accompaniment_rms"]


def test_identity_preprocessing_slices_and_normalizes_segments():
    trainer = SimpleRVCTrainer()
    sample_rate = 48000
    waveform = torch.zeros(sample_rate * 8)
    t = torch.linspace(0, 8, sample_rate * 8)
    waveform += torch.sin(2 * torch.pi * 220 * t) * 0.1

    segments = trainer._identity_segments(
        waveform,
        sample_rate,
        {
            "preprocess_clip_seconds": 3.7,
            "preprocess_overlap_seconds": 0.3,
            "preprocess_min_segment_seconds": 1.0,
        },
    )

    assert len(segments) >= 2
    assert all(segment.ndim == 1 for segment in segments)
    assert all(segment.shape[0] <= int(sample_rate * 3.7) for segment in segments)
    assert all(torch.isfinite(segment).all() for segment in segments)


def test_offline_pipeline_preserves_duration_with_context_padding(monkeypatch):
    monkeypatch.setenv("RVC_CUDA_GRAPH", "0")

    class FakeHubert:
        @staticmethod
        def extract(audio, version):
            assert version == "v2"
            return torch.ones(max(1, audio.numel() // 320), 768)

    class FakeGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.emb_g = torch.nn.Embedding(1, 1)

        def infer(self, phone, lengths, sid):
            del lengths, sid
            output = torch.full((1, 1, phone.shape[1] * 160), 0.1, device=phone.device)
            return output, None, None

    waveform = torch.from_numpy(np.sin(np.linspace(0, 100, 16000 * 2, dtype=np.float32)))
    converted = SimpleRVCConverter()._convert_waveform(
        waveform,
        FakeGenerator(),
        FakeHubert(),
        None,
        None,
        None,
        {"is_half": False, "retrieval_strength": 0, "rms_mix_rate": 1},
        torch.device("cpu"),
        "v2",
        False,
        16000,
    )

    assert converted.shape == waveform.shape
    assert torch.allclose(converted, torch.full_like(converted, 0.1))
