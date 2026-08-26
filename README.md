# huggingface-hub-rvc

`huggingface-hub-rvc` provides a small Hugging Face style API for RVC voice conversion artifacts:

- `RVCPipeline.from_pretrained(...)`
- `RVCPipeline.save_pretrained(...)`
- `RVCPipeline.push_to_hub(...)`
- `RVCPipeline.export_webui(...)`
- `RVCPipeline.train(...)`
- `RVCPipeline.convert_file(...)`
- `RVCPipeline.convert_directory(...)`

Training currently produces RVC v2 F0 48 kHz models and stores new weights as safetensors. Inference also loads classic
RVC WebUI v1/v2, F0/non-F0, multi-speaker `.pth` checkpoints and their optional retrieval indexes with restricted
`weights_only=True` deserialization.

## Artifact Layout

```text
config.json
voice_transform/
  manifest.json
  model.safetensors
  features.safetensors
  index.index
```

`model.safetensors` contains the RVC generator weights plus string metadata for the RVC config and training summary.
`features.safetensors` contains retrieval vectors as tensor payload. `index.index` is the FAISS index and remains a separate binary file.
`config.json` includes `model_name`, which is also mirrored into `voice_transform/manifest.json` so the artifact remains identifiable even if it is downloaded into a generic cache or renamed folder.
`README.md` is generated as a Hub model card when `save_pretrained` writes the artifact.

Legacy `model.pth`, WebUI `.pth`, and `features.npy` artifacts can still be loaded. New writes prefer safetensors.

## Load

```python
from huggingface_hub_rvc import RVCPipeline

pipe = RVCPipeline.from_pretrained("org/rvc-model")
pipe.convert_directory(
    "input_audio",
    "converted_audio",
    pitch_shift=0,
    f0_method="rmvpe",  # rmvpe, fcpe, or pm
    protect=0.33,
    rms_mix_rate=1.0,
    retrieval_strength=0.75,
    speaker_id=0,
)
```

Use `local_files_only=True` to avoid network lookup:

```python
pipe = RVCPipeline.from_pretrained("./my-rvc-model", local_files_only=True)
```

The converter uses reflected context padding, low-energy chunk boundaries, and standard RVC consonant protection and
volume-envelope mixing. `audio_mode="separate_convert_remix"` supports `separation_method="pymss"` (default) or
`separation_method="demucs"`. CUDA Graph capture is opt-in with `use_cuda_graph=True` because its benefit depends on
input-shape reuse.

## Train

```python
from huggingface_hub_rvc import RVCPipeline

pipe = RVCPipeline.train(
    identity_dir="identity_audio",
    output_dir="rvc_artifact",
    model_name="Example Voice",
    training_steps=1000,
    identity_audio_mode="separate",
)
pipe.convert_directory("source_audio", "converted_audio")
```

Set `separation_method="demucs"` in `RVCPipeline.train(...)` to retain Demucs preprocessing instead of the default
PyMSS vocal separator. Already-isolated vocals should use `identity_audio_mode="vocal_only"`.

## Save And Push

```python
pipe.save_pretrained("rvc_artifact", model_name="Example Voice")
pipe.push_to_hub("org/rvc-model", folder_path="rvc_artifact")
```

or:

```python
pipe.save_pretrained("rvc_artifact", push_to_hub=True, repo_id="org/rvc-model")
```

Export a classic WebUI bundle when needed:

```python
pipe.export_webui("rvc_artifact/webui", model_name="Example Voice", training_steps=1000)
```
