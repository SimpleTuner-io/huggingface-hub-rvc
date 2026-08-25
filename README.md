# huggingface-hub-rvc

`huggingface-hub-rvc` provides a small Hugging Face style API for RVC voice conversion artifacts:

- `RVCPipeline.from_pretrained(...)`
- `RVCPipeline.save_pretrained(...)`
- `RVCPipeline.push_to_hub(...)`
- `RVCPipeline.train(...)`
- `RVCPipeline.convert_file(...)`
- `RVCPipeline.convert_directory(...)`

The package uses the working RVC v2 F0 48 kHz architecture and stores new model weights as safetensors.

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

Legacy `model.pth` and `features.npy` artifacts can still be loaded.

## Load

```python
from huggingface_hub_rvc import RVCPipeline

pipe = RVCPipeline.from_pretrained("org/rvc-model")
pipe.convert_directory("input_audio", "converted_audio")
```

Use `local_files_only=True` to avoid network lookup:

```python
pipe = RVCPipeline.from_pretrained("./my-rvc-model", local_files_only=True)
```

## Train

```python
from huggingface_hub_rvc import RVCPipeline

pipe = RVCPipeline.train(
    identity_dir="identity_audio",
    output_dir="rvc_artifact",
    training_steps=1000,
    identity_audio_mode="separate",
)
pipe.convert_directory("source_audio", "converted_audio")
```

## Save And Push

```python
pipe.save_pretrained("rvc_artifact")
pipe.push_to_hub("org/rvc-model", folder_path="rvc_artifact")
```

or:

```python
pipe.save_pretrained("rvc_artifact", push_to_hub=True, repo_id="org/rvc-model")
```
