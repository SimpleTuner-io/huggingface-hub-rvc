"""FCPE pitch extraction adapter used by RVC inference."""

from __future__ import annotations

import torch

from huggingface_hub_rvc.core.cuda_graph import cuda_graph_enabled, run_cuda_graph


class FCPEInfer:
    def __init__(self, device: torch.device | str) -> None:
        try:
            from torchfcpe import spawn_bundled_infer_model
        except ImportError as exc:
            raise ImportError("FCPE inference requires the 'torchfcpe' package.") from exc

        self.device = torch.device(device)
        self.infer_model = spawn_bundled_infer_model(self.device)
        if self.device.type == "cuda":
            self.local_offsets = torch.arange(9, device=self.device, dtype=torch.long).view(1, 1, 9)

    def _graphable_model_infer(self, mel: torch.Tensor, decoder_mode: str, threshold: float) -> torch.Tensor:
        model = self.infer_model.model
        latent = model(mel)
        batch, frames, _ = latent.shape
        cents = model.cent_table[None, None, :].expand(batch, frames, -1)

        if decoder_mode == "argmax":
            confidence = torch.max(latent, dim=-1, keepdim=True).values
            decoded = torch.sum(cents * latent, dim=-1, keepdim=True) / torch.sum(latent, dim=-1, keepdim=True)
        elif decoder_mode == "local_argmax":
            confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
            local_index = (self.local_offsets + (max_index - 4)).clamp(0, model.out_dims - 1)
            local_cents = torch.gather(cents, -1, local_index)
            local_latent = torch.gather(latent, -1, local_index)
            decoded = torch.sum(local_cents * local_latent, dim=-1, keepdim=True) / torch.sum(
                local_latent, dim=-1, keepdim=True
            )
        else:
            raise ValueError(f"Unknown FCPE decoder mode: {decoder_mode}")

        confidence_mask = torch.ones_like(confidence)
        confidence_mask.masked_fill_(confidence <= threshold, float("-inf"))
        return 10.0 * torch.pow(2.0, decoded * confidence_mask / 1200.0)

    @torch.no_grad()
    def infer(
        self,
        wav: torch.Tensor,
        sr: int,
        decoder_mode: str = "local_argmax",
        threshold: float = 0.006,
    ) -> torch.Tensor:
        wav = wav.to(self.device)
        if cuda_graph_enabled(wav.device):
            mel = self.infer_model.wav2mel(wav, sr)
            return run_cuda_graph(
                self.infer_model.model,
                f"fcpe-core-{decoder_mode}-{threshold}",
                lambda input_mel: self._graphable_model_infer(input_mel, decoder_mode, threshold),
                mel,
            )
        return self.infer_model.infer(wav, sr=sr, decoder_mode=decoder_mode, threshold=threshold)
