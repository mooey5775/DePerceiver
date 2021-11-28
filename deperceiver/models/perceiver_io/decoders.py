from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch import nn

from deperceiver.models.perceiver_io.attention import CrossAttention, MultiHeadAttention


class BasePerceiverDecoder(nn.Module, metaclass=ABCMeta):
    """Abstract decoder class."""
    @abstractmethod
    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        return NotImplementedError


class ProjectionDecoder(BasePerceiverDecoder):
    """Projection decoder without using a cross-attention layer."""
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.projection = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        latents = latents.mean(dim=1)
        logits = self.projection(latents)
        return logits


class PerceiverMultipleDecoderLayer(BasePerceiverDecoder):
    """Single layer of multi-layer decoder"""
    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        widening_factor: int = 1,
        num_heads: int = 1,
        dropout: int = 0.1,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        use_query_residual: bool = False
    ) -> None:
        super().__init__()
        self.partial_self_attention = MultiHeadAttention(
            kv_dim=query_dim,
            q_dim=query_dim,
            output_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attention = CrossAttention(
            kv_dim=latent_dim,
            q_dim=query_dim,
            widening_factor=widening_factor,
            num_heads=num_heads,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            use_query_residual=use_query_residual,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        if projection_dim is not None:
            self.projection = nn.Linear(query_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        *,
        tgt: torch.Tensor,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        if q_mask is not None:
            q_mask = q_mask[:, None, None, :].transpose(-2, -1)
        initial_qk = tgt + query
        tgt2 = self.partial_self_attention(
            inputs_kv=initial_qk,
            inputs_q=initial_qk,
            attention_mask=q_mask,
        )
        # Maybe need extra dropout here?
        # Although tgt2 already has dropout applied
        # However, DeTR has dropout applied anyway for some reason
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        outputs = self.cross_attention(
            inputs_kv=latents,
            inputs_q=(tgt + query),
            attention_mask=q_mask
        )
        return self.projection(outputs)


class PerceiverMultipleDecoder(BasePerceiverDecoder):
    """Multi-layer decoder with cross-attention."""
    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        num_layers: int,
        widening_factor: int = 1,
        num_heads: int = 1,
        dropout: int = 0.1,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        use_query_residual: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PerceiverMultipleDecoderLayer(
                    latent_dim=latent_dim,
                    query_dim=query_dim,
                    widening_factor=widening_factor,
                    num_heads=num_heads,
                    dropout=dropout,
                    qk_out_dim=qk_out_dim,
                    v_out_dim=v_out_dim,
                    projection_dim=projection_dim,
                    use_query_residual=use_query_residual
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        tgt = torch.zeros_like(query)
        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                query=query,
                latents=latents,
                q_mask=q_mask
            )
        return tgt


class PerceiverDecoder(BasePerceiverDecoder):
    """Basic cross-attention decoder."""
    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        widening_factor: int = 1,
        num_heads: int = 1,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        use_query_residual: bool = False
    ):
        super().__init__()
        self.cross_attention = CrossAttention(
            kv_dim=latent_dim,
            q_dim=query_dim,
            widening_factor=widening_factor,
            num_heads=num_heads,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            use_query_residual=use_query_residual
        )
        if projection_dim is not None:
            self.projection = nn.Linear(query_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        if q_mask is not None:
            q_mask = q_mask[:, None, None, :].transpose(-2, -1)
        outputs = self.cross_attention(
            inputs_kv=latents,
            inputs_q=query,
            attention_mask=q_mask
        )
        return self.projection(outputs)


class ClassificationDecoder(BasePerceiverDecoder):
    """Classification decoder. Based on PerceiverDecoder."""
    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        widening_factor: int = 1,
        num_heads: int = 1,
    ):
        super().__init__()
        self.task_ids = nn.Parameter(torch.randn(1, num_classes))
        self.decoder = PerceiverDecoder(
            latent_dim=latent_dim,
            query_dim=num_classes,
            widening_factor=widening_factor,
            num_heads=num_heads,
            projection_dim=None,
            use_query_residual=False
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        latents: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None
    ):
        batch_size = latents.size(0)
        logits = self.decoder.forward(
            query=self.task_ids.repeat(batch_size, 1, 1),
            latents=latents,
            q_mask=q_mask
        )
        return logits.squeeze(1)

