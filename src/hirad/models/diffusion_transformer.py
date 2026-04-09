from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from .dit_layers import (
    DiTBlock,
    DetokenizerModuleBase,
    TokenizerModuleBase,
    get_detokenizer,
    get_tokenizer,
)

from .dit_conditioning_embedders import (
    ConditioningEmbedder,
    ConditioningEmbedderType,
    get_conditioning_embedder,
)

class DiT(nn.Module):
    r"""
    The Diffusion Transformer (DiT) model.

    Parameters
    ----------
    input_size : Union[int, Tuple[int]]
        Spatial dimensions of the input. If an integer is provided, the input is assumed to be on a square 2D domain.
        If a tuple is provided, the input is assumed to be on a multi-dimensional domain.
    in_channels : int
        The number of input channels.
    patch_size : Union[int, Tuple[int]], optional, default=(8, 8)
        The size of each image patch. If an integer is provided, a square 2D patch is assumed.
        If a tuple is provided, a multi-dimensional patch is assumed.
    tokenizer : Union[Literal["patch_embed_2d"], nn.Module], optional, default="patch_embed_2d"
        The tokenizer to use. Either a string in ``{"patch_embed_2d"}`` or an instantiated :class:`~nn.Module` implementing
        :class:`~hirad.models.TokenizerModuleBase`, with forward accepting input of shape :math:`(B, C, *\text{spatial\_dims})` and returning :math:`(B, L, D)`.
    detokenizer : Union[Literal["proj_reshape_2d"], nn.Module], optional, default="proj_reshape_2d"
        The detokenizer to use. Either a string in ``{"proj_reshape_2d"}`` or an instantiated :class:`~nn.Module` implementing
        :class:`~hirad.models.DetokenizerModuleBase`, with forward accepting :math:`(B, L, D)` and :math:`(B, D)` and returning :math:`(B, C, *\text{spatial\_dims})`.
    out_channels : Union[None, int], optional, default=None
        The number of output channels. If ``None``, set to ``in_channels``.
    hidden_size : int, optional, default=384
        The dimensionality of the transformer embeddings.
    depth : int, optional, default=12
        The number of transformer blocks.
    num_heads : int, optional, default=8
        The number of attention heads.
    mlp_ratio : float, optional, default=4.0
        The ratio of the MLP hidden dimension to the embedding dimension.
    attention_backend : Literal["timm", "transformer_engine", "natten2d"], optional, default="timm"
        The attention backend to use. See :class:`~hirad.models.DiTBlock` for a description of each built-in backend.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        If ``"apex"``, uses FusedLayerNorm from apex. If ``"torch"``, uses :class:`torch.nn.LayerNorm`. Also passed to :class:`~hirad.models.Natten2DSelfAttention` when ``qk_norm=True``.
    condition_dim : int, optional, default=None
        Dimensionality of conditioning. If ``None``, the model is unconditional.
    dit_initialization : bool, optional, default=True
        If ``True``, applies DiT-specific initialization.
    conditioning_embedder : Literal["dit", "edm", "zero"] or ConditioningEmbedder, optional, default="dit"
        The conditioning embedder type or an instantiated :class:`~hirad.models.nn.ConditioningEmbedder`.
    conditioning_embedder_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the conditioning embedder.
    tokenizer_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the tokenizer module.
    detokenizer_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the detokenizer module.
    block_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the DiTBlock modules.
    attn_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments for the attention module constructor (e.g. ``na2d_kwargs`` when using ``attention_backend="natten2d"``).
    drop_path_rates : list[float], optional, default=None
        DropPath (stochastic depth) rates, one per block. Must have length equal to ``depth``. If ``None``, no drop path is applied.
    force_tokenization_fp32 : bool, optional, default=False
        If ``True``, forces tokenization and de-tokenization to run in fp32.

    Forward
    -------
    x : torch.Tensor
        Spatial inputs of shape :math:`(N, C, *\text{spatial\_dims})`. ``spatial_dims`` is determined by ``input_size``.
    t : torch.Tensor
        Diffusion timesteps of shape :math:`(N,)`.
    condition : Optional[torch.Tensor]
        Conditions of shape :math:`(N, d)`.
    p_dropout : Optional[Union[float, torch.Tensor]], optional
        Dropout probability for the intermediate dropout (pre-attention) in each DiTBlock. If ``None``, no dropout. If a scalar, same for all samples; if a tensor, shape :math:`(B,)` for per-sample dropout.
    attn_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the attention module's forward method.
    tokenizer_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the tokenizer's forward method.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, \text{out\_channels}, *\text{spatial\_dims})`.

    Notes
    -----
    Reference: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers.
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4195-4205).

    Examples
    --------
    >>> model = DiT(
    ...     input_size=(32, 64),
    ...     patch_size=4,
    ...     in_channels=3,
    ...     out_channels=3,
    ...     condition_dim=8,
    ... )
    >>> x = torch.randn(2, 3, 32, 64)
    >>> t = torch.randint(0, 1000, (2,))
    >>> condition = torch.randn(2, 8)
    >>> output = model(x, t, condition)
    >>> output.shape
    torch.Size([2, 3, 32, 64])
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int]],
        in_channels: int,
        patch_size: Union[int, Tuple[int]] = (8, 8),
        tokenizer: Union[
            Literal["patch_embed_2d"], nn.Module
        ] = "patch_embed_2d",
        detokenizer: Union[
            Literal["proj_reshape_2d"], nn.Module
        ] = "proj_reshape_2d",
        out_channels: Optional[int] = None,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backend: Literal["timm", "transformer_engine", "natten2d"] = "timm",
        layernorm_backend: Literal["apex", "torch"] = "torch",
        condition_dim: Optional[int] = None,
        conditioning_embedder: Literal["dit", "edm", "zero"]
        | ConditioningEmbedder = "dit",
        dit_initialization: Optional[int] = True,
        conditioning_embedder_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        detokenizer_kwargs: Dict[str, Any] = {},
        block_kwargs: Dict[str, Any] = {},
        attn_kwargs: Dict[str, Any] = {},
        drop_path_rates: list[float] | None = None,
        force_tokenization_fp32: bool = False,
    ):
        super().__init__(meta=MetaData())
        self.input_size = (
            input_size
            if isinstance(input_size, (tuple, list))
            else (input_size, input_size)
        )
        self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = (
            patch_size
            if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )
        self.num_heads = num_heads
        self.condition_dim = condition_dim
        if attention_backend == "natten2d":
            latent_hw = (
                self.input_size[0] // self.patch_size[0],
                self.input_size[1] // self.patch_size[1],
            )
            self.attn_kwargs_forward = {"latent_hw": latent_hw}
        else:
            self.attn_kwargs_forward = {}

        # Input validation
        if attention_backend not in ["timm", "transformer_engine", "natten2d"]:
            raise ValueError(
                "attention_backend must be one of 'timm', 'transformer_engine', 'natten2d'"
            )

        if layernorm_backend not in ["apex", "torch"]:
            raise ValueError("layernorm_backend must be one of 'apex', 'torch'")

        if isinstance(tokenizer, str) and tokenizer not in [
            "patch_embed_2d",
        ]:
            raise ValueError("tokenizer must be 'patch_embed_2d'")

        if isinstance(detokenizer, str) and detokenizer not in [
            "proj_reshape_2d",
        ]:
            raise ValueError(
                "detokenizer must be 'proj_reshape_2d'"
            )

        # Tokenizer module: accept string or pre-instantiated Module
        if isinstance(tokenizer, str):
            self.tokenizer = get_tokenizer(
                input_size=self.input_size,
                patch_size=self.patch_size,
                in_channels=in_channels,
                hidden_size=hidden_size,
                tokenizer=tokenizer,
                **tokenizer_kwargs,
            )
        else:
            if not isinstance(tokenizer, TokenizerModuleBase):
                raise TypeError(
                    "tokenizer must be a string or a Module instance subclassing hirad.models.TokenizerModuleBase"
                )
            self.tokenizer = tokenizer

        # Conditioning embedder: accept enum or pre-instantiated Module
        if isinstance(conditioning_embedder, str):
            self.conditioning_embedder = get_conditioning_embedder(
                ConditioningEmbedderType[conditioning_embedder.upper()],
                hidden_size=hidden_size,
                condition_dim=condition_dim or 0,
                amp_mode=self.meta.amp_gpu,
                **conditioning_embedder_kwargs,
            )
        else:
            if not isinstance(conditioning_embedder, ConditioningEmbedder):
                raise TypeError(
                    "conditioning_embedder must be a ConditioningEmbedderType or a Module implementing the ConditioningEmbedder protocol"
                )
            self.conditioning_embedder = conditioning_embedder

        # Detokenizer module: accept string or pre-instantiated Module
        if isinstance(detokenizer, str):
            self.detokenizer = get_detokenizer(
                input_size=self.input_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                hidden_size=hidden_size,
                layernorm_backend=layernorm_backend,
                detokenizer=detokenizer,
                **detokenizer_kwargs,
            )
        else:
            if not isinstance(detokenizer, DetokenizerModuleBase):
                raise TypeError(
                    "detokenizer must be a string or a Module instance subclassing hirad.models.DetokenizerModuleBase"
                )
            self.detokenizer = detokenizer

        # Validate drop_path_rates
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        else:
            if len(drop_path_rates) != depth:
                raise ValueError(
                    f"drop_path_rates length ({len(drop_path_rates)}) must match DiT depth ({depth})"
                )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    attention_backend=attention_backend,
                    layernorm_backend=layernorm_backend,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[i],
                    condition_embed_dim=self.conditioning_embedder.output_dim,
                    **block_kwargs,
                    **attn_kwargs,
                )
                for i in range(depth)
            ]
        )

        if dit_initialization:
            self.initialize_weights()

        self.force_tokenization_fp32 = force_tokenization_fp32


    def initialize_weights(self):
        r"""Apply DiT-specific weight initialization.

        Applies Xavier uniform to linear layers, then delegates to tokenizer,
        detokenizer, and each block's ``initialize_weights``.

        Parameters
        ----------
        None
            Uses ``self`` (module state).

        Returns
        -------
        None
            Modifies module parameters in-place.
        """

        # Apply a basic Xavier uniform initialization to all linear layers.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Delegate custom weight initialization to the tokenizer, detokenizer, and blocks
        self.tokenizer.initialize_weights()
        self.detokenizer.initialize_weights()
        for block in self.blocks:
            block.initialize_weights()

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels *spatial_dims"],
        t: Float[torch.Tensor, " batch"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        p_dropout: Optional[float | Float[torch.Tensor, " batch"]] = None,
        attn_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ) -> Float[torch.Tensor, "batch out_channels *spatial_dims"]:
        # Tokenize: (B, C, H, W) -> (B, L, D)
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.tokenizer(x, **tokenizer_kwargs)
            x = x.to(dtype)
        else:
            x = self.tokenizer(x, **tokenizer_kwargs)

        # Compute conditioning embedding
        c = self.conditioning_embedder(t, condition=condition)  # (B, D)

        for block in self.blocks:
            x = block(
                x,
                c,
                p_dropout=p_dropout,
                attn_kwargs={**self.attn_kwargs_forward, **attn_kwargs},
            )  # (B, L, D)

        # De-tokenize: (B, L, D) -> (B, C, H, W)
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.detokenizer(x, c)
            x = x.to(dtype)
        else:
            x = self.detokenizer(x, c)

        return x