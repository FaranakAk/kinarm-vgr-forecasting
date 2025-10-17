import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model

from momentfm.common import TASKS
from momentfm.data.base import TimeseriesOutputs
from momentfm.models.layers.embed import PatchEmbedding, Patching
from momentfm.models.layers.revin import RevIN
from momentfm.utils.masking import Masking
from momentfm.utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        x = x.flatten(start_dim=2, end_dim=3)
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_channels * d_model, n_classes)

    def forward(self, x, input_mask: torch.Tensor = None):
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        y = self.linear(x)
        return y


# class ForecastingHead(nn.Module):
#     def __init__(
#         self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
#     ):
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.dropout = nn.Dropout(head_dropout)
#         self.linear = nn.Linear(head_nf, forecast_horizon)

#     def forward(self, x, input_mask: torch.Tensor = None):
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.dropout(x)
#         return x

# Option 1 - dir1
# class ForecastingHead(nn.Module):
#     def __init__(self, head_nf: int, forecast_horizon: int = 64, head_dropout: int = 0):
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=-2)  # Flatten patches and model dimension
#         self.dropout = nn.Dropout(head_dropout)
#         self.linear = nn.Linear(head_nf + 1, forecast_horizon)  # Add 1 for desired_direction

#     def forward(self, x, desired_direction, input_mask: torch.Tensor = None):
#         # Flatten input tensor
#         x = self.flatten(x)  # Shape: [batch_size, n_channels, n_patches * d_model]
#         batch_size, n_channels, flattened_features = x.shape

#         # Repeat desired_direction to match the flattened input shape
#         desired_direction = desired_direction.unsqueeze(-1)  # Shape: [batch_size, 1, 1]
#         desired_direction = desired_direction.repeat(1, n_channels, 1)  # Shape: [batch_size, n_channels, 1]

#         # Concatenate desired_direction along the feature dimension
#         x = torch.cat([x, desired_direction], dim=2)  # Shape: [batch_size, n_channels, flattened_features + 1]

#         # Process through the linear layer
#         x = self.linear(x)  # Shape: [batch_size, n_channels, forecast_horizon]
#         x = self.dropout(x)

#         return x

# Option 2 - dir2
# class ForecastingHead(nn.Module):
#     def __init__(self, head_nf: int, forecast_horizon: int = 64, head_dropout: int = 0):
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.embedding = nn.Embedding(8, head_nf)
#         self.dropout = nn.Dropout(head_dropout)
#         self.linear = nn.Linear(head_nf * 3, forecast_horizon)

#     def forward(self, x, context_directions, forecast_direction):
#         x = self.flatten(x)
#         batch_size, n_channels, flattened_features = x.shape

#         context_embeds = self.embedding(context_directions)
#         forecast_embed = self.embedding(forecast_direction).unsqueeze(1)

#         weights = (context_directions == forecast_direction.unsqueeze(1)).float()
#         weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

#         # Weight both context embeddings and historic features
#         weighted_context_embed = (context_embeds * weights.unsqueeze(-1)).sum(dim=1)
#         x_weights = weights.unsqueeze(-1)
#         weighted_features = (x * x_weights).sum(dim=1)

#         # Combine all components
#         combined_features = torch.cat([weighted_features, weighted_context_embed, forecast_embed.squeeze(1)], dim=1)
#         x = self.linear(combined_features)
#         x = self.dropout(x)
#         return x.unsqueeze(1)




    
    
# Option 3 - dir3
# class ForecastingHead(nn.Module):
#     """
#     A forecasting head that:
#       1) Embeds context directions for each patch,
#       2) Embeds the single forecast direction,
#       3) Weighs context patches by how many match the forecast direction,
#       4) Merges weighted patch features + direction embeddings to produce a forecast.
#     """
#     def __init__(self, head_nf: int, forecast_horizon: int = 64, head_dropout: int = 0, num_directions: int = 8):
#         """
#         Args:
#             head_nf: The number of features after the patches have been encoded (e.g., d_model * n_patches).
#             forecast_horizon: How many steps we forecast (64 in your case).
#             head_dropout: Dropout probability.
#             num_directions: Number of possible directions (default = 8).
#         """
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=-2)    # Flatten [B, n_channels, n_patches, d_model] -> [B, n_channels, n_patches*d_model]
#         self.embedding = nn.Embedding(num_directions, head_nf)  # Embedding for direction indices
#         self.dropout = nn.Dropout(head_dropout)

#         # Our final linear layer uses 'head_nf * 3' because:
#         #   (1) Weighted patch features => shape [head_nf]
#         #   (2) Weighted context direction embedding => shape [head_nf]
#         #   (3) Forecast direction embedding => shape [head_nf]
#         # Summed => 3 * head_nf
#         self.linear = nn.Linear(head_nf * 3, forecast_horizon)

#     def forward(self, x: torch.Tensor, context_directions: torch.Tensor, forecast_direction: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x:                [batch_size, n_channels, n_patches, d_model]
#             context_directions: [batch_size, n_patches] repeated directions for each patch
#             forecast_direction: [batch_size] single direction index for forecast
#         Returns:
#             A forecast of shape [batch_size, 1, forecast_horizon]
#         """

#         # Flatten patch dimension
#         # Example shape: [B, n_channels, 64, d_model] -> [B, n_channels, 64*d_model]
#         x = self.flatten(x)  # => [batch_size, n_channels, n_patches * d_model]
#         batch_size, n_channels, flattened_features = x.shape

#         # We want to "sum" or "weight" across the patch dimension, so let's partly reshape
#         # from [B, n_channels, n_patches*d_model] back to [B, n_channels, n_patches, d_model].
#         d_model = self.embedding.weight.shape[1]   # dimension of each direction embedding
#         n_patches = flattened_features // d_model
#         x_reshaped = x.view(batch_size, n_channels, n_patches, d_model)  # => [B, n_channels, n_patches, d_model]

#         # Average across channels -> [B, n_patches, d_model]
#         x_reshaped = x_reshaped.mean(dim=1)  # => [B, n_patches, d_model]

#         # Embed context directions => [B, n_patches, head_nf]
#         context_embeds = self.embedding(context_directions)  # => [B, n_patches, head_nf]

#         # Embed forecast direction => [B, head_nf]
#         forecast_embed = self.embedding(forecast_direction)  # => [B, head_nf]

#         # Weight context embeddings by how many match the forecast direction
#         # context_directions => [B, n_patches]
#         # forecast_direction => [B], unsqueeze to compare => [B, 1]
#         weights = (context_directions == forecast_direction.unsqueeze(1)).float()  # => [B, n_patches]
#         weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)        # => normalized across paches

#         # Weighted context direction embedding => [B, head_nf]
#         weighted_context_embed = (context_embeds * weights.unsqueeze(-1)).sum(dim=1)

#         # Weighted patch features => [B, d_model]
#         #  If you want to weigh the patch-encoded features similarly:
#         weighted_patch_features = (x_reshaped * weights.unsqueeze(-1)).sum(dim=1)  # => [B, d_model]

#         # Combine
#         # => final shape: [B, head_nf*3], because both 'weighted_context_embed' and
#         #    'weighted_patch_features' are dimension head_nf, and forecast_embed is head_nf.
#         combined_features = torch.cat([weighted_patch_features, weighted_context_embed, forecast_embed], dim=1)

#         out = self.linear(combined_features)  # => [B, forecast_horizon]
#         out = self.dropout(out)
#         return out.unsqueeze(1)  # => [B, 1, forecast_horizon]



# dir3 - set
class ForecastingHead(nn.Module):
    """
    Forecasting head that:
     - Interprets the 4096-length input as 8 sub-trials (each 512) => 512 patches
     - Uses context directions (8 total) to weigh sub-trials that match the forecast direction
     - Avoids matmul shape errors by reshaping to 2D before linear
    """
    def __init__(self, head_nf: int, forecast_horizon: int = 64, head_dropout: int = 0, num_directions: int = 8):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)

        # Suppose your frozen backbone has d_model=768 and patch_len=8 => n_patches = 4096/8 = 512
        # head_nf is typically d_model * n_patches (e.g. 768*512 = 393216).
        # We'll do sub-trial projection from (64 * d_model) -> d_model inside forward(),
        # so let's store d_model for dimension references:
        self.d_model = head_nf // 512  # e.g. if head_nf=393216, then d_model=768

        # Projection layer: each sub-trial chunk (64*d_model) -> d_model
        self.subtrial_project = nn.Linear(64 * self.d_model, self.d_model)

        # Embedding for directions, shape => (num_directions, d_model)
        self.embedding = nn.Embedding(num_directions, self.d_model)

        # Final linear: combine [d_model * 3] => forecast_horizon
        self.linear = nn.Linear(self.d_model * 3, forecast_horizon)

    def forward(self, x: torch.Tensor, context_directions: torch.Tensor, forecast_direction: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_channels, n_patches, d_model], with n_patches=512 if patch_len=8 & total length=4096
        context_directions: [B, 8] => one direction per sub-trial
        forecast_direction: [B]
        Returns: [B, 1, forecast_horizon]
        """
        B, n_channels, n_patches, d_model = x.shape
        # Flatten channels & patches:
        # => [B, n_channels, 512*d_model]
        x = x.reshape(B, n_channels, n_patches * d_model)
        # Average over channels => [B, 512*d_model]
        x = x.mean(dim=1)

        # We interpret 512 patches as 8 sub-trials x 64 patches each => (64*d_model) per sub-trial
        # => reshape to [B, 8, 64*d_model]
        x_reshaped = x.view(B, 8, 64 * d_model)

        # ---- 1) Project each sub-trial from (64*d_model) -> d_model ----
        # Flatten sub-trials => shape: [B*8, 64*d_model]
        x_2d = x_reshaped.reshape(B * 8, 64 * d_model)
        x_subtrial_2d = self.subtrial_project(x_2d)   # => shape (B*8, d_model)
        x_subtrial = x_subtrial_2d.view(B, 8, self.d_model)  # => (B,8,d_model)

        # ---- 2) Embed each sub-trial direction => [B,8,d_model]
        context_embeds = self.embedding(context_directions)   # [B,8,d_model]

        # ---- 3) Embed the forecast direction => [B,d_model]
        forecast_embed = self.embedding(forecast_direction)

        # ---- 4) Weight the sub-trials by whether their direction == forecast_direction
        # context_directions => [B,8], forecast_direction => [B]
        weights = (context_directions == forecast_direction.unsqueeze(1)).float()  # => [B,8]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)       # => normalized

        # Weighted sub-trial features => [B,d_model]
        weighted_subtrial = (x_subtrial * weights.unsqueeze(-1)).sum(dim=1)

        # Weighted sub-trial direction embed => [B,d_model]
        weighted_context_embed = (context_embeds * weights.unsqueeze(-1)).sum(dim=1)

        # Combine => [B, 3*d_model]
        combined = torch.cat([weighted_subtrial, weighted_context_embed, forecast_embed], dim=1)

        out = self.linear(combined)   # => [B, forecast_horizon]
        out = self.dropout(out)
        return out.unsqueeze(1)  
    
    
# For set 2:   
class SingleTrialAggregatorHead(nn.Module):
    """
    Aggregates the embeddings from 8 single trials, weighting those whose direction matches forecast_dir
    and then outputs the forecast.
    """
    def __init__(self, d_model=768, forecast_horizon=64, num_directions=8, dropout_prob=0.5):
        super().__init__()
        self.num_directions = num_directions
        self.dir_embed = nn.Embedding(num_directions, d_model)
        # Add a dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        self.proj = nn.Linear(d_model*2, forecast_horizon)  # e.g. combine (subtrial_embed, direction_embed, forecast_embed)

    def forward(self, pipeline, context, forecast_dir, context_dirs):
        """
        context: shape => (n_channels,8,64)
        context_dirs: shape => (8,)
        forecast_dir => scalar
        pipeline => your MOMENTPipeline, so we can call pipeline.encode_single_trial(...)
        """
        device = context.device
        B = 1  # we have a single subject in this "batch"

        # We'll collect subtrial embeddings & direction embeddings
        subtrial_embeds = []
        subtrial_dirs   = []

        # gather sub-trial embeddings
        # shape => (8,n_channels,64)
        context = context.permute(1,0,2) # => (8,n_channels,64)
        for i in range(8):
            single_trial = context[i]  # => shape(n_channels,64)
            single_trial = single_trial.unsqueeze(0)  # => shape(1,n_channels,64)
            emb = pipeline.encode_single_trial(single_trial) # => shape(1,d_model)
            subtrial_embeds.append(emb)                      # => (1,d_model)

            d = context_dirs[i]
            subtrial_dirs.append(d)

        # stack => shape (8,d_model)
        subtrial_embeds = torch.cat(subtrial_embeds, dim=0)  # => (8,d_model)
        subtrial_dirs   = torch.tensor(subtrial_dirs, dtype=torch.long, device=device) # => (8,)

        # forecast_dir => shape( ), we can embed
        forecast_dir_embed = self.dir_embed(forecast_dir.unsqueeze(0))  # => (1,d_model)

        # embed context dirs => shape(8,d_model)
        context_dirs_embed = self.dir_embed(subtrial_dirs)              # => (8,d_model)

        # We do weighting: sub-trials whose direction == forecast_dir get higher weight
        weights = (subtrial_dirs == forecast_dir).float()  # => (8,)
        # e.g. sum of weights is how many match
        weights_sum = weights.sum()
        weights = weights / weights_sum.clamp_min(1e-6)
        # => shape(8,)

        # Weighted subtrial embedding => (d_model)
        weighted_subtrial = (subtrial_embeds * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        # Weighted context direction => (d_model)
        weighted_dir_embed = (context_dirs_embed * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

        # Combine => shape => (1, d_model*3)
        combined = torch.cat([weighted_subtrial, forecast_dir_embed], dim=1) # => (1,d_model*3) #commented weighted_dir_embed as the second parameter
        
        # Apply dropout for stochasticity.
        combined = self.dropout(combined)
        
        
        out = self.proj(combined)  # => (1, forecast_horizon)
        return out.unsqueeze(1)    # => (1,1,64)








class MOMENT(nn.Module):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len
        
        
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.patch_len)
        print()

        self.normalizer = RevIN(
            num_features=2, affine=config.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=config.patch_len, stride=config.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            dropout=config.getattr("dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
            num_directions=8,  # Add number of directions #dir3
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_transformer_backbone(config)
        self.head = self._get_head(self.task_name)

        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _update_inputs(
        self, config: Namespace | dict, **kwargs: dict
    ) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            config.d_model = get_huggingface_model_dimensions(
                config.transformer_backbone
            )
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name == TASKS.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("dropout", 0.1),
            )
        elif task_name == TASKS.FORECASTING:
            num_patches = (
                max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
            ) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * num_patches
            return ForecastingHead(
                self.head_nf,
                self.config.forecast_horizon,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        if config.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(config.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized transformer from {config.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                config.transformer_backbone
            )
            logging.info(
                f"Initializing pre-trained transformer from {config.transformer_backbone}."
            )

        transformer_backbone = transformer_backbone.get_encoder()

        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        context_directions: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        # enc_in = self.patch_embedding(x_enc, mask=input_mask)
        enc_in = self.patch_embedding(x_enc, mask=input_mask, directions=context_directions) #dir3


        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def reconstruction(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        context_directions: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # Prevent too short time-series from causing NaNs
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)
        # enc_in = self.patch_embedding(x_enc, mask=mask)
        enc_in = self.patch_embedding(x_enc, mask=mask, directions=context_directions) #dir3


        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.config.getattr("debug", False):
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        )

    def reconstruct(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        context_directions: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")

        x_enc = self.tokenizer(x=x_enc)
        # enc_in = self.patch_embedding(x_enc, mask=mask)
        enc_in = self.patch_embedding(x_enc, mask=mask, directions=context_directions) #dir3

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0).to(
            x_enc.device
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)
            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def detect_anomalies(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        anomaly_criterion: str = "mse",
        **kwargs,
    ) -> TimeseriesOutputs:
        outputs = self.reconstruct(x_enc=x_enc, input_mask=input_mask, context_directions=context_directions,) #dir3
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )
    
    # For set 2
    # In your MOMENTPipeline class or in MOMENT class:
    def encode_single_trial(self, single_trial: torch.Tensor) -> torch.Tensor:
        """
        single_trial => shape (n_channels,64) or (1,n_channels,64). We'll unify to (1,n_channels,64).
        We'll produce a shape (1, d_model) embedding by averaging patches + channels from the frozen encoder.
        """
        if single_trial.dim() == 2:
            # => (n_channels,64)
            single_trial = single_trial.unsqueeze(0)  # => (1,n_channels,64)
        B, n_channels, seq_len = single_trial.shape
    
        device = single_trial.device
        mask = torch.ones((B, seq_len), dtype=torch.float32, device=device)
    
        # 1) Normalize
        single_trial = self.normalizer(x=single_trial, mask=mask, mode="norm")
        single_trial = torch.nan_to_num(single_trial, nan=0.0, posinf=0.0, neginf=0.0)
    
        # 2) Tokenize
        x_enc = self.tokenizer(x=single_trial)  # => shape (B,n_channels,n_patches,patch_len)
        # 3) Patch embedding (no directions here, or pass None)
        enc_in = self.patch_embedding(x_enc, mask=mask, directions=None)
        # => shape (B,n_channels,n_patches,d_model)
    
        # flatten channels => (B*n_channels,n_patches,d_model)
        B2, n_chan2, n_patches, d_model = enc_in.shape
        enc_in = enc_in.view(B2*n_chan2, n_patches, d_model)
    
        # attention mask => ones
        attn_mask = torch.ones((B2*n_chan2, n_patches), dtype=torch.float32, device=device)
    
        # 4) Frozen T5 encoder
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attn_mask)
        enc_out = outputs.last_hidden_state
        # => shape (B2*n_chan2,n_patches,d_model)
    
        # reshape => (B2,n_chan2,n_patches,d_model)
        enc_out = enc_out.view(B2, n_chan2, n_patches, d_model)
    
        # average across patches, then channels => shape (B2,d_model)
        enc_out = enc_out.mean(dim=2).mean(dim=1) # => (B2,d_model) => typically (1,d_model)
    
        return enc_out


    # def forecast(
    #     self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, desired_direction: torch.Tensor = None, **kwargs
    # ) -> TimeseriesOutputs:
    #     batch_size, n_channels, seq_len = x_enc.shape
    #     # print(f"x_enc shape (input to forecasting): {x_enc.shape}")

    #     x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
    #     x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

    #     x_enc = self.tokenizer(x=x_enc)
    #     enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))
        
        
        
    #     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.patch_embedding)

    #     n_patches = enc_in.shape[2]
    #     enc_in = enc_in.reshape(
    #         (batch_size * n_channels, n_patches, self.config.d_model)
    #     )
    #     # print(f"enc_in shape (after patch embedding): {enc_in.shape}")

    #     patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
    #     attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
    #     outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
    #     enc_out = outputs.last_hidden_state
    #     enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
    #     # print(f"enc_out shape (output of encoder): {enc_out.shape}")
    #     # [batch_size x n_channels x n_patches x d_model]

    #     # dec_out = self.head(enc_out)  # [batch_size x n_channels x forecast_horizon]
    #     dec_out = self.head(enc_out, desired_direction)
    #     # print(f"dec_out shape (after head): {dec_out.shape}")
    #     dec_out = self.normalizer(x=dec_out, mode="denorm")

    #     return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)
    
    # # Option 2 - dir2
    # def forecast(
    #     self, x_enc: torch.Tensor, input_mask: torch.Tensor, context_directions: torch.Tensor, forecast_direction: torch.Tensor, **kwargs
    # ) -> TimeseriesOutputs:
    #     batch_size, n_channels, seq_len = x_enc.shape

    #     # Normalize and tokenize input
    #     x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
    #     x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
    #     x_enc = self.tokenizer(x=x_enc)

    #     # Patch embedding
    #     enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))
    #     n_patches = enc_in.shape[2]
    #     enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model))

    #     # Pass through the transformer backbone
    #     patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
    #     attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
    #     outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
    #     enc_out = outputs.last_hidden_state
    #     enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

    #     # Pass to the forecasting head
    #     dec_out = self.head(enc_out, context_directions, forecast_direction)
    #     dec_out = self.normalizer(x=dec_out, mode="denorm")

    #     return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)
    
    # Option 3 - dir3
    def forecast(
        self, x_enc: torch.Tensor, input_mask: torch.Tensor, context_directions: torch.Tensor, forecast_direction: torch.Tensor, **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        # Normalize and tokenize input
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        x_enc = self.tokenizer(x=x_enc)

        # Patch embedding
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask), directions=context_directions)
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model))

        # Pass through the transformer backbone
        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        # Pass to the forecasting head
        dec_out = self.head(enc_out, context_directions, forecast_direction)
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

    def short_forecast(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        forecast_horizon: int = 1,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape
        num_masked_patches = ceil(forecast_horizon / self.patch_len)
        num_masked_timesteps = num_masked_patches * self.patch_len

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        # Shift the time-series and mask the last few timesteps for forecasting
        x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
        input_mask = torch.roll(input_mask, shifts=-num_masked_timesteps, dims=1)

        # Attending to mask tokens
        input_mask[:, -num_masked_timesteps:] = 1
        mask = torch.ones_like(input_mask)
        mask[:, -num_masked_timesteps:] = 0

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]

        end = -num_masked_timesteps + forecast_horizon
        end = None if end == 0 else end

        dec_out = self.normalizer(x=dec_out, mode="denorm")
        forecast = dec_out[:, :, -num_masked_timesteps:end]

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            forecast=forecast,
            metadata={"forecast_horizon": forecast_horizon},
        )

    def classify(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        logits = self.head(enc_out, input_mask=input_mask)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)

    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        context_directions: torch.Tensor = None,
        forecast_direction: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask, context_directions=context_directions, **kwargs
            ) #dir3
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, context_directions=context_directions, **kwargs) #dir3
        # elif self.task_name == TASKS.FORECASTING:
        #     return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        # elif self.task_name == TASKS.FORECASTING:# dir1
        #     return self.forecast(
        #         x_enc=x_enc, input_mask=input_mask, desired_direction=desired_direction, **kwargs
        #     )
        # elif self.task_name == TASKS.FORECASTING: #dir2
        #     return self.forecast(
        #         x_enc=x_enc,
        #         input_mask=input_mask,
        #         context_directions=context_directions,
        #         forecast_direction=forecast_direction,
        #         **kwargs,
        #     )
        elif self.task_name == TASKS.FORECASTING: #dir3
            return self.forecast(
                x_enc=x_enc,
                input_mask=input_mask,
                context_directions=context_directions,  # Pass directions here
                forecast_direction=forecast_direction,
                **kwargs,
            )

        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.RECONSTRUCTION
        )
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == TASKS.FORECASTING:
            if not hasattr(config, "forecast_horizon"):
                raise ValueError(
                    "forecast_horizon must be specified for long-horizon forecasting."
                )

        if config.task_name == TASKS.CLASSIFICATION:
            if not hasattr(config, "n_channels"):
                raise ValueError("n_channels must be specified for classification.")
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")

    def init(self) -> None:
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)

def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model























