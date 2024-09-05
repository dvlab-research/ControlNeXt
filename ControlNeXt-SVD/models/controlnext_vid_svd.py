from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import math
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks  import get_down_block, UNetMidBlockSpatioTemporal
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from einops import rearrange



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNeXtOutput(BaseOutput):
    """
    The output of [`ControlNeXtModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in zip(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
    

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, *args):
        if len(args) > 0:
            return args[0]
        else:
            return None


class BasicBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                out_channels: Optional[int] = None,
                stride=1,
                conv_shortcut: bool = False,
                dropout: float = 0.0,
                temb_channels: int = 512,
                groups: int = 32,
                groups_out: Optional[int] = None,
                pre_norm: bool = True,
                eps: float = 1e-6,
                non_linearity: str = "swish",
                skip_time_act: bool = False,
                time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
                kernel: Optional[torch.FloatTensor] = None,
                output_scale_factor: float = 1.0,
                use_in_shortcut: Optional[bool] = None,
                up: bool = False,
                down: bool = False,
                conv_shortcut_bias: bool = True,
                conv_2d_out_channels: Optional[int] = None,):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=3 if stride != 1 else 1, 
                          stride=stride, 
                          padding=1 if stride != 1 else 0, 
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, *args):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            # in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # ResnetBlock2D(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     temb_channels=temb_channels,
                #     eps=resnet_eps,
                #     groups=resnet_groups,
                #     dropout=dropout,
                #     time_embedding_norm=resnet_time_scale_shift,
                #     non_linearity=resnet_act_fn,
                #     output_scale_factor=output_scale_factor,
                #     pre_norm=resnet_pre_norm,
                BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ) if i == num_layers - 1 else \
                IdentityModule()
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # Downsample2D(
                    #     out_channels,
                    #     use_conv=True,
                    #     out_channels=out_channels,
                    #     padding=downsample_padding,
                    #     name="op",
                    # )
                    BasicBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        stride=2,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
    




class ControlProject(nn.Module):
    def __init__(self, num_channels, scale=8, is_empty=False) -> None:
        super().__init__()
        assert scale and scale & (scale - 1) == 0
        self.is_empty = is_empty
        self.scale = scale
        if not is_empty:
            if scale > 1:
                self.down_scale = nn.AvgPool2d(scale, scale)
            else:
                self.down_scale = nn.Identity()
            self.out = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=False)
            for p in self.out.parameters():
                nn.init.zeros_(p)

    def forward(
        self,
        hidden_states: torch.FloatTensor):
        if self.is_empty:
            shape = list(hidden_states.shape)
            shape[-2] = shape[-2] // self.scale
            shape[-1] = shape[-1] // self.scale
            return torch.zeros(shape).to(hidden_states)
        
        if len(hidden_states.shape) == 5:
            B, F, C, H, W = hidden_states.shape
            hidden_states = rearrange(hidden_states, "B F C H W -> (B F) C H W")
            hidden_states = self.down_scale(hidden_states)
            hidden_states = self.out(hidden_states)
            hidden_states = rearrange(hidden_states, "(B F) C H W -> B F C H W", F=F)
        else:
            hidden_states = self.down_scale(hidden_states)
            hidden_states = self.out(hidden_states)
        return hidden_states




    


class ControlNeXtSVDModel(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 3,
        down_block_types: Tuple[str] = (
            "Block2D",
            "Block2D",
            "Block2D",
            "Block2D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = 25,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels : Optional[Tuple[int, ...]] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]
        time_embed_dim = 256
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)
        in_channels = [128, 128, 256]
        out_channels = [128, 256, 256]
        groups = [4, 8, 8]

        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.down_res = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(len(in_channels)):
            self.down_res.append(
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i]
                ),
            )
            self.down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name="op",
                )
            )
        
        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels[-1]),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GroupNorm(8, out_channels[-1]),
        ))
        self.mid_convs.append(
            nn.Conv2d(
            in_channels=out_channels[-1],
            out_channels=1280,
            kernel_size=1,
            stride=1,
        ))


        # self.scale_linear = nn.Linear(time_embed_dim, time_embed_dim)
        # self.time_out_scale = nn.Linear(time_embed_dim, 1, bias=False)
        # nn.init.zeros_(self.time_out_scale.weight)
        # self.out = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        # for p in self.out.parameters():
        #     nn.init.zeros_(p)
        self.scale = nn.Parameter(torch.tensor(1.))
        # self.scale = nn.Parameter(torch.tensor(0.8766))


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,
        added_time_ids: torch.Tensor = None,
        controlnext_cond: torch.FloatTensor = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        guess_mode: bool = False,
        conditioning_scale: float = 1.0,
    ) -> Union[ControlNeXtOutput, Tuple]:
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb_batch = self.time_embedding(t_emb)

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb_batch.repeat_interleave(num_frames, dim=0)

        sample = self.embedding(sample)

        for res, downsample in zip(self.down_res, self.down_sample):
            sample = res(sample, emb)
            sample = downsample(sample, emb)
        
        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)
        
        # sample = self.outt(sample)
        # sample = rearrange(sample, "(b f) c h w -> b f c h w", f=num_frames)

        # scale = self.scale_linear(emb_batch)
        # scale = self.time_out_scale(scale)
        # scale = rearrange(scale, "b 1 -> b 1 1 1 1")

        return {
            'output': sample,
            'scale': self.scale,
        }
    

    

    
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

