import torch
import torch.nn as nn
from ..models.backbone.audio_encoders import AudioEncoder
from realnvp.vector_realnvp import RealNVP, make_realnvp_masks


class AudioRealNVPModel(nn.Module):

    def __init__(self, config):
        super(AudioRealNVPModel, self).__init__()

        # self.l2 = config.training.l2
        joint_embed = config.joint_embed

        if config.cnn_encoder.model == 'cnn14':
            self.encoder = AudioEncoder(config=config)
        else:
            print("wrong cnn encoder")

        if config.cnn_encoder.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(2048, joint_embed * 2),
            nn.ReLU(),
            nn.Linear(joint_embed * 2, joint_embed)
        )
        masks = make_realnvp_masks(joint_embed, config.real_nvp.n_flow_layers)
        self.flow = RealNVP(masks=masks, hidden_size=config.real_nvp.flow_hidden)

    def _encode(self, waveform: torch.Tensor, **enc_kwargs) -> torch.Tensor:
        x_raw = self.encoder(waveform, **enc_kwargs)  # expects (B, D_raw)
        x = self.proj(x_raw)                          # -> (B, emb_dim)
        return x

    def forward(self, waveform: torch.Tensor, **enc_kwargs):
        """waveform -> x -> (z, log_det)."""
        x = self._encode(waveform, **enc_kwargs)
        z, log_det = self.flow.forward(x)
        return z, log_det

    def log_prob(self, waveform: torch.Tensor, **enc_kwargs) -> torch.Tensor:
        x = self._encode(waveform, **enc_kwargs)
        return self.flow.log_probability(x)

    def nll(self, waveform: torch.Tensor, **enc_kwargs) -> torch.Tensor:
        return -self.log_prob(waveform, **enc_kwargs)
