#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..utils.utils import l2norm
from backbone.audio_encoders import AudioEncoder
from backbone.text_encoder import BertEncoder
from backbone.bert_config import MODELS



class EncoderModel(nn.Module):

    def __init__(self, config):
        super(EncoderModel, self).__init__()

        self.l2 = config.training.l2
        joint_embed = config.joint_embed

        self.audio_enc = AudioEncoder(config)

        if config.cnn_encoder.model == 'ResNet38' or config.cnn_encoder.model == 'Cnn14':
            self.audio_linear = nn.Sequential(
                nn.Linear(2048, joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )

        # self.audio_gated_linear = nn.Linear(joint_embed, joint_embed)
        if config.text_encoder == 'bert':
            self.text_enc = BertEncoder(config)
            bert_type = config.bert_encoder.type
            self.text_linear = nn.Sequential(
                nn.Linear(MODELS[bert_type][2], joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )


    def encode_audio(self, audios):
        return self.audio_enc(audios)

    def encode_text(self, captions):
        return self.text_enc(captions)

    def forward(self, audios, captions):

        audio_encoded = self.encode_audio(audios)     # batch x channel
        caption_encoded = self.encode_text(captions)

        audio_embed = self.audio_linear(audio_encoded)

        caption_embed = self.text_linear(caption_encoded)

        if self.l2:
            # apply l2-norm on the embeddings
            audio_embed = l2norm(audio_embed)
            caption_embed = l2norm(caption_embed)

        return audio_embed, caption_embed