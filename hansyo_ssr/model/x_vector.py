#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""


import torch
import torch.nn as nn
# from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from hansyo_ssr.model.tdnn import TDNN


class X_vector(nn.Module):
    def __init__(
        self,
        input_dim=80,
        num_classes=2,
    ):
        super(X_vector, self).__init__()
        # x_vec_dim = 512
        x_vec_dim = 64
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=x_vec_dim, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=x_vec_dim, output_dim=x_vec_dim, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=x_vec_dim, output_dim=x_vec_dim, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=x_vec_dim, output_dim=x_vec_dim, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=x_vec_dim, output_dim=x_vec_dim, context_size=1, dilation=3, dropout_p=0.5)
        # Frame levelPooling
        self.segment6 = nn.Linear(x_vec_dim*2, x_vec_dim)
        self.segment7 = nn.Linear(x_vec_dim, x_vec_dim)
        self.output = nn.Linear(x_vec_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, voice_a, voice_b):
        voice_a_output, voice_a_pred, voice_a_x_vec = self._forward_single(voice_a)
        voice_b_output, voice_b_pred, voice_b_x_vec = self._forward_single(voice_b)
        voice_distance = self.cosine_similarity(voice_a_x_vec, voice_b_x_vec)
        return (
            (voice_a_output, voice_a_pred, voice_a_x_vec),
            (voice_b_output, voice_b_pred, voice_b_x_vec),
            voice_distance,
        )

    def _forward_single(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        # return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        # Stat Pool
        mean = torch.mean(tdnn5_out, 1)
        std = torch.std(tdnn5_out, 1)
        stat_pooling = torch.cat((mean, std), 1)
        # Linear
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)

        # sigmoid -> 0 ~ 1
        # x_vec = torch.sigmoid(x_vec)

        # # NOTE: L2 normalize
        # x_vec_norm = torch.norm(x_vec, dim=1, keepdim=True)
        # x_vec = torch.div(x_vec, x_vec_norm)

        output = self.output(x_vec)
        predictions = self.softmax(output)
        return output, predictions, x_vec
