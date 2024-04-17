# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn


class Tower(nn.Layer):
    def __init__(self,
                 input_dim: int,
                 dims=[128, 64, 32],
                 drop_prob=[0.1, 0.3, 0.3]):
        super(Tower, self).__init__()
        self.dims = dims
        self.drop_prob = drop_prob
        self.layer = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.ReLU(),
            nn.Dropout(drop_prob[0]),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(drop_prob[1]),
            nn.Linear(dims[1], dims[2]), nn.ReLU(), nn.Dropout(drop_prob[2]))

    def forward(self, x):
        x = paddle.flatten(x, 1)
        x = self.layer(x)
        return x


class Attention(nn.Layer):
    """Self-attention layer for click and purchase"""

    def __init__(self, dim=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias_attr=False)
        self.k_layer = nn.Linear(dim, dim, bias_attr=False)
        self.v_layer = nn.Linear(dim, dim, bias_attr=False)
        self.softmax = nn.Softmax(1)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = (Q * K).sum(-1) / (self.dim**0.5)
        a = self.softmax(a)
        outputs = (a.unsqueeze(-1) * V).sum(1)
        return outputs


class AITM(nn.Layer):
    def __init__(self,
                 feature_vocabulary,
                 embedding_size,
                 tower_dims=[128, 64, 32],
                 drop_prob=[0.1, 0.3, 0.3]):
        super(AITM, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.LayerList()
        self.__init_weight()

        self.tower_input_size = len(feature_vocabulary) * embedding_size
        self.ctr_tower = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.cvr_tower = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.ctcvr_tower = Tower(self.tower_input_size, tower_dims, drop_prob)
        # self.imputation_tower = Tower(self.tower_input_size, tower_dims, drop_prob)

        self.ctr_attention_layer = Attention(tower_dims[-1])

        self.ctr_info_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 32),
            nn.ReLU(), nn.Dropout(drop_prob[-1]))
        
        self.ctcvr_attention_layer = Attention(tower_dims[-1])

        self.ctcvr_info_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 32),
            nn.ReLU(), nn.Dropout(drop_prob[-1]))

        self.ctr_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        
        self.cvr_info_layer = nn.Sequential(
            nn.Linear(2*tower_dims[-1], 32),
            nn.ReLU(), nn.Dropout(drop_prob[-1]))
        
        self.cvr_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        
        self.ctcvr_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 1), nn.Sigmoid())

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            self.embedding_dict.append(emb)

    def forward(self, x):
        feature_embedding = []
        for i in range(len(x)):
            embed = self.embedding_dict[i](x[i])
            feature_embedding.append(embed)

        feature_embedding = paddle.concat(feature_embedding, 1)
        tower_ctr = self.ctr_tower(feature_embedding)
        tower_ctcvr = self.ctcvr_tower(feature_embedding)

        tower_cvr = paddle.unsqueeze(self.cvr_tower(feature_embedding), 1)

        ctr_info = paddle.unsqueeze(self.ctr_info_layer(tower_ctr), 1)

        ctr_ait = self.ctr_attention_layer(paddle.concat([tower_cvr, ctr_info], 1))

        ctr = paddle.squeeze(self.ctr_layer(tower_ctr), 1)
        ctcvr = paddle.squeeze(self.ctcvr_layer(tower_ctcvr), 1)

        ctcvr_info = paddle.unsqueeze(self.ctcvr_info_layer(tower_ctcvr), 1)

        ctcvr_ait = self.ctcvr_attention_layer(paddle.concat([tower_cvr, ctcvr_info], 1))


        cvr = paddle.squeeze(self.cvr_layer(self.cvr_info_layer(paddle.concat([ctr_ait, ctcvr_ait], 1))), 1)

        return ctr, ctcvr, cvr, cvr
