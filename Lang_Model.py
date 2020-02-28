# Author: Arman Kabiri
# Date: Feb. 27, 2020
# Email: Arman.Kabiri94@fmail.com
######################################3

import numpy as np
import torch
import torch.nn as nn


class LanguageModel(nn.Module):

    def __init__(self, n_layers: int, hidden_size: int, n_vocab: int, input_size: int, dropout: float,
                 bidirectional: bool, pret_emb_matrix: np.array = None,
                 freez_emb: bool = True, tie_weights: bool = False, use_gpu=False):

        super().__init__()

        if pret_emb_matrix is not None:
            assert n_vocab == pret_emb_matrix.shape[0] and input_size == pret_emb_matrix.shape[1]

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.n_layers = n_layers

        # Initializing Layers

        self.embedding = nn.Embedding(n_vocab, input_size)

        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=n_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.decoder = nn.Linear(hidden_size, n_vocab)
        # TODO: Doesn't it need a Softmax layer?
        # Network Initialization
        initrange = 0.1
        self.init_weights(pret_emb_matrix, freez_emb, tie_weights, initrange)

    def init_weights(self, pret_emb_matrix, freez_emb, tie_weights, initrange=0.1):

        if tie_weights and freez_emb:
            raise ValueError('tie_weights and trainable_emb flags should be used in a compatible way.')

        if pret_emb_matrix is None and not freez_emb is False:
            raise ValueError('When pre-trained embeddings are not given, weights should be trainable.')

        if pret_emb_matrix is not None:

            pret_emb_matrix = torch.from_numpy(pret_emb_matrix)
            if self.use_gpu:
                pret_emb_matrix = pret_emb_matrix.cuda()

            self.embedding.from_pretrained(pret_emb_matrix, freez_emb)
            # self.embedding.load_state_dict({'weight': pret_emb_matrix})
            # self.embedding.weight.requires_grad = trainable_emb
        else:
            self.embedding.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()

        if tie_weights:

            if self.hidden_size != self.input_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to input_size')

            self.decoder.weight = self.embedding.weight

        else:
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):

        output = self.embedding(x)

        output = self.dropout(output)

        output, hidden = self.rnn(output, hidden)

        output = self.dropout(output)

        # Not Sure why---------------------
        # output = output.contiguous().view(-1, self.hidden_size)

        output = self.decoder(output)

        return output, hidden

    def init_hidden(self, batch_size: int):

        num_directions = 2 if self.bidirectional else 1

        if self.use_gpu:
            hidden_state = (torch.zeros(self.n_layers * num_directions, batch_size, self.hidden_size).cuda(),
                            torch.zeros(self.n_layers * num_directions, batch_size, self.hidden_size).cuda())
        else:
            hidden_state = (torch.zeros(self.n_layers * num_directions, batch_size, self.hidden_size),
                            torch.zeros(self.n_layers * num_directions, batch_size, self.hidden_size))

        return hidden_state

# ### Shapes
# **input** of shape `(seq_len, batch, input_size)
# **h_0** of shape `(num_layers * num_directions, batch, hidden_size)
# **c_0** of shape `(num_layers * num_directions, batch, hidden_size)
