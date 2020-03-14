# Author: Arman Kabiri
# Date: Feb. 27, 2020
# Email: Arman.Kabiri94@fmail.com
######################################

import numpy as np
import torch
import torch.nn as nn


class LanguageModel(nn.Module):

    def __init__(self, n_layers: int = 2, hidden_size: int = 300, n_vocab: int = 10000, input_size: int = 300,
                 dropout_prob: float = 0.25, bidirectional: bool = False, pret_emb_matrix: np.array = None,
                 freez_emb: bool = True, tie_weights: bool = False, use_gpu=False, path_to_pretrained_model=None):

        super().__init__()

        if path_to_pretrained_model is not None:
            self.use_gpu = use_gpu
            self.load_model(path_to_pretrained_model)

        else:

            if pret_emb_matrix is not None:
                assert n_vocab == pret_emb_matrix.shape[0] and input_size == pret_emb_matrix.shape[1]

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.use_gpu = use_gpu
            self.bidirectional = bidirectional
            self.n_layers = n_layers
            self.n_vocab = n_vocab
            self.freez_emb = freez_emb
            self.tie_weights = tie_weights
            self.dropout_prob = dropout_prob

            self.__build_model()

            initrange = 0.1
            self.init_weights(pret_emb_matrix, freez_emb, tie_weights, initrange)

    def __build_model(self):
        self.embedding = nn.Embedding(self.n_vocab, self.input_size)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                           dropout=self.dropout_prob, bidirectional=self.bidirectional, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, self.n_vocab)

    def load_model(self, modelpath: str):

        print("Loading Model from file...")
        loaded_parameters = torch.load(modelpath, map_location=torch.device('gpu' if self.use_gpu else 'cpu'))
        self.n_layers = loaded_parameters['n_layers']
        self.hidden_size = loaded_parameters['hidden_size']
        self.n_vocab = loaded_parameters['n_vocab']
        self.input_size = loaded_parameters['input_size']
        # TODO
        # self.dropout_prob = loaded_parameters['dropout_prob']
        self.dropout_prob = 0.25

        self.bidirectional = loaded_parameters['bidirectional']
        self.freez_emb = loaded_parameters['freez_emb']
        self.tie_weights = loaded_parameters['tie_weights']

        self.__build_model()

        self.load_state_dict(loaded_parameters['state_dict'])

    def init_weights(self, pret_emb_matrix, freez_emb, tie_weights, initrange=0.1):

        if tie_weights and freez_emb:
            raise ValueError('tie_weights and trainable_emb flags should be used in a compatible way.')

        if pret_emb_matrix is None and freez_emb is True:
            raise ValueError('When pre-trained embeddings are not given, weights should be trainable.')

        if pret_emb_matrix is not None:

            pret_emb_matrix = torch.from_numpy(pret_emb_matrix)
            if self.use_gpu:
                pret_emb_matrix = pret_emb_matrix.cuda()

            self.embedding.from_pretrained(pret_emb_matrix, freez_emb)
            # or
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

    def save_model(self, file_path):

        data_to_save = {
            'state_dict': self.state_dict(),
            'n_layers': self.n_layers,
            'hidden_size': self.hidden_size,
            'n_vocab': self.n_vocab,
            'input_size': self.input_size,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'freez_emb': self.freez_emb,
            'tie_weights': self.tie_weights
        }

        torch.save(data_to_save, file_path)

# ### Shapes
# **input** of shape `(seq_len, batch, input_size)
# **h_0** of shape `(num_layers * num_directions, batch, hidden_size)
# **c_0** of shape `(num_layers * num_directions, batch, hidden_size)
