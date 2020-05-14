# Author: Arman Kabiri
# Date: Feb. 27, 2020
# Email: Arman.Kabiri94@fmail.com
######################################

import numpy as np
import torch
import torch.nn as nn

from CharLevelCNNNetwork import CharLevelCNNNetwork


class LanguageModel(nn.Module):
    word_emb_dim = None
    hidden_size = None
    features_level = None
    n_chars = None
    char_emb_dim = None
    cnn_kernels = None
    use_gpu = None
    bidirectional = None
    n_layers = None
    n_vocab = None
    freez_emb = None
    tie_weights = None
    dropout_prob = None
    globaliter = 0
    current_in_progress_epoch = 0

    object_is_instantiated = False

    optimizer_state = None

    def __init__(self):
        super().__init__()
        self.object_is_instantiated = False

    @classmethod
    def instantiate_model(cls, n_layers: int, hidden_size: int, n_vocab: int,
                          word_emb_dim: int, features_level: list, dropout_prob: float, bidirectional: bool,
                          use_gpu: bool, freez_emb: bool, tie_weights: bool,
                          pret_emb_matrix: np.ndarray = None, n_chars: int = None,
                          char_emb_dim: int = None, cnn_kernels: list = None,
                          ):

        # <-----------Assertion---------->
        if pret_emb_matrix is not None:
            assert n_vocab == pret_emb_matrix.shape[0] and word_emb_dim == pret_emb_matrix.shape[1]
        if tie_weights and freez_emb:
            raise ValueError('tie_weights and trainable_emb flags should be used in a compatible way.')
        if pret_emb_matrix is None and freez_emb is True:
            raise ValueError('When pre-trained embeddings are not given, weights should be trainable.')
        # </-----------Assertion---------->

        # <-----------Storing arguments---------->
        model_object = cls()
        model_object.word_emb_dim = word_emb_dim
        model_object.hidden_size = hidden_size
        model_object.features_level = features_level
        model_object.n_chars = n_chars
        model_object.char_emb_dim = char_emb_dim
        model_object.cnn_kernels = cnn_kernels
        model_object.use_gpu = use_gpu
        model_object.bidirectional = bidirectional
        model_object.n_layers = n_layers
        model_object.n_vocab = n_vocab
        model_object.freez_emb = freez_emb
        model_object.tie_weights = tie_weights
        model_object.dropout_prob = dropout_prob
        # </-----------Storing arguments---------->

        model_object.__build_model()

        init_range = 0.1
        model_object.__init_weights(pret_emb_matrix, freez_emb, tie_weights, init_range)

        model_object.object_is_initiated = True

        return model_object

    @classmethod
    def load_model(cls, use_gpu, path_to_pretrained_model):

        print("Loading Model from file...")

        model_object = cls()
        model_object.use_gpu = use_gpu

        loaded_parameters = torch.load(path_to_pretrained_model,
                                       map_location=torch.device('cuda' if use_gpu else 'cpu'))

        model_object.n_layers = loaded_parameters['n_layers']
        model_object.hidden_size = loaded_parameters['hidden_size']
        model_object.n_vocab = loaded_parameters['n_vocab']
        model_object.word_emb_dim = loaded_parameters['word_emb_dim']
        model_object.n_chars = loaded_parameters['n_chars']
        model_object.char_emb_dim = loaded_parameters['char_emb_dim']
        model_object.features_level = loaded_parameters['features_level']
        model_object.cnn_kernels = loaded_parameters['cnn_kernels']
        model_object.dropout_prob = loaded_parameters['dropout_prob']
        model_object.bidirectional = loaded_parameters['bidirectional']
        model_object.freez_emb = loaded_parameters['freez_emb']
        model_object.tie_weights = loaded_parameters['tie_weights']
        model_object.globaliter = loaded_parameters['globaliter']
        model_object.current_in_progress_epoch = loaded_parameters['current_in_progress_epoch']

        model_object.__build_model()

        model_object.load_state_dict(loaded_parameters['state_dict'])
        try:
            model_object.optimizer_state = loaded_parameters['optimizer_state']
        except:
            print('Optimizer state is not loaded.')

        model_object.object_is_initiated = True

        return model_object

    def save_model(self, file_path, optimizer):

        data_to_save = {
            'state_dict': self.state_dict(),
            'n_layers': self.n_layers,
            'hidden_size': self.hidden_size,
            'n_vocab': self.n_vocab,
            'word_emb_dim': self.word_emb_dim,
            'n_chars': self.n_chars,
            'char_emb_dim': self.char_emb_dim,
            'features_level': self.features_level,
            'cnn_kernels': self.cnn_kernels,
            'dropout_prob': self.dropout_prob,
            'bidirectional': self.bidirectional,
            'freez_emb': self.freez_emb,
            'tie_weights': self.tie_weights,
            'globaliter': self.globaliter,
            'current_in_progress_epoch': self.current_in_progress_epoch,
            'optimizer_state': optimizer.state_dict()
        }

        torch.save(data_to_save, file_path)

    def __build_model(self):

        self.lstm_input_size = 0
        if 'character' in self.features_level:
            self.char_CNN_network = CharLevelCNNNetwork(self.n_chars, self.char_emb_dim, self.cnn_kernels, self.use_gpu)
            self.lstm_input_size += self.char_CNN_network.output_tensor_dim
        if 'word' in self.features_level:
            self.word_embedding_layer = nn.Embedding(self.n_vocab, self.word_emb_dim)
            self.lstm_input_size += self.word_emb_dim

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            dropout=self.dropout_prob, bidirectional=self.bidirectional, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, self.n_vocab)

    def __init_weights(self, pret_emb_matrix, freez_emb, tie_weights, init_range=0.1):

        # <-------------Word Embeddings------------->
        if 'word' in self.features_level:

            if pret_emb_matrix is not None:
                pret_emb_matrix = torch.from_numpy(pret_emb_matrix)
                if self.use_gpu:
                    pret_emb_matrix = pret_emb_matrix.cuda()

                self.word_embedding_layer.from_pretrained(pret_emb_matrix, freez_emb)

            else:
                self.word_embedding_layer.weight.data.uniform_(-init_range, init_range)
        # </-------------Word Embeddings------------->

        # <-------------Decoder------------->
        if tie_weights and 'word' in self.features_level and 'character' not in self.features_level:

            if self.hidden_size != self.word_emb_dim:
                raise ValueError('When using the tied flag, hidden_size must be equal to input_size')

            self.decoder.weight = self.word_embedding_layer.weight

        else:
            self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        # </-------------Decoder------------->

        # Comment: LSTM and CONV weights and biases are not initialized. default initialization in used.

    def forward(self, x_word, x_char, hidden):
        """
        :param x_word: input variable Tensor of shape (batch_size, seq_len)
        This parameter represent word-level word representations
        :param x_char: input variable Tensor of shape (batch_size, seq_len, Max_word_len + 2)
        This parameter represent char-level word representations
        :param hidden:
        :return:
        """

        # get word representation
        if 'word' in self.features_level:
            x_word = self.word_embedding_layer(x_word)
            x_word = self.dropout(x_word)
            # x_word shape : (batch_size, seq_len, emb_dim)

        # get character-level word features
        if 'character' in self.features_level:
            x_char = self.char_CNN_network(x_char.long())
            # x_char shape : (batch_size,seq_len,total_n_of_kernels)

        # concat word-level and character-level representations:
        x = torch.cat((x_word, x_char), 2) if len(set(self.features_level).intersection({'word', 'character'})) == 2 \
            else x_word if 'word' in self.features_level else x_char if 'character' in self.features_level else None

        # x is a Tensor of shape (batch_size, seq_len, lstm_input_size)
        x, hidden = self.lstm(x, hidden)

        # x is a Tensor of shape (batch_size, seq_len, lstm_hidden_size)
        x = self.dropout(x)

        x = self.decoder(x)

        # x is a Tensor of shape (batch_size, seq_len, vocab_size)
        return x, hidden

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
# **input** of shape `(batch, seq_len, input_size)
# **h_0** of shape `(num_layers * num_directions, batch, hidden_size)
# **c_0** of shape `(num_layers * num_directions, batch, hidden_size)
