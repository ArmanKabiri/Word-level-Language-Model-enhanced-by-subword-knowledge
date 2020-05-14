# Author: Arman Kabiri
# Date: April. 21, 2020
# Email: Arman.Kabiri94@fmail.com
######################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLevelCNNNetwork(nn.Module):

    def __init__(self, num_char, char_emb_dim, cnn_kernels, use_gpu):

        super(CharLevelCNNNetwork, self).__init__()
        self.char_emb_dim = char_emb_dim

        # char embedding layer
        self.char_embed = nn.Embedding(num_char, char_emb_dim)

        # list of tuples: (the number of filter, width)
        self.cnn_kernels = cnn_kernels

        # convolutions of filters with different sizes
        self.convolutions = nn.ModuleList([
            nn.Conv2d(
                1,  # in_channel
                out_channel,  # out_channel
                kernel_size=(char_emb_dim, filter_width),  # (height, width)
                bias=True
            ) for out_channel, filter_width in self.cnn_kernels
        ])

        self.highway_input_dim = sum([x for x, y in self.cnn_kernels])
        self.output_tensor_dim = self.highway_input_dim

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        if use_gpu:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.highway1 = self.highway1.cuda()
            self.highway2 = self.highway2.cuda()
            self.char_embed = self.char_embed.cuda()
            self.batch_norm = self.batch_norm.cuda()

    def forward(self, x):
        """
        :param x: Input variable of Tensor with shape [batch_size, seq_len, max_word_len+2]
        :return: Variable of Tensor with shape [batch_size, seq_len, total_num_filters]
        """

        batch_size = x.size()[0]
        seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [batch_size*seq_len, max_word_len+2]

        x = self.char_embed(x)
        # [batch_size*seq_len, max_word_len+2, char_emb_dim]

        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        # [batch_size*seq_len, 1, char_emb_dim, max_word_len+2]

        x = self.conv_layers(x)
        # [batch_size*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [batch_size*seq_len, total_num_filters]

        x = self.highway1(x)
        x = self.highway2(x)
        # [batch_size*seq_len, total_num_filters]

        x = x.contiguous().view(batch_size, seq_len, -1)
        # [batch_size, seq_len, total_num_filters]

        return x

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            # Max Pooling Over Time:
            batch_size = feature_map.size()[0]
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)
            chosen = chosen.squeeze()
            chosen = chosen.view((batch_size, -1))
            # (batch_size, out_channel)
            chosen_list.append(chosen)

        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)


class Highway(nn.Module):
    """Highway network"""

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1 - t, x)
