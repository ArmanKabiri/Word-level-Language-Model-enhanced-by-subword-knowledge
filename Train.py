# Author: Arman Kabiri
# Date: Feb. 18, 2020
# Email: Arman.Kabiri94@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gensim
import logging
from tqdm import tqdm
import typing
import argparse
from EmbeddingsLoader import EmbeddingsLoader
from Lang_Model import LanguageModel
from CorpusReader import CorpusReader
from Dictionary import Dictionary


def main():
    parser = argparse.ArgumentParser(description='LSTM Language Model')
    parser.add_argument('--corpus_file', type=str, help='location of the data corpus')
    parser.add_argument('--embeddings_file', type=str)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--dropout_probablity', type=float)
    parser.add_argument('--bidirectional_model', type=bool)
    parser.add_argument('--tie_weights', type=bool)
    parser.add_argument('--trainable_embeddings', type=bool)
    parser.add_argument('--embeddings_dim', type=int)
    parser.add_argument('--gpu', type=bool)

    args = parser.parse_args()

    corpus_reader = CorpusReader(args.corpus_file, 1000000)

    logging.info("Generating Dictionaries")
    dictionary = Dictionary(corpus_reader)
    dictionary.build_dictionary()

    logging.info("Loading Embeddings")
    emb_loader = EmbeddingsLoader()
    embeddings_matrix = emb_loader.get_embeddings_matrix(args.embeddings_file, dictionary, args.embeddings_dim)

    model = LanguageModel(n_layers=args.n_layers, hidden_size=args.hidden_size, n_vocab=dictionary.get_dic_size(),
                          input_size=args.embeddings_dim, dropout=args.dropout_probablity,
                          bidirectional=args.bidirectional_model, pret_emb_matrix=embeddings_matrix,
                          trainable_emb=args.trainable_embeddings, tie_weights=args.tie_weights, use_gpu=args.gpu)

    ###############
    total_param = []
    for p in model.parameters():
        total_param.append(int(p.numel()))
    print(total_param)
    print(sum(total_param))
    ###############

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Network
    epochs = 50
    batch_size = 128
    seq_len = 100

    # put it into train mode.
    model.train()

    if args.gpu:
        model.cuda()

    logging.info("Training starts ...")
    step = 0
    for i in range(epochs):

        logging.info(f"Epoch {i+1}:")

        batch_generator = corpus_reader.batchify(dictionary, batch_size, seq_len)
        hidden = model.init_hidden(batch_size)

        for x, y in batch_generator:

            step += 1
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            if args.gpu:
                x = x.cuda()
                y = y.cuda()

            # TODO: Not sure about this line
            model.init_hidden(batch_size)
            model.zero_grad()

            y_hat, hidden = model.forward(x, hidden)
            # TODO: Should do some reshaping : targets.view(batch_size*seq_len).long()
            loss = criterion.forward(y, y_hat)
            loss.backward()

            # TODO: POSSIBLE EXPLODING GRADIENT PROBLEM! -> CLIP JUST IN CASE :
            # nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)

            optimizer.step()

            if step % 25 == 0:
                print(f"Epoch {i}, Step {step},     Loss = {loss}")


if __name__ == '__main__':
    main()
