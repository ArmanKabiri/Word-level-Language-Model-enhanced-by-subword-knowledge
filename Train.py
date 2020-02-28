# Author: Arman Kabiri
# Date: Feb. 18, 2020
# Email: Arman.Kabiri94@gmail.com


import argparse
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from CorpusReader import CorpusReader
from Dictionary import Dictionary
from EmbeddingsLoader import EmbeddingsLoader
from Lang_Model import LanguageModel


def main():
    parser = argparse.ArgumentParser(description='LSTM Language Model')
    parser.add_argument('--corpus_file', type=str, help='location of the data corpus')
    parser.add_argument('--embeddings_file', type=str)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--dropout_probablity', type=float)
    parser.add_argument('--embeddings_dim', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--bidirectional_model', action='store_true')
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--freez_embeddings', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --gpu")
    else:
        if args.gpu:
            print("You do not have a GPU device, so you should run CPU without --gpu option.")
            exit()

    device = torch.device("cuda" if args.gpu else "cpu")
    torch.manual_seed(100)
    corpus_reader = CorpusReader(args.corpus_file, 10000000)

    logging.info("Generating Dictionaries")
    dictionary = Dictionary(corpus_reader)
    dictionary.build_dictionary()

    logging.info("Loading Embeddings")

    embeddings_matrix = None
    if args.embeddings_file is not None:
        emb_loader = EmbeddingsLoader()
        embeddings_matrix = emb_loader.get_embeddings_matrix(args.embeddings_file, dictionary, args.embeddings_dim)

    model = LanguageModel(n_layers=args.n_layers, hidden_size=args.hidden_size, n_vocab=dictionary.get_dic_size(),
                          input_size=args.embeddings_dim, dropout=args.dropout_probablity,
                          bidirectional=args.bidirectional_model, pret_emb_matrix=embeddings_matrix,
                          freez_emb=args.freez_embeddings, tie_weights=args.tie_weights, use_gpu=args.gpu)

    ###############
    total_param = []
    for p in model.parameters():
        total_param.append(int(p.numel()))
    print(total_param)
    print(sum(total_param))
    ###############

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # put it into train mode.
    model.train()

    if args.gpu:
        model.cuda()

    print("Training starts ...")
    step = 0
    for i in range(args.epochs):

        print(f"Epoch {i + 1}:")

        batch_generator = corpus_reader.batchify(dictionary, args.batch_size, args.seq_len)
        hidden = model.init_hidden(args.batch_size)

        for x, y in tqdm(batch_generator):

            step += 1
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            if args.gpu:
                x = x.cuda()
                y = y.cuda()

            # TODO: Not sure about this line
            hidden = model.init_hidden(args.batch_size)
            model.zero_grad()

            y_hat, hidden = model.forward(x, hidden)
            # TODO: Should do some reshaping : targets.view(batch_size*seq_len).long()

            loss = criterion.forward(y_hat.view(-1, dictionary.get_dic_size()),
                                     y.reshape(args.batch_size * args.seq_len).long())
            loss.backward()

            # TODO: POSSIBLE EXPLODING GRADIENT PROBLEM! -> CLIP JUST IN CASE :
            # nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)

            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {i}, Step {step},     Loss = {loss}")


if __name__ == '__main__':
    main()
