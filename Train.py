# Author: Arman Kabiri
# Date: Feb. 18, 2020
# Email: Arman.Kabiri94@gmail.com

# %%
from DictionaryCharacter import DictionaryCharacter

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

# %%

if IN_COLAB:
    from google.colab import drive

    drive.mount('/gdrive')

# %%

if IN_COLAB:
    import os

    os.chdir('/gdrive/My Drive/NLP_Stuff/My_Language_Model')

# %%


# %%

import os.path as path
import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import GPUtil

from CorpusReader import CorpusReader
from DictionaryWord import DictionaryWord
from EmbeddingsLoader import EmbeddingsLoader
from Lang_Model import LanguageModel

# %%

import tensorflow as tf
from tensorflow import summary

# %%

# !rm - rf logs
current_time = str(datetime.datetime.now().timestamp())
train_log_dir = 'logs/tensorboard/train/' + current_time
test_log_dir = 'logs/tensorboard/test/' + current_time
train_summary_writer = summary.create_file_writer(train_log_dir)
test_summary_writer = summary.create_file_writer(test_log_dir)

# %% md

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_mnist_experiment_1')


# %%

class Args:
    ## Corpus Files:
    #   corpus_train_file='Data/WestburyLab.Wikipedia.Corpus_AdagramTokenized.txt'
    corpus_train_file = 'Data/corpus-test.txt'
    corpus_valid_file = ''

    ## Word Embeddings
    word_embeddings_file = 'Data/English_Wiki_1Billion_embeddings.bin'
    word_embeddings_file = None
    word_embeddings_dim = 300

    output_model_path = 'Data/model.bin'
    output_id2word_path = 'Data/id2word.txt'
    output_word2id_path = 'Data/word2id.txt'

    # Training HyperParameters:
    batch_size = 32
    seq_len = 10
    epochs = 5
    lr = 0.01
    seed = 120
    clip_grad = 5
    dropout_probablity = .25

    ## Network Properties:
    n_lstm_layers = 1
    hidden_size = 300

    ## Character Feature Detector:
    features_level = ['character']
    character_embedding_dim = 10

    ## Flags:
    bidirectional_model = False
    tie_weights = False
    freez_embeddings = False
    gpu = False

    ## Debug
    print_steps = 10


args = Args()

# %%

torch.cuda.is_available()


# %%

def train(corpus_train_reader, dictionary_word, dictionary_char, model, optimizer, criterion, epoch_number):
    batch_generator = corpus_train_reader.batchify(dictionary_word, args.batch_size, args.seq_len)
    hidden = model.init_hidden(args.batch_size)

    step = 0
    if args.gpu:
        GPU_device_logger = GPUtil.getGPUs()[0]

    # Shape of X : (batch_size, seq_len)
    for x_word, y_word in tqdm(batch_generator):

        step += 1

        x_char = None
        if 'character' in args.features_level:
            x_char = dictionary_char.encode_batch(x_word)
            x_char = torch.from_numpy(x_char)

        x_word = torch.from_numpy(x_word)
        y_word = torch.from_numpy(y_word)

        if args.gpu:
            x_word = x_word.cuda()
            y_word = y_word.cuda()
            x_char = x_char.cuda() if 'character' in args.features_level else x_char

        hidden = detach_hidden(hidden)
        model.zero_grad()

        y_hat, hidden = model.forward(x_word, x_char, hidden)

        loss = criterion.forward(y_hat.view(-1, dictionary_word.get_dic_size()),
                                 y_word.reshape(args.batch_size * args.seq_len).long())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

        optimizer.step()

        if step % args.print_steps == 0:
            with train_summary_writer.as_default():
                args.globaliter += 1
                tf.summary.scalar('loss', loss.item(), step=args.globaliter)
                tf.summary.text('progress',
                                f"Epoch {epoch_number} progress = {corpus_train_reader.get_progress()}% ,  Loss = {loss.item()} ,  PPL = {np.exp(loss.item())}",
                                step=args.globaliter)
                if args.gpu:
                    tf.summary.text('GPU Summary', 'Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(
                        GPU_device_logger.memoryFree, GPU_device_logger.memoryTotal,
                        GPU_device_logger.memoryUtil * 100), step=args.globaliter)


# %%

def detach_hidden(hidden: tuple):
    return tuple(v.detach() for v in hidden)


# %%

def save_dictionary(dictionary: DictionaryWord, output_id2word_path, output_word2id_path):
    with open(output_word2id_path, 'w') as file:
        for word, word_id in dictionary.word2id.items():
            if '\t' in word:
                exit()
            file.write(f"{word}\t{word_id}\n")

    with open(output_id2word_path, 'w') as file:
        for word in dictionary.id2word:
            file.write(f"{word}\n")


# %%

def main():
    torch.set_num_threads(8)

    if torch.cuda.is_available():
        if not args.gpu:
            print("WARNING: You have a CUDA device, so you should probably run with --gpu")
    else:
        if args.gpu:
            print("You do not have a GPU device, so you should run CPU without --gpu option.")
            exit()

    if 'word' not in args.features_level and 'character' not in args.features_level:
        exit("features_level argument is empty. It should include at least one of [word,character] items.")

    torch.manual_seed(args.seed)
    corpus_train_reader = CorpusReader(args.corpus_train_file, 1000000)  # 100MB

    dictionary_word = DictionaryWord()
    dictionary_char = None

    if 'character' in args.features_level:
        dictionary_char = DictionaryCharacter()

    # Load the pre-trained Model for fine-tuning
    if path.exists(args.output_model_path):
        print("Loading Dictionaries...")
        dictionary_word.load_dictionary(id2word_filepath=args.output_id2word_path,
                                        word2id_filepath=args.output_word2id_path)
        # TODO: Load Char Dictionary

        print("Loading pre-trained Model...")
        model = LanguageModel(path_to_pretrained_model=args.output_model_path, use_gpu=args.gpu)

    # Initialize the model
    else:
        print("Generating Dictionaries...")
        dictionary_word.build_dictionary(corpus_train_reader)
        if 'character' in args.features_level:
            dictionary_char.build_dictionary(dictionary_word)

        print("Saving Dictionary...")
        save_dictionary(dictionary_word, args.output_id2word_path, args.output_word2id_path)
        # TODO: Save Character Dictionary

        embeddings_matrix = None
        if args.word_embeddings_file is not None and 'word' in args.features_level:
            print("Loading Embeddings...")
            emb_loader = EmbeddingsLoader()
            embeddings_matrix = emb_loader.get_embeddings_matrix(args.word_embeddings_file, dictionary_word,
                                                                 args.word_embeddings_dim)

        model = LanguageModel(n_layers=args.n_lstm_layers, hidden_size=args.hidden_size,
                              n_vocab=dictionary_word.get_dic_size(), word_emb_dim=args.word_embeddings_dim,
                              n_chars=dictionary_char.dic_size if 'character' in args.features_level else None,
                              char_emb_dim=args.character_embedding_dim if 'character' in args.features_level else None,
                              features_level=args.features_level,
                              dropout_prob=args.dropout_probablity, bidirectional=args.bidirectional_model,
                              pret_emb_matrix=embeddings_matrix,
                              freez_emb=args.freez_embeddings, tie_weights=args.tie_weights, use_gpu=args.gpu)

    ###############
    print("Parametes:")
    total_param = []
    for name, param in model.named_parameters():
        total_param.append(int(param.numel()))
        print(f"{name} ({'trainable' if param.requires_grad else 'non-trainable'}) : {int(param.numel())}")
    print(f"Number of Parametes: {sum(total_param)}\n")
    ###############

    # put it into train mode.
    model.train()
    if args.gpu:
        model.cuda()

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training Model
    print("Training Model...")
    args.globaliter = 0
    for i in range(1, args.epochs + 1):
        print(f"Epoch {i}:")
        train(corpus_train_reader, dictionary_word, dictionary_char, model, optimizer, criterion, epoch_number=i)
        print(f"Saving Model at epoch {i}...\n")
        model.save_model(args.output_model_path)


# %%
# %tensorboard --logdir logs/tensorboard
# %%

main()
