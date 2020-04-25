# %%

#### Author: Arman Kabiri
#### Date: Feb. 18, 2020
#### Email: Arman.Kabiri94@gmail.com


import os.path as path
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import GPUtil

from CorpusReader import CorpusReader
from DictionaryWord import DictionaryWord
from DictionaryCharacter import DictionaryCharacter
from EmbeddingsLoader import EmbeddingsLoader
from Lang_Model import LanguageModel

getattr(tqdm, '_instances', {}).clear()

# %%

parser = argparse.ArgumentParser(description='LSTM Language Model - Text Generator')
parser.add_argument('--corpus_train_file', type=str, default='Data/corpus-100mb-train.txt',
                    help='location of the corpus for training')
parser.add_argument('--corpus_dev_file', type=str, default='Data/corpus-100mb-dev.txt',
                    help='location of the corpus for evaluation')
parser.add_argument('--word_embeddings_file', type=str, default='Data/English_Wiki_1Billion_embeddings.bin',
                    help='If pretrained embeddings exist, load them here.')
parser.add_argument('--word_embeddings_dim', type=int, default=300, help='The dimension of the embeddings')

parser.add_argument('--output_model_path', type=str, default='Data/model.bin',
                    help='Path to save or load the trained model.')
parser.add_argument('--output_id2word_path', type=str, default='Data/id2word.txt',
                    help='Path to save or dictionary file (id2word)')
parser.add_argument('--output_word2id_path', type=str, default='Data/word2id.txt',
                    help='Path to save or dictionary file (word2id)')
parser.add_argument('--output_id2char_path', type=str, default='Data/id2char.txt',
                    help='Path to save or dictionary file (id2char)')
parser.add_argument('--output_char2id_path', type=str, default='Data/char2id.txt',
                    help='Path to save or dictionary file (char2id)')

parser.add_argument('--batch_size', type=int, default=256, help='Number of samples per batch.')
parser.add_argument('--seq_len', type=int, default=10, help='Length of the sequence for back propagation.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
parser.add_argument('--lr', type=int, default=0.001, help='Learning Rate')
parser.add_argument('--clip_grad', type=int, default=120,
                    help='Clip gradients during training to prevent exploding gradients.')
parser.add_argument('--dropout_probablity', type=int, default=256,
                    help='Dropout probablity applied on embeddings layer and LSTM layer.')
parser.add_argument('--seed', type=int, default=120, help='The seed for randomness')

parser.add_argument('--n_lstm_layers', type=int, default=2, help='Number of LSTM layers stacked on top of each other.')
parser.add_argument('--hidden_size', type=int, default=300, help='Number of hidden units in each LSTM layer')
parser.add_argument('--character_embedding_dim', type=int, default=10, help='The dimension of the character embeddings')
parser.add_argument('--cnn_kernels', type=str, nargs='+', default=['(10,2)', '(30,3)', '(40,4)', '(40,5)'],
                    help="CNN Kernels : (n_kernel,width_kernel). Sample input: (10,2) (30,3) (40,4) (40,5)."
                         "Notice the spaces and parentheses.")
parser.add_argument('--features_level', nargs='+', type=str, default=['word', 'character'],
                    help='Specify the level of features by which you want to represent your words.')

parser.add_argument('--bidirectional_model', action='store_true',
                    help='Use it if you want your LSTM to be bidirectional.')
parser.add_argument('--tie_weights', action='store_true',
                    help='Tie weights of the last decoder layer to the embeddings layer.')
parser.add_argument('--freez_embeddings', action='store_true',
                    help='Prevent the pretrained loaded embeddings from fine-tuning.')
parser.add_argument('--use_tensorboard', action='store_true', help='Turn on if you use Tensorboard')
parser.add_argument('--print_steps', type=int, default=50, help='evaluate and print every n iteration.')
parser.add_argument('--gpu', action='store_true', help='Turn it on if you have a GPU device.')


class Args:
    ## Corpus Files:
    # corpus_train_file='Data/WestburyLab.Wikipedia.Corpus_AdagramTokenized.txt'
    corpus_train_file = 'Data/corpus-100mb-train.txt'
    corpus_dev_file = 'Data/corpus-100mb-dev.txt'

    ## Word Embeddings
    word_embeddings_file = 'Data/English_Wiki_1Billion_embeddings.bin'
    # word_embeddings_file = None
    word_embeddings_dim = 300

    output_model_path = 'Data/model.bin'
    output_id2word_path = 'Data/id2word.txt'
    output_word2id_path = 'Data/word2id.txt'
    output_id2char_path = 'Data/id2char.txt'
    output_char2id_path = 'Data/char2id.txt'

    # Training HyperParameters:
    batch_size = 256
    seq_len = 10
    epochs = 9
    lr = 0.001
    seed = 120
    clip_grad = 5
    dropout_probablity = .25

    ## Network Properties:
    # LSTM Layer:
    n_lstm_layers = 2
    hidden_size = 300
    # CNN Layer:
    cnn_kernels = ['(10, 2)', '(30, 3)', '(40, 4)', '(40, 5)']

    ## Character Feature Detector:
    features_level = ['word', 'character']  # 'word' and 'character'
    character_embedding_dim = 10

    ## Flags:
    bidirectional_model = False
    tie_weights = False
    freez_embeddings = False
    gpu = True

    ## Debug
    print_steps = 50
    use_tensorboard = True


# args = Args()
args = parser.parse_args()

args.cnn_kernels = [tuple(map(int, item.replace('(', '').replace(')', '').replace(' ', '').split(','))) for item in
                    args.cnn_kernels]

# %%

if args.use_tensorboard:
    import tensorflow as tf
    from tensorflow import summary

    # !rm -rf logs
    current_time = str(datetime.datetime.now().timestamp())
    train_log_dir = 'logs/tensorboard/train/' + current_time
    test_log_dir = 'logs/tensorboard/test/' + current_time
    train_summary_writer = summary.create_file_writer(train_log_dir)
    test_summary_writer = summary.create_file_writer(test_log_dir)

# %%

torch.cuda.is_available()


# %%

def evaluate_on_dev(model, corpus_dev_reader, dictionary_word, dictionary_char):
    loss_values = []
    batch_generator_dev = corpus_dev_reader.batchify(dictionary_word, args.batch_size, args.seq_len)

    with torch.no_grad():

        hidden = model.init_hidden(args.batch_size)

        for x_word, y_word in batch_generator_dev:

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
            # model.zero_grad()

            y_hat, hidden = model.forward(x_word, x_char, hidden)

            loss = criterion.forward(y_hat.view(-1, dictionary_word.get_dic_size()),
                                     y_word.reshape(args.batch_size * args.seq_len).long())
            loss_values.append(loss)

    return sum(loss_values) / len(loss_values)


# %%

def train(corpus_train_reader, corpus_dev_reader, dictionary_word, dictionary_char, model, optimizer, criterion,
          epoch_number):
    batch_generator_train = corpus_train_reader.batchify(dictionary_word, args.batch_size, args.seq_len)
    hidden = model.init_hidden(args.batch_size)

    step = 0
    if args.gpu:
        GPU_device_logger = GPUtil.getGPUs()[0]

    with tqdm(unit='words', unit_scale=True, postfix=f'Epoch {epoch_number}') as pbar:

        # Shape of X : (batch_size, seq_len)
        for x_word, y_word in batch_generator_train:

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

            pbar.set_description(f'progress = {corpus_train_reader.get_progress()}%')
            # update number of processed words
            pbar.update(args.batch_size * args.seq_len)

            ####### EVALUATION--------------------
            if step % args.print_steps == 0:

                model.eval()
                dev_loss = evaluate_on_dev(model, corpus_dev_reader, dictionary_word, dictionary_char)
                model.train()

                if args.use_tensorboard:

                    args.globaliter += 1

                    with test_summary_writer.as_default():
                        tf.summary.scalar('loss', dev_loss.item(), step=args.globaliter)
                        tf.summary.text('progress',
                                        f"Epoch {epoch_number} progress = {corpus_train_reader.get_progress()}% ,  Loss = {dev_loss.item()} ,  PPL = {np.exp(dev_loss.item())}",
                                        step=args.globaliter)

                    with train_summary_writer.as_default():

                        tf.summary.scalar('loss', loss.item(), step=args.globaliter)
                        tf.summary.text('progress',
                                        f"Epoch {epoch_number} progress = {corpus_train_reader.get_progress()}% ,  Loss = {loss.item()} ,  PPL = {np.exp(loss.item())}",
                                        step=args.globaliter)
                        if args.gpu:
                            tf.summary.text('GPU Summary',
                                            'Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(
                                                GPU_device_logger.memoryFree, GPU_device_logger.memoryTotal,
                                                GPU_device_logger.memoryUtil * 100), step=args.globaliter)
                else:
                    print(
                        f'epoch:{epoch_number}, step:{step}, progress = {corpus_train_reader.get_progress()}% ,  Loss = {loss.item()} ,  PPL = {np.exp(loss.item())}')


# %%

def detach_hidden(hidden: tuple):
    return tuple(v.detach() for v in hidden)


# %%


# Main script:

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
corpus_dev_reader = CorpusReader(args.corpus_dev_file, 1000000)  # 100MB

dictionary_word = DictionaryWord()
dictionary_char = None

if 'character' in args.features_level:
    dictionary_char = DictionaryCharacter()

model = None
# Load the pre-trained Model for fine-tuning
if path.exists(args.output_model_path):

    print("Loading pre-trained Model...")
    model = LanguageModel.load_model(use_gpu=args.gpu, path_to_pretrained_model=args.output_model_path)

    print("Loading Dictionaries...")
    dictionary_word.load_dictionary(id2word_filepath=args.output_id2word_path,
                                    word2id_filepath=args.output_word2id_path)

    if 'character' in model.features_level:
        dictionary_char.load_dictionary(id2char_filepath=args.output_id2char_path,
                                        char2id_filepath=args.output_char2id_path,
                                        loaded_word_dictionary=dictionary_word)

# Building the model
else:
    print("Generating Dictionaries...")
    dictionary_word.build_dictionary(corpus_train_reader)
    dictionary_word.save_dictionary(args.output_id2word_path, args.output_word2id_path)
    if 'character' in args.features_level:
        dictionary_char.build_dictionary(dictionary_word)
        dictionary_char.save_dictionary(args.output_id2char_path, args.output_char2id_path)
    print("Dictionaries are saved...")

    embeddings_matrix = None
    if args.word_embeddings_file is not None and 'word' in args.features_level:
        print("Loading Embeddings...")
        emb_loader = EmbeddingsLoader()
        embeddings_matrix = emb_loader.get_embeddings_matrix(args.word_embeddings_file, dictionary_word,
                                                             args.word_embeddings_dim)

    print("Instantiating Model...")
    model = LanguageModel.instantiate_model(n_layers=args.n_lstm_layers, hidden_size=args.hidden_size,
                                            n_vocab=dictionary_word.get_dic_size(),
                                            word_emb_dim=args.word_embeddings_dim,
                                            n_chars=dictionary_char.dic_size if 'character' in args.features_level else None,
                                            char_emb_dim=args.character_embedding_dim if 'character' in args.features_level else None,
                                            cnn_kernels=args.cnn_kernels if 'character' in args.features_level else None,
                                            features_level=args.features_level,
                                            dropout_prob=args.dropout_probablity,
                                            bidirectional=args.bidirectional_model,
                                            pret_emb_matrix=embeddings_matrix,
                                            freez_emb=args.freez_embeddings, tie_weights=args.tie_weights,
                                            use_gpu=args.gpu)

###############
print("\nParametes:")
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
    train(corpus_train_reader, corpus_dev_reader, dictionary_word, dictionary_char, model, optimizer, criterion,
          epoch_number=i)
    print(f"Saving Model at epoch {i}...\n")
    model.save_model(args.output_model_path)

# %%
