#### Author: Arman Kabiri
#### Date: Feb. 18, 2020
#### Email: Arman.Kabiri94@gmail.com

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


if IN_COLAB:
    from google.colab import drive

    drive.mount('/gdrive')

if IN_COLAB:
    import os

    os.chdir('/gdrive/My Drive/NLP_Stuff/My_Language_Model')

import os.path as path
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

getattr(tqdm, '_instances', {}).clear()
import GPUtil

from CorpusReader import CorpusReader
from DictionaryWord import DictionaryWord
from DictionaryCharacter import DictionaryCharacter
from EmbeddingsLoader import EmbeddingsLoader
from Lang_Model import LanguageModel


class Args:
    ## Corpus Files:
    # corpus_train_file='Data/WestburyLab.Wikipedia.Corpus_AdagramTokenized.txt'
    #     corpus_train_file = 'Data/corpus-100mb-train.txt'
    #     corpus_dev_file = 'Data/corpus-100mb-dev.txt'
    corpus_train_file = '../Data/Twitter/COVID19-Tweets-KaggleDataset-parsed-cleaned-100mb-train.txt'
    corpus_dev_file = '../Data/Twitter/COVID19-Tweets-KaggleDataset-parsed-cleaned-100mb-dev.txt'

    ## Word Embeddings
    word_embeddings_file = '../Data/English_Wiki_1Billion_embeddings.bin'
    # word_embeddings_file = None
    word_embeddings_dim = 300

    output_model_path = '../Data/model.bin'
    output_id2word_path = '../Data/id2word.txt'
    output_word2id_path = '../Data/word2id.txt'
    output_id2char_path = '../Data/id2char.txt'
    output_char2id_path = '../Data/char2id.txt'

    # Training HyperParameters:
    batch_size = 128
    seq_len = 10
    epochs = 6
    lr = 0.001
    seed = 120
    clip_grad = 5
    dropout_probablity = 0.30

    ## Network Properties:
    # LSTM Layer:
    n_lstm_layers = 2
    hidden_size = 300
    # CNN Layer:
    cnn_kernels = ['(10, 2)', '(20, 3)', '(30, 4)', '(40, 5)']

    ## Character Feature Detector:
    features_level = ['word', 'character']  # 'word' and 'character'
    character_embedding_dim = 10

    ## Flags:
    bidirectional_model = False
    tie_weights = False
    freez_embeddings = False
    gpu = True

    ## Debug
    print_loss_steps = 100
    evaluate_dev_steps = 1000
    use_tensorboard = False


def get_args_from_terminal():
    parser = argparse.ArgumentParser(description='LSTM Language Model - Train')

    parser.add_argument('--corpus_train_file', type=str,
                        default='../Data/Twitter/COVID19-Tweets-KaggleDataset-parsed-cleaned-100mb-train.txt',
                        help='location of the corpus for training')
    parser.add_argument('--corpus_dev_file', type=str,
                        default='../Data/Twitter/COVID19-Tweets-KaggleDataset-parsed-cleaned-100mb-dev.txt',
                        help='location of the corpus for evaluation')
    parser.add_argument('--word_embeddings_file', type=str, default='../Data/English_Wiki_1Billion_embeddings.bin',
                        help='If pretrained embeddings exist, load them here.')
    parser.add_argument('--word_embeddings_dim', type=int, default=300, help='The dimension of the embeddings')

    parser.add_argument('--output_model_path', type=str, default='../Data/model.bin',
                        help='Path to save or load the trained model.')
    parser.add_argument('--output_id2word_path', type=str, default='../Data/id2word.txt',
                        help='Path to save or dictionary file (id2word)')
    parser.add_argument('--output_word2id_path', type=str, default='../Data/word2id.txt',
                        help='Path to save or dictionary file (word2id)')
    parser.add_argument('--output_id2char_path', type=str, default='../Data/id2char.txt',
                        help='Path to save or dictionary file (id2char)')
    parser.add_argument('--output_char2id_path', type=str, default='../Data/char2id.txt',
                        help='Path to save or dictionary file (char2id)')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per batch.')
    parser.add_argument('--seq_len', type=int, default=15, help='Length of the sequence for back propagation.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--lr', type=int, default=0.001, help='Learning Rate')
    parser.add_argument('--clip_grad', type=int, default=5,
                        help='Clip gradients during training to prevent exploding gradients.')
    parser.add_argument('--dropout_probablity', type=int, default=0.25,
                        help='Dropout probablity applied on embeddings layer and LSTM layer.')
    parser.add_argument('--seed', type=int, default=120, help='The seed for randomness')

    parser.add_argument('--n_lstm_layers', type=int, default=2,
                        help='Number of LSTM layers stacked on top of each other.')
    parser.add_argument('--hidden_size', type=int, default=300, help='Number of hidden units in each LSTM layer')
    parser.add_argument('--character_embedding_dim', type=int, default=10,
                        help='The dimension of the character embeddings')
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
    parser.add_argument('--print_loss_steps', type=int, default=50, help='print training loss every n iteration.')
    parser.add_argument('--evaluate_dev_steps', type=int, default=100, help='evaluate dev and print every n iteration.')
    parser.add_argument('--gpu', action='store_true', help='Turn it on if you have a GPU device.')

    args = parser.parse_args()
    return args

# args = Args()
if is_interactive():
    args = Args()
else:
    args = get_args_from_terminal()

args.cnn_kernels = [tuple(map(int, item.replace('(', '').replace(')', '').replace(' ', '').split(','))) for item in
                    args.cnn_kernels]

assert args.evaluate_dev_steps % args.print_loss_steps == 0

train_summary_writer, test_summary_writer = None, None
if args.use_tensorboard:
    import tensorflow as tf
    from tensorflow import summary

    # !rm -rf logs
    current_time = str(datetime.datetime.now().timestamp())
    train_log_dir = '../logs/tensorboard/train/' + current_time
    test_log_dir = '../logs/tensorboard/test/' + current_time
    train_summary_writer = summary.create_file_writer(train_log_dir)
    test_summary_writer = summary.create_file_writer(test_log_dir)

    print(f"log file: {current_time}")

torch.cuda.is_available()


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


def train(corpus_train_reader, corpus_dev_reader, dictionary_word, dictionary_char, model, optimizer, criterion):
    batch_generator_train = corpus_train_reader.batchify(dictionary_word, args.batch_size, args.seq_len)
    hidden = model.init_hidden(args.batch_size)

    step = 0
    if args.gpu:
        GPU_device_logger = GPUtil.getGPUs()[0]

    with tqdm(unit='words', unit_scale=True, postfix=f'Epoch: {model.current_in_progress_epoch}') as pbar:

        pbar.postfix = ' - Training ...'

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

            if step % args.evaluate_dev_steps == 0:
                pbar.postfix = ' - Evaluation Dev ...'
            pbar.set_description(f'progress = {corpus_train_reader.get_progress()}%')

            # update number of processed words
            pbar.update(args.batch_size * args.seq_len)

            ####### EVALUATION--------------------
            if step % args.print_loss_steps == 0:
                model.globaliter += 1

                print_evaluation(loss, model.globaliter, model.current_in_progress_epoch,
                                 corpus_train_reader.get_progress(), train_summary_writer)

                if args.gpu:
                    print_GPU_Usage(GPU_device_logger, model.globaliter, train_summary_writer)

            if step % args.evaluate_dev_steps == 0:
                if ~args.use_tensorboard:
                    print('Evaluating...')
                model.eval()
                dev_loss = evaluate_on_dev(model, corpus_dev_reader, dictionary_word, dictionary_char)
                model.train()
                print_evaluation(dev_loss, model.globaliter, model.current_in_progress_epoch,
                                 corpus_train_reader.get_progress(), test_summary_writer)
                pbar.postfix = ' - Training ...'

        # Evaluate at the end of epoch:
        model.globaliter += 1
        print_evaluation(loss, model.globaliter, model.current_in_progress_epoch, corpus_train_reader.get_progress(),
                         train_summary_writer)
        model.eval()
        dev_loss = evaluate_on_dev(model, corpus_dev_reader, dictionary_word, dictionary_char)
        model.train()
        print_evaluation(dev_loss, model.globaliter, model.current_in_progress_epoch,
                         corpus_train_reader.get_progress(), test_summary_writer)


def print_evaluation(loss, globaliter, epoch, progress, summary_writer):
    if args.use_tensorboard:

        with summary_writer.as_default():

            tf.summary.scalar('Loss', loss.item(), step=globaliter)
            tf.summary.text('progress',
                            f"Epoch {epoch} progress = {progress}% ,  Loss = {loss.item()} ,  PPL = {np.exp(loss.item())}",
                            step=globaliter)

    else:
        print(
            f'epoch:{epoch}, step:{globaliter}, progress = {progress}% ,  Loss = {loss.item()} ,  PPL = {np.exp(loss.item())}')


def print_GPU_Usage(GPU_device_logger, globaliter, summary_writer):
    if args.use_tensorboard:
        with summary_writer.as_default():
            tf.summary.text('GPU Summary', 'Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(
                GPU_device_logger.memoryFree, GPU_device_logger.memoryTotal,
                GPU_device_logger.memoryUtil * 100), step=globaliter)


def detach_hidden(hidden: tuple):
    return tuple(v.detach() for v in hidden)


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

    # print("Loading Dictionaries...")
    # dictionary_word.load_dictionary(id2word_filepath=args.output_id2word_path,
    #                                 word2id_filepath=args.output_word2id_path)

    # if 'character' in args.features_level:
    #     dictionary_char.load_dictionary(id2char_filepath=args.output_id2char_path,
    #                                     char2id_filepath=args.output_char2id_path,
    #                                     loaded_word_dictionary=dictionary_word)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
if model.optimizer_state is not None:
    optimizer.load_state_dict(model.optimizer_state)

criterion = nn.CrossEntropyLoss()

# Training Model
print("Training Model...")
for i in range(1, args.epochs + 1):
    model.current_in_progress_epoch += 1
    print(f"\nEpoch {model.current_in_progress_epoch}:")
    train(corpus_train_reader, corpus_dev_reader, dictionary_word, dictionary_char, model, optimizer, criterion)
    print(f"Saving Model at epoch {model.current_in_progress_epoch} ...\n")
    model.save_model(args.output_model_path, optimizer)
