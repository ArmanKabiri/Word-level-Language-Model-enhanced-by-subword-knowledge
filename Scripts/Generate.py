# Created by Arman Kabiri on 2020-03-13 - 10:12 a.m.
# Author's Email Address: Arman.Kabiri94@gmail.com

from DictionaryWord import DictionaryWord
from DictionaryCharacter import DictionaryCharacter
from Lang_Model import LanguageModel

import numpy as np
import torch
import torch.nn.functional as F
import argparse

BEGINNING_OF_TWEET_SYMBOL = '<BOT>'
END_OF_TWEET_SYMBOL = '<EOT>'


class Args:
    model_path = '../Data/model.bin'
    id2word_path = '../Data/id2word.txt'
    word2id_path = '../Data/word2id.txt'
    id2char_path = '../Data/id2char.txt'
    char2id_path = '../Data/char2id.txt'

    seed_sequence = "rats should"
    seed_random = 120
    gpu = False


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_args_from_terminal():
    parser = argparse.ArgumentParser(description='LSTM Language Model - Train')

    parser.add_argument('--model_path', type=str, default='../Data/model.bin',
                        help='location of the trained model')
    parser.add_argument('--id2word_path', type=str, default='../Data/id2word.txt',
                        help='Path to the dictionary file (id2word)')
    parser.add_argument('--word2id_path', type=str, default='../Data/word2id.txt',
                        help='Path to the dictionary file (word2id)')
    parser.add_argument('--id2char_path', type=str, default='../Data/id2char.txt',
                        help='Path to the dictionary file (id2char)')
    parser.add_argument('--char2id_path', type=str, default='../Data/char2id.txt',
                        help='Path to the dictionary file (char2id)')
    parser.add_argument('--seed_sequence', type=str, default='my girlfriend',
                        help='A sequence given to the model for Tweet generation.')
    parser.add_argument('--seed_random', type=int, default=120, help='The seed for randomness')
    parser.add_argument('--gpu', action='store_true', help='Turn it on if you have a GPU device.')

    args = parser.parse_args()

    return args


if is_interactive():
    args = Args()
else:
    args = get_args_from_terminal()


def predict_next_word(model: LanguageModel, dictionary_word, dictionary_char, hidden, input_text: str, k=10) -> tuple:
    input_word = dictionary_word.encode_text(input_text)
    input_word = np.array(input_word)
    input_word = input_word.reshape((1, 1))
    input_char = None
    if 'character' in model.features_level:
        input_char = dictionary_char.encode_batch(input_word)
        input_char = torch.from_numpy(input_char)
        input_char = input_char.cuda() if args.gpu else input_char

    input_word = torch.from_numpy(input_word)
    if args.gpu:
        input_word = input_word.cuda()

    input_word = input_word.view(-1, 1)

    output, hidden = model.forward(input_word, input_char, hidden)

    probs = F.softmax(output, 2)

    # move back to CPU to use with numpy
    if args.gpu:
        probs = probs.cpu()

    probs, picked_indexes = probs.topk(k)
    picked_indexes = picked_indexes.numpy().squeeze()
    probs = probs.numpy().flatten()
    probs = probs / probs.sum()
    word = np.random.choice(picked_indexes, p=probs)

    word = dictionary_word.decode_text([word.item()])

    return word, hidden


def generate_text(model: LanguageModel, dictionary_word: DictionaryWord, dictionary_char: DictionaryCharacter,
                  seed_seq: str, k=5):
    if args.gpu:
        model.cuda()
    else:
        model.cpu()

    model.eval()

    with torch.no_grad():

        hidden = model.init_hidden(batch_size=1)
        seed_text = (BEGINNING_OF_TWEET_SYMBOL + ' ' + seed_seq).split()
        output = []

        for word in seed_text:
            output.append(word)
            pred_word, hidden = predict_next_word(model, dictionary_word, dictionary_char, hidden, word, k)

        output.append(pred_word)

        while pred_word != END_OF_TWEET_SYMBOL:
            pred_word, hidden = predict_next_word(model, dictionary_word, dictionary_char, hidden, pred_word, k)
            output.append(pred_word)

    output = ' '.join(output)
    output = output.replace(BEGINNING_OF_TWEET_SYMBOL, '').replace(END_OF_TWEET_SYMBOL, '').strip()
    return output


def main():
    torch.manual_seed(args.seed_random)

    # Loading Model
    model = LanguageModel.load_model(args.gpu, args.model_path)
    model.eval()

    # loading dictionaries
    dictionary_word = DictionaryWord()

    dictionary_char = None
    if 'character' in model.features_level:
        dictionary_char = DictionaryCharacter()

    dictionary_word.load_dictionary(args.id2word_path, args.word2id_path, )
    dictionary_char.load_dictionary(args.id2char_path, args.char2id_path, dictionary_word)

    args.seed_sequence = args.seed_sequence.lower()

    for i in range(10):
        result = generate_text(model=model, dictionary_word=dictionary_word, dictionary_char=dictionary_char,
                               seed_seq=args.seed_sequence, k=10)
        print(result)


main()