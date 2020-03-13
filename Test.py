# Created by Arman Kabiri on 2020-03-13 - 10:12 a.m.
# Author's Email Address: Arman.Kabiri94@gmail.com

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from Dictionary import Dictionary
from Lang_Model import LanguageModel

parser = argparse.ArgumentParser(description='LSTM Language Model - Text Generator')
parser.add_argument('--model_path', type=str, help='location of the trained model')
parser.add_argument('--seed_word', type=str, help='Seed word')


def main():
    # TODO
    generate_text()


def generate_text(model: LanguageModel, dictionary: Dictionary, seed: str, gpu_enabled, k=1):
    if gpu_enabled:
        model.cuda()
    else:
        model.cpu()

    model.eval()

    with torch.no_grad():
        hidden = model.init_hidden(1)
        input_text = seed
        output = [seed]

        for i in range(10):
            word, hidden = predict_next_word(model, dictionary, hidden, input_text, gpu_enabled, k)
            output.append(word)
            input_text = word

    print(' '.join(output))


def predict_next_word(model: LanguageModel, dictionary, hidden, input_text: str, gpu_enabled, k=1) -> tuple:
    input_tensor = dictionary.encode_text(input_text)
    input_tensor = np.array(input_tensor)
    input_tensor = torch.from_numpy(input_tensor)
    if gpu_enabled:
        input_tensor = input_tensor.cuda()

    input_tensor = input_tensor.view(-1, 1)
    output, hidden = model.forward(input_tensor, hidden)
    # TODO here
    probs = F.softmax(output, 2)

    # move back to CPU to use with numpy
    if gpu_enabled:
        probs = probs.cpu()

    probs, picked_indexes = probs.topk(k)
    picked_indexes = picked_indexes.numpy().squeeze()
    probs = probs.numpy().flatten()
    probs = probs / probs.sum()
    word = np.random.choice(picked_indexes, p=probs)

    word = dictionary.decode_text([word.item()])

    return word, hidden
