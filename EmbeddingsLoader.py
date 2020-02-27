# Created by Arman Kabiri on 2020-02-27 - 1:59 p.m.
# Author's Email Address: Arman.Kabiri94@gmail.com

import logging
import gensim
from tqdm import tqdm
from Dictionary import Dictionary
import numpy as np


class EmbeddingsLoader:

    def __init__(self):

        self.dim = 0
        self.embeddings_size = 0
        self.emb_matrix = None

    def __load_pretrained_embeddings(self, input_file: str) -> dict:

        logging.info("Loading pretrained embeddings...")
        vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=input_file, binary=True)
        logging.info("Pretrained embeddings are loaded.")
        self.emb_dict = vectors
        self.dim = len(vectors.values()[0])
        return vectors

    def get_embeddings_matrix(self, input_file: str, dictionary: Dictionary, emb_dim) -> np.array:

        pretrained_emb = self.__load_pretrained_embeddings(input_file)
        assert emb_dim == self.dim
        self.embeddings_size = dictionary.get_dic_size()
        weights_matrix = np.zeros((self.embeddings_size, self.dim))

        for i, word in enumerate(dictionary.id2word):

            try:
                weights_matrix[i] = pretrained_emb[word]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.dim,))

        self.emb_matrix = weights_matrix

        return weights_matrix
