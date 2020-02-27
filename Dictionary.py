# Created by Arman Kabiri on 2020-02-27 - 1:50 p.m.
# Author's Email Address: Arman.Kabiri94@gmail.com

from CorpusReader import CorpusReader
import logging
from tqdm import tqdm


class Dictionary:

    def __init__(self, corpus_reader: CorpusReader):

        self.word2id = dict()
        self.id2word = list()
        self.corpus_reader = corpus_reader

    def build_dictionary(self):

        logging.info("Building dictionaries...")
        self.word2id = dict()
        self.id2word = list()
        reader = self.corpus_reader.load_corpus_inchunk()

        for chunk in tqdm(reader):
            words = chunk.split(' ')
            for word in words:
                if word not in self.word2id:
                    self.id2word.append(word)
                    self.word2id[word] = len(self.id2word) - 1

        self.vocab_size = len(self.id2word)
        logging.info(f"Dictionaries are built - Vocab size is {len(self.id2word)}")

    def encode_text(self, text: str) -> list:
        return [self.word2id[word] for word in text.split(' ')]

    def decode_text(self, sequence: list) -> str:
        ' '.join([self.id2word[idx] for idx in sequence])

    def get_dic_size(self) -> int:
        return len(self.id2word)
