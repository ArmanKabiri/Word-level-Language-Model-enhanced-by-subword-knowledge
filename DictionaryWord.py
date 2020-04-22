# Created by Arman Kabiri on 2020-02-27 - 1:50 p.m.
# Author's Email Address: Arman.Kabiri94@gmail.com

from tqdm import tqdm


class DictionaryWord:

    def __init__(self):

        self.word2id = dict()
        self.id2word = list()
        self.vocab_size = 0

    def build_dictionary(self, corpus_reader):

        print("Building dictionaries...")
        self.word2id = dict()
        self.id2word = list()
        reader = corpus_reader.load_corpus_inchunk()

        for chunk in tqdm(reader):
            words = chunk.split(' ')
            for word in words:
                if word not in self.word2id:
                    self.id2word.append(word)
                    self.word2id[word] = len(self.id2word) - 1

        self.vocab_size = len(self.id2word)
        print(f"Dictionaries are built - Vocab size is {self.vocab_size}")

    def load_dictionary(self, id2word_filepath: str, word2id_filepath: str):

        with open(id2word_filepath, 'r') as file:
            self.id2word = [word.rstrip() for word in file]

        with open(word2id_filepath, 'r') as file:
            for line in file:
                word, word_id = line.rstrip().split('\t')
                self.word2id[word] = int(word_id)

        assert len(self.word2id) == len(self.id2word)
        self.vocab_size = len(self.word2id)
        print(f"Dictionaries are loaded - Vocab size is {len(self.id2word)}")

    def encode_text(self, text: str) -> list:
        return [self.word2id[word] for word in text.split(' ')]

    def decode_text(self, sequence: list) -> str:
        return ' '.join([self.id2word[idx] for idx in sequence])

    def get_dic_size(self) -> int:
        return len(self.id2word)

    def get_vocabulary(self) -> set:
        return set(self.word2id.keys())
