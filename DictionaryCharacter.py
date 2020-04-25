# Created by Arman Kabiri on 2020-04-21 - 7:20 a.m.
# Author's Email Address: Arman.Kabiri94@gmail.com

import numpy as np

from DictionaryWord import DictionaryWord


class DictionaryCharacter:
    TOKEN_BEGINING_OF_WORD = "BOW"
    TOKEN_END_OF_WORD = "EOW"
    TOKEN_PADING = "PAD"

    def __init__(self):
        self.char2id = dict()
        self.id2char = list()
        self.dic_size = 0
        self.max_word_len = 0

    def build_dictionary(self, word_dict: DictionaryWord):

        print("Building character dictionaries...")
        self.word_dict = word_dict

        # initializing control characters:
        self.id2char.append(self.TOKEN_PADING)
        self.char2id[self.TOKEN_PADING] = 0

        vocabulary = word_dict.get_vocabulary()
        char_id = 1

        for word in vocabulary:
            self.max_word_len = max(self.max_word_len, len(word))
            for ch in word:
                if ch not in self.char2id:
                    self.id2char.append(ch)
                    self.char2id[ch] = char_id
                    char_id += 1

        # initializing control characters:
        self.id2char.append(self.TOKEN_BEGINING_OF_WORD)
        self.id2char.append(self.TOKEN_END_OF_WORD)
        self.char2id[self.TOKEN_BEGINING_OF_WORD] = char_id
        self.char2id[self.TOKEN_END_OF_WORD] = char_id + 1

        self.dic_size = len(self.char2id)
        print(f"Character Dictionaries are built - Vocab size is {self.dic_size}")

    def encode_word(self, word):
        """
        :return the shape of the encoded word is always (self.max_word_len + 2,)
        """
        encoded_word = []
        for ch in word:
            id_ch = self.char2id[ch]
            encoded_word.append(id_ch)

        # Padding
        if len(encoded_word) < self.max_word_len:
            encoded_word += [self.char2id[self.TOKEN_PADING] for _ in range(self.max_word_len - len(encoded_word))]

        encoded_word = [self.char2id[self.TOKEN_BEGINING_OF_WORD]] + encoded_word \
                       + [self.char2id[self.TOKEN_END_OF_WORD]]

        return np.array(encoded_word)

    def encode_batch(self, input_batch: np.ndarray):
        """

        :param input_batch: Numpy array with the shape of (batch_size,seq_len)
        this values are already encoded from words to ids
        :return: Output is a numpy array object with shape of (batch_size, seq_len, self.max_word_len + 2)
        """
        batch_size, seq_len = input_batch.shape
        input_batch = input_batch.reshape(batch_size * seq_len)
        output_batch = np.zeros((batch_size * seq_len, self.max_word_len + 2))

        for index, word_id in enumerate(input_batch):
            word = self.word_dict.id2word[word_id]
            encoded_word = self.encode_word(word)
            output_batch[index] = encoded_word

        output_batch = output_batch.reshape((batch_size, seq_len, self.max_word_len + 2))
        return output_batch

    def save_dictionary(self, id2char_filepath: str, char2id_filepath: str):

        with open(char2id_filepath, 'w') as file:
            for char, char_id in self.char2id.items():
                if '\t' in char:
                    exit()
                file.write(f"{char}\t{char_id}\n")

        with open(id2char_filepath, 'w') as file:
            for char in self.id2char:
                file.write(f"{char}\n")

    def load_dictionary(self, id2char_filepath: str, char2id_filepath: str, loaded_word_dictionary):

        with open(id2char_filepath, 'r') as file:
            self.id2char = [char.rstrip() for char in file]

        with open(char2id_filepath, 'r') as file:
            for line in file:
                char, char_id = line.rstrip().split('\t')
                self.char2id[char] = int(char_id)

        assert len(self.char2id) == len(self.id2char)
        self.dic_size = len(self.char2id)

        self.max_word_len = 0
        vocabulary = loaded_word_dictionary.get_vocabulary()
        for word in vocabulary:
            self.max_word_len = max(self.max_word_len, len(word))

        self.word_dict = loaded_word_dictionary
        print(f"Character dictionary is loaded - Vocab size is {len(self.id2char)}")
