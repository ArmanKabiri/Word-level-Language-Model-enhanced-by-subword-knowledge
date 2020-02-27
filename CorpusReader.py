# Created by Arman Kabiri on 2020-02-27 - 1:52 p.m.
# Author's Email Address: Arman.Kabiri94@gmail.com
from Dictionary import Dictionary
import numpy as np


class CorpusReader:

    def __init__(self, input_file: str, chunk_size: int = 100000000):
        self.input_file = input_file
        self.chunk = chunk_size

    def load_corpus_inchunk(self) -> str:

        with open(self.input_file, 'r') as f:
            while True:

                buf = f.read(self.chunk)
                if not buf:
                    break

                # make sure we end on a space (word boundary)
                while not str.isspace(buf[-1]):
                    ch = f.read(1)
                    if not ch:
                        break
                    buf += ch

                yield buf
            yield ''  # handle the scene that the file is empty

    def batchify(self, dictionary: Dictionary, batch_size: int, seq_len: int):

        reader = self.load_corpus_inchunk()
        left_from_previous_chunk = []

        for chunk in reader:

            encoded_text = dictionary.encode_text(chunk)
            encoded_text = left_from_previous_chunk + encoded_text

            # -(1*batch_size) is for y of the last sample per chunk
            n_batches = (len(encoded_text) - batch_size) // (batch_size * seq_len)
            left_over = (len(encoded_text) - batch_size) % (batch_size * seq_len)

            if left_over != 0:
                left_from_previous_chunk = encoded_text[-left_over:]
                encoded_text = encoded_text[:-left_over]
            else:
                left_from_previous_chunk = []

            encoded_text = np.array(encoded_text)
            encoded_text = encoded_text.reshape((batch_size, -1))

            for i in range(0, encoded_text.shape[1] - 1, seq_len):
                x = encoded_text[:, i: i + seq_len]
                y = encoded_text[:, i + 1:i + seq_len + 1]
                yield x, y
