# Created by Arman Kabiri on 2020-02-27 - 1:52 p.m.
# Author's Email Address: Arman.Kabiri94@gmail.com


class CorpusReader:

    def __init__(self, input_file: str):
        self.input_file = input_file

    def load_corpus_inchunk(self, chunk: int = 100000000) -> str:

        with open(self.input_file, 'r') as f:
            while True:

                buf = f.read(chunk)
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
