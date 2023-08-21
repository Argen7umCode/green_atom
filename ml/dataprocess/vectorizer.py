from json import load


class Vectorizer:
    def __init__(self):
        with open('../data/word2int.json') as file:
            self.word2int_dict = load(file)
     
    def encode_sequence(self, sequence):
        return [self.word2int_dict[word] for word in sequence]
    
    