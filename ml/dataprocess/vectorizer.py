class Vectorizer:
    def __init__(self):
        pass

     
    def set_word2int_dict(self, word2int_dict):
        self.word2int_dict = word2int_dict

        

    def encode_sequence(self, sequence):
        return [self.word2int_dict[word] for word in sequence]
    
    