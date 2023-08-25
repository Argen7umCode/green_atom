import os 
from numpy import load, save
from config import basedir
from json import load, dump

class DataImporter:
    
    def __init__(self, pathes=None) -> None:
        self.pathes = pathes
        pattern = f'{basedir}'
        self.word2vec_matrix_path = pattern + 'data/word2vec_martix.npy'
        self.stopwords_path = pattern + '/data/stopwords.json'
        self.word2int_path = pattern + '/data/word2ind.json'

    def get_text_and_score(self):
        text_score = []
        for path in self.pathes:
            file_names = os.listdir(path)
            for file_name in file_names:
                with open(f'{path}/{file_name}', 'r') as file:
                    text = file.read()
                    score = int(file_name.split('.')[0].split('_')[-1])
                text_score.append([text, score])
        return text_score
    
    def __load_json_file(self, path, func_if_exception):
        try:
            with open(path, 'r') as file:
                data = load(file)
        except FileNotFoundError:
            data = func_if_exception()
            # Тут нужна обработка 

        return data
    
    def make_word2ver_matrix(self):
        pass

    def get_word2vec_matrix(self):
        path = self.word2vec_matrix_path
        wav2vec_matrix = self.__load_json_file(path, None)

        return wav2vec_matrix
    
    def make_stopwords_matrix(self):
        pass

    def get_stopwords(self):
        path = self.stopwords_path
        stopwords = self.__load_json_file(path, None)

        return stopwords
 
    def make_word2int_dict(self):
        pass
    
    def get_word2int_dict(self):
        path = self.word2int_path
        word2int_dict = self.__load_json_file(path, None)

        return word2int_dict