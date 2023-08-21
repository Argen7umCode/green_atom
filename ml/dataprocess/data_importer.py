import os 
from numpy import load, save
from config import basedir
from json import load, dump

class DataImporter:
    
    def __init__(self, pathes=None) -> None:
        self.pathes = pathes
        self.wav2vec_matrix_path = f'{basedir}/data/word2vec_martix.npy'
        self.stopwords_path = f'{basedir}/data/stopwords.json'

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
    
    def get_wav2vec_matrix(self):
        if os.path.isfile(self.wav2vec_matrix_path):
            with open(self.wav2vec_matrix_path, 'rb') as file:
                wav2vec_matrix = load(file)
        else:
            pass

        return wav2vec_matrix
    
    def get_stopwords(self):
        with open(self.stopwords_path, 'r') as file:
            stopwords = load(file)

        return stopwords