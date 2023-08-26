"""
    Для решения проблемы циклического импорта 
    сведу два класса в один
"""

from .data_importer import DataImporter
from .data_preprocesser import DataPreProcesser
from .vectorizer import Vectorizer


class DataManager:
    def __init__(self, **kwargs) -> None:
        self.__setup_data_importer(**kwargs)
        self.__setup_data_preprocesser(**kwargs)
        self.__setup_vectoriser()

    def __setup_data_importer(self, **kwargs) -> None:
        self.data_importer = DataImporter(kwargs.get('pathes'))

    def __setup_data_preprocesser(self, **kwargs) -> None:
        self.data_preprocesser = DataPreProcesser(kwargs.get('max_length'))
        stopwords = self.data_importer.get_stopwords()
        self.data_preprocesser.set_stopwords(stopwords)
        self.data_preprocesser.set_rex_to_rem_stopwords()
    
    def __setup_vectoriser(self, **kwargs) -> None:
        self.vectorizer = Vectorizer()
        word2int_dict = self.data_importer.get_word2int_dict()
        self.vectorizer.set_word2int_dict(word2int_dict)
        