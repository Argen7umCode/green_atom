"""
    Для решения проблемы циклического импорта 
    сведу два класса в один
"""

from data_importer import DataImporter
from data_preprocesser import DataPreProcesser


class DataManager:
    def __init__(self, **kwargs) -> None:
        self.__setup_data_importer(**kwargs)
        self.__setup_data_preprocesser(**kwargs)


    def __setup_data_importer(self, **kwargs) -> None:
        self.data_importer = DataImporter(kwargs['pathes'])

    def __setup_data_preprocesser(self, **kwargs) -> None:
        self.data_preprocesser = DataPreProcesser(kwargs['max_length'])
        stopwords = self.data_importer.get_stopwords()
        self.data_preprocesser.set_stopwords(stopwords)
        self.data_preprocesser.set_rex_to_rem_stopwords()
    
    
        