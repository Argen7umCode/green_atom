import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.utils import pad_sequences
from collections import Counter
from pandarallel import pandarallel
from re import


class DataPreProcesser:

    def __init__(self, max_lenth) -> None:
        nltk.download('stopwords')
        self.max_lenth = max_lenth
        self.stopwords = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize_text(self, text):
        return word_tokenize(text)

    def split_tords_by_space(self, text):
        return text.split()

    def make_lower(self, word_list):
        return list(map(lambda t: t.lower(), word_list))

    def lemmatize(self, word_list):
        return list(map(self.lemmatizer.lemmatize, word_list))

    def remove_stop_words(self, word_list):
        return [word for word in word_list if word not in stopwords]

    def add_bos_tag(self, word_list):
        word_list.insert(0, '<bos>')
        return word_list

    def limit_sequence(self, word_list, max_lenth):
        return word_list[:max_lenth]

    def join_word_list(word_list):
        return ' '.join(word_list)

    def preprocess_text(self, text):
        word_list = self.tokenize_text(text)
        lower_word_list = self.make_lower(word_list)
        lemmatized_word_list = self.make_lower(lower_word_list)
        word_list_with_bos = self.add_bos_tag(self.remove_stop_words(lemmatized_word_list))
        preprocessed_text = self.join_word_list(self.limit_sequence(word_list_with_bos, self.max_lenth))
        
        return preprocessed_text

    def preprocess_dataset(self, raw_dataset):
