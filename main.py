from model.model import BinaryTextClassifier
from model.trainer import Trainer
from torch.nn import BCEWithLogitsLoss
from dataprocess import DataImporter, DataPreProcesser, TextDataset, split_data
from torch.optim import Adam
import torch
import pandas as pd
from pprint import pprint
import os


LEARNING_RATE = 1e-3
N_EPOCH = 20
BATCH_SIZE = 128
MAX_TEXT_LENTH = 10_000

model_params = {
    "embedding_matrix" : 'ff', 
    "hidden_dim"       : 50,
    "output_labels"    : 1
}
path = os.getcwdb()



# device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = BinaryTextClassifier(**model_params)
# criterion = BCEWithLogitsLoss()
# optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# trainer = Trainer(model, criterion, device, optimizer, BATCH_SIZE, N_EPOCH, LEARNING_RATE)

pathes = [
    f'data/{i}' for i in ['/train/pos/', '/train/neg/', '/test/pos/', '/test/neg/']
]

try:
    data = pd.read_csv('data/processed_data.csv')
    assert 'preprocessed text' not in data.columns
except Exception as e:
    if e is FileNotFoundError:
        pass
    if e is AssertionError:
        pass

    importer = DataImporter(pathes)
    preprocesser = DataPreProcesser(MAX_TEXT_LENTH)
    raw_data = importer.get_text_and_score()

    dataset = pd.DataFrame(raw_data, columns=('data', 'target'))
    dataset['cleaned text'] = dataset['data'].parallel_apply(clean_regex)
    dataset['preprocessed text'] = dataset['cleaned text'].parallel_apply(preprocess_text)
    dataset['preprocessed text']