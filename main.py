from model.model import BinaryTextClassifier
from model.trainer import Trainer
from torch.nn import BCEWithLogitsLoss
from dataprocess import DataImporter, DataPreProcesser, TextDataset, split_data
from torch.optim import Adam
import torch
from pprint import pprint
import os


LEARNING_RATE = 1e-3
N_EPOCH = 20
BATCH_SIZE = 128

model_params = {
    "embedding_matrix" : 'ff', 
    "hidden_dim"       : 50,
    "output_labels"    : 1
}
path = os.getcwdb()
print(os.listdir(f'{str(path)}/data'))


# device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = BinaryTextClassifier(**model_params)
# criterion = BCEWithLogitsLoss()
# optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# trainer = Trainer(model, criterion, device, optimizer, BATCH_SIZE, N_EPOCH, LEARNING_RATE)



# pathes = [
#     f'{path}{i}' for i in ['/train/pos/', '/train/neg/', '/test/pos/', '/test/neg/']
# ]

# # raw_data = DataImporter(pathes).get_text_and_score()

# # if op.get