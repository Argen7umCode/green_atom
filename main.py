from model.model import BinaryTextClassifier
from model.trainer import Trainer
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch
from pprint import pprint


LEARNING_RATE = 1e-3
N_EPOCH = 20


model_params = {
    "embedding_matrix" : 'ff', 
    "hidden_dim"       : 50,
    "output_labels"    : 1
}

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryTextClassifier(**model_params)
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)