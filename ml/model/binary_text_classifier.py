import torch
import torch.nn as nn


class BinaryTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, output_labels=1):
        super(BinaryTextClassifier, self).__init__()

        self.embedded = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(input_size=self.embedded, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, text):
        embedded = self.embedding(text) 
        lstm_out, _ = self.lstm(embedded)
        lstm_avg = torch.mean(lstm_out, dim=1)
        output = self.fc(lstm_avg).squeeze(1) 

        return output