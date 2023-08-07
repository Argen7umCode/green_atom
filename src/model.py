import torch.nn as nn
import torch

class BinaryTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_labels):
        super(BinaryTextClassifier, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim 
        self.output_labels = output_labels
        self.embed_dim = self.embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_labels)
        
    def forward(self, text):
        embedded = self.embedding(text) 
        lstm_out, _ = self.lstm(embedded)
        lstm_avg = torch.mean(lstm_out, dim=1)
        output = self.fc(lstm_avg).squeeze(1)
        return output