import torch.nn as nn
from .binary_text_classifier import BinaryTextClassifier
import torch

class Pytorch_model():
    def __init__(self, model_path, gpu_id=None, **kwargs):
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % (self.gpu_id))
        else:
            self.device = torch.device("cpu")

        if self.gpu_id is not None and isinstance(self.gpu_id, int):
            self.use_gpu = True
        else:
            self.use_gpu = False

        if not self.use_gpu:
            self.net = torch.load(
                model_path, map_location=lambda storage, loc: storage.cpu())
        else:
            self.net = torch.load(
                model_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
            
        self.model = BinaryTextClassifier(embedding_matrix=self.net['embedding.weight'], 
                                     hidden_dim=kwargs.get('hidden_dim'),
                                     num_layers=kwargs.get('num_layers')).to(self.device)
        self.model.load_state_dict(self.net)
        self.model.eval()

    def predict(self, text):
        with torch.no_grad():
            outputs = nn.functional.sigmoid(self.model(torch.tensor(text)))
        
        return 1 if outputs[0] > 0.5 else 0
        