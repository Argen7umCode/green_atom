import torch.nn as nn
import torch


class Pytorch_model():
    def __init__(self, model_path, gpu_id=None):
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
        self.net.eval()

    def predict(self, text):
        outputs = self.net(text.squeeze())
        return outputs