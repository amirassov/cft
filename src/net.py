import torch
from src.lm.utils import translate


class LMNet:
    MAX_SEQ_LEN = 320

    def __init__(self, model_path, device):
        self.device = device
        self.model = torch.load(model_path).to(device)

    def predict(self, text):
        prediction, _ = translate(
            self.model,
            list(text) + ['<eos>'],
            self.MAX_SEQ_LEN,
            self.device)
        return prediction
