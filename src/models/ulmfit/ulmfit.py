import warnings

import torch
from torch import nn
import torch.nn.functional as F


class RNNCore(nn.Module):
    """AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."""
    init_range = 0.1

    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            pad_id: int,
            bidirectional: bool = False,
            hidden_p: float = 0.2,
            input_p: float = 0.6,
            embedding_p: float = 0.1,
            weight_p: float = 0.5):

        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_id)
        self.encoder_dropout = EmbeddingDropout(self.embedding, embedding_p)
        self.rnn = [
            nn.LSTM(
                input_size=embedding_size if i == 0 else hidden_size,
                hidden_size=hidden_size // self.num_directions,
                num_layers=1,
                bidirectional=bidirectional) for i in range(num_layers)]
        self.rnn = [WeightDropout(rnn, weight_p) for rnn in self.rnn]
        self.rnn = torch.nn.ModuleList(self.rnn)
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
        self.input_dropout = RNNDropout(input_p)
        self.hidden_dropouts = nn.ModuleList([RNNDropout(hidden_p) for _ in range(num_layers)])
        self.hidden = None
        self.weights = None
        self.batch_size = None

    def forward(self, input: torch.Tensor, hidden=None):
        self.batch_size = input.size(1)
        self.reset(hidden=hidden)
        raw_output = self.input_dropout(self.encoder_dropout(input))
        new_hidden, raw_outputs, outputs = [], [], []
        for i, (rnn, hidden_dropout) in enumerate(zip(self.rnn, self.hidden_dropouts)):
            raw_output, new_h = rnn(raw_output, self.hidden[i])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.num_layers - 1:
                raw_output = hidden_dropout(raw_output)
            outputs.append(raw_output)
        self.hidden = new_hidden
        return raw_outputs, outputs

    def _one_hidden(self):
        """Return one hidden state."""
        return self.weights.new(self.num_directions, self.batch_size, self.hidden_size // self.num_directions).zero_()

    def reset(self, hidden=None):
        """Reset the hidden states."""
        [r.reset() for r in self.rnn if hasattr(r, 'reset')]
        self.weights = next(self.parameters()).data
        if hidden is None:
            self.hidden = [(self._one_hidden(), self._one_hidden()) for _ in range(self.num_layers)]
        else:
            self.hidden = [(hidden[0][i].unsqueeze(0), hidden[1][i].unsqueeze(0)) for i in range(self.num_layers)]


def dropout_mask(x, size, p: float):
    """Return a dropout mask of the same type as x, size sz, with probability p to cancel an element."""
    return x.new(*size).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    """Dropout that is consistent on the seq_len dimension."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return x * m


class LinearDecoder(nn.Module):
    """To go on top of a RNNCore module and create a Language Model."""

    init_range = 0.1

    def __init__(self, output_size, hidden_size, output_p, tie_encoder=None, bias=True):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, output_size, bias=bias)
        self.decoder.weight.data.uniform_(-self.init_range, self.init_range)
        self.output_dropout = RNNDropout(output_p)
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.output_dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded, raw_outputs, outputs


class EmbeddingDropout(nn.Module):
    """Apply dropout in the embedding layer by zeroing out some elements of the embedding vector."""

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class WeightDropout(nn.Module):
    """A module that warps another layer in which some weights will be replaced by 0 during training."""

    def __init__(self, module: nn.Module, weight_p: float, layer_names=('weight_hh_l0',)):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))

    def _set_weights(self):
        """Apply dropout to the raw weights."""
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._set_weights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()


class SequentialRNN(nn.Sequential):
    """A sequential module that passes the reset call to its children."""

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'):
                c.reset()
