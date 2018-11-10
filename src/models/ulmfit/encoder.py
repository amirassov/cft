import torch
import torch.nn as nn

from .ulmfit import RNNCore


class EncoderRNN(nn.Module):
    def __init__(
            self,
            vocabulary_size: int=None,
            embedding_size: int=128,
            pad_id: int=None,
            hidden_size: int=128,
            num_layers: int=1,
            bidirectional: bool=True,
            hidden_p: float=0.2,
            input_p: float=0.6,
            embedding_p: float=0.1,
            weight_p: float=0.5):
        super().__init__()

        self.rnn = RNNCore(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pad_id=pad_id,
            bidirectional=bidirectional,
            hidden_p=hidden_p,
            input_p=input_p,
            embedding_p=embedding_p,
            weight_p=weight_p)

    def forward(self, src_seqs, src_lens, hidden=None):
        outputs = self.rnn(src_seqs.transpose(0, 1))[1][-1]
        hidden = self.rnn.hidden
        hidden = (torch.cat([x[0] for x in hidden]), torch.cat([x[1] for x in hidden]))

        if self.rnn.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)
        return outputs, hidden

    @staticmethod
    def _cat_directions(hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-true_seq2seq/blob/master/true_seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, (list, tuple)):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden
