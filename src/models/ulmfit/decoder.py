import torch
import torch.nn as nn
import random
from .attention import Attention
from .ulmfit import RNNCore
from ...utils import detach_hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, sos_id, pad_id,
                 use_attention, hidden_size, num_layers,
                 bias, tie_embeddings, hidden_p, input_p, embedding_p, weight_p):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025

            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling"
                (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462
        """
        super().__init__()
        self.rnn = RNNCore(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pad_id=pad_id,
            hidden_p=hidden_p,
            input_p=input_p,
            embedding_p=embedding_p,
            weight_p=weight_p)
        self.vocabulary_size = vocabulary_size
        self.sos_id = sos_id
        self.use_attention = use_attention
        self.tie_embeddings = tie_embeddings

        if self.use_attention:
            self.attention = Attention(hidden_size, bias)

        if self.tie_embeddings:
            self.W_p = nn.Linear(hidden_size, embedding_size, bias=bias)
            self.W_s = nn.Linear(embedding_size, vocabulary_size, bias=bias)
            self.W_s.weight = self.rnn.embedding.weight
        else:
            self.W_s = nn.Linear(hidden_size, vocabulary_size, bias=bias)

    def forward(self, tgt_seqs, decoder_hidden, encoder_outputs, attention_mask=None, teacher_forcing_ratio=1.0):
        batch_size, max_tgt_len = tgt_seqs.size()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_outputs = torch.zeros(max_tgt_len, batch_size, self.vocabulary_size, device=tgt_seqs.device)
        input_seq = torch.tensor([self.sos_id] * batch_size, dtype=torch.long, device=tgt_seqs.device)
        for t in range(max_tgt_len):
            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            decoder_output, decoder_hidden, attention_weights = self.forward_step(
                input_seq=input_seq,
                decoder_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask)

            # Store decoder outputs.
            decoder_outputs[t] = decoder_output

            # Next input is current tgt
            if use_teacher_forcing:
                input_seq = tgt_seqs[:, t]
            else:
                input_seq = decoder_output.data.topk(1)[1].squeeze(1).detach()
            # Detach hidden state:
            detach_hidden(decoder_hidden)
        return decoder_outputs, decoder_hidden

    def forward_step(self, input_seq, decoder_hidden, encoder_outputs, attention_mask):
        """ Args:
            - input_seq      : (batch_size)
            - decoder_hidden : (t=0) last encoder hidden state (num_layers * num_directions,
                                batch_size, hidden_size)
                               (t>0) previous decoder hidden state (num_layers, batch_size, hidden_size)
            - encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)

            Returns:
            - output           : (batch_size, vocab_size)
            - decoder_hidden   : (num_layers, batch_size, hidden_size)
            - attention_weights: (batch_size, max_src_len)
        """
        # (batch_size) => (seq_len=1, batch_size)
        input_seq = input_seq.unsqueeze(0)

        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)
        decoder_output = self.rnn(input_seq, decoder_hidden)[1][-1]
        decoder_hidden = self.rnn.hidden
        decoder_hidden = (torch.cat([x[0] for x in decoder_hidden]), torch.cat([x[1] for x in decoder_hidden]))

        # (seq_len=1, batch_size, hidden_size) => (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0, 1)

        if self.attention:
            concat_output, attention_weights = self.attention(decoder_output, encoder_outputs, attention_mask)
        else:
            attention_weights = None
            concat_output = decoder_output

        # If input and output embeddings are tied,
        # project `decoder_hidden_size` to `word_vec_size`.
        if self.tie_embeddings:
            output = self.W_s(self.W_p(concat_output))
        else:
            # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)
            output = self.W_s(concat_output)

            # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)

        return output, decoder_hidden, attention_weights
