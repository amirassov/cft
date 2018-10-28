import torch
import torch.nn as nn

from .attention import Attention
from ..utils import detach_hidden


class DecoderRNN(nn.Module):
    def __init__(self, encoder, sos_id, embedding=None, use_attention=True,
                 bias=True, tie_embeddings=False, dropout=0.3):
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

        self.sos_id = sos_id
        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        self.embedding = embedding
        self.use_attention = use_attention
        self.tie_embeddings = tie_embeddings

        self.vocab_size = self.embedding.num_embeddings
        self.word_vec_size = self.embedding.embedding_dim

        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.word_vec_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout)

        if self.use_attention:
            self.attention = Attention(self.hidden_size, bias)

        if self.tie_embeddings:
            self.W_p = nn.Linear(self.hidden_size, self.word_vec_size, bias=bias)
            self.W_s = nn.Linear(self.word_vec_size, self.vocab_size, bias=bias)
            self.W_s.weight = self.embedding.weight
        else:
            self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)

    def forward(self, tgt_seqs, decoder_hidden, encoder_outputs, attention_mask=None):
        batch_size, max_tgt_len = tgt_seqs.size()
        decoder_outputs = torch.zeros(max_tgt_len, batch_size, self.vocab_size, device=tgt_seqs.device)
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
            input_seq = tgt_seqs[:, t]
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

        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size)
        emb = self.embedding(input_seq)

        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)
        decoder_output, decoder_hidden = self.rnn(emb, decoder_hidden)

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
