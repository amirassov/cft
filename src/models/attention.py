import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_c = nn.Linear(2 * hidden_size, hidden_size, bias=bias)

    def forward(self, decoder_output, encoder_outputs, mask):
        """
        ------------------------------------------------------------------------------------------
        Notes of computing attention scores
        ------------------------------------------------------------------------------------------
        # For-loop version:

        max_src_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attention_scores = torch.zeros(batch_size, max_src_len)

        # For every batch, every time step of encoder's hidden state, calculate attention score.
        for b in range(batch_size):
            for t in range(max_src_len):
                attention_scores[b,t] = decoder_output[b].dot(attention.W_a(encoder_outputs[t,b]))

        ------------------------------------------------------------------------------------------
        # Vectorized version:

        1. decoder_output: (batch_size, seq_len=1, hidden_size)
        2. encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        3. W_a(encoder_outputs): (max_src_len, batch_size, hidden_size)
                        .transpose(0,1)  : (batch_size, max_src_len, hidden_size)
                        .transpose(1,2)  : (batch_size, hidden_size, max_src_len)
        4. attention_scores:
                        (batch_size, seq_len=1, hidden_size) * (batch_size, hidden_size, max_src_len)
                        => (batch_size, seq_len=1, max_src_len)
        """
        # attention_scores: (batch_size, seq_len=1, max_src_len)
        attention_scores = torch.bmm(decoder_output, self.W_a(encoder_outputs).transpose(0, 1).transpose(1, 2))

        if mask is not None:
            attention_scores.data.masked_fill_(mask, -float('inf'))

        # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax`
        # => (batch_size, seq_len=1, max_src_len)
        attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)

        # context_vector:
        # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
        # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
        context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1))

        # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
        concat_input = torch.cat([context_vector, decoder_output], -1)

        # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size,
        # seq_len=1, decoder_hidden_size)
        concat_output = F.tanh(self.W_c(concat_input))

        # Prepare returns:
        # (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len)
        attention_weights = attention_weights.squeeze(1)
        return concat_output, attention_weights
