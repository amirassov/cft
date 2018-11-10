import torch.nn as nn

from .encoder import EncoderRNN
from .decoder import DecoderRNN


class Seq2Seq(nn.Module):
    def __init__(
            self, vocabulary_size, embedding_size, hidden_size, num_layers, pad_id, sos_id,
            bidirectional=True, hidden_p=0.2, input_p=0.2, embedding_p=0.1, weight_p=0.2,
            use_attention=True, bias=True, share_embeddings=True, tie_embeddings=True):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = EncoderRNN(
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

        self.decoder = DecoderRNN(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            pad_id=pad_id,
            hidden_p=hidden_p,
            input_p=input_p,
            embedding_p=embedding_p,
            weight_p=weight_p,
            use_attention=use_attention,
            bias=bias,
            tie_embeddings=tie_embeddings,
            sos_id=sos_id)

        if share_embeddings:
            self.encoder.rnn.embedding.weight = self.decoder.rnn.embedding.weight

    def forward(self, src_seqs, src_lens, tgt_seqs=None, teacher_forcing_ratio=1.0):
        encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lens.data.tolist())

        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden
        decoder_outputs, decoder_hidden = self.decoder(
            tgt_seqs=tgt_seqs,
            decoder_hidden=decoder_hidden,
            encoder_outputs=encoder_outputs,
            attention_mask=src_seqs.eq(self.pad_id).unsqueeze(1).data,
            teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_outputs, decoder_hidden
