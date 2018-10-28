import torch.nn as nn

from .encoder import EncoderRNN
from .decoder import DecoderRNN


class Seq2Seq(nn.Module):
    def __init__(
            self, src_vocab=None, tgt_vocab=None,
            word_vec_size=128, rnn_type='LSTM', hidden_size=128,
            num_layers=1, dropout=0.3, bidirectional=True,
            use_attention=True, bias=True, share_embeddings=True,
            tie_embeddings=False):
        super().__init__()
        self.pad_id = src_vocab.pad_id
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        src_embedding = nn.Embedding(len(src_vocab), word_vec_size, padding_idx=src_vocab.pad_id)
        tgt_embedding = nn.Embedding(len(tgt_vocab), word_vec_size, padding_idx=tgt_vocab.pad_id)

        if share_embeddings and self.src_vocab == self.tgt_vocab:
            tgt_embedding.weight = src_embedding.weight

        self.encoder = EncoderRNN(
            embedding=src_embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
        self.decoder = DecoderRNN(
            encoder=self.encoder,
            embedding=tgt_embedding,
            dropout=dropout,
            use_attention=use_attention,
            bias=bias,
            tie_embeddings=tie_embeddings,
            sos_id=tgt_vocab.sos_id)

    def forward(self, src_seqs, src_lens, tgt_seqs=None):
        encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lens.data.tolist())

        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden
        decoder_outputs, decoder_hidden = self.decoder(
            tgt_seqs=tgt_seqs,
            decoder_hidden=decoder_hidden,
            encoder_outputs=encoder_outputs,
            attention_mask=src_seqs.eq(self.src_vocab.pad_id).unsqueeze(1).data)
        return decoder_outputs, decoder_hidden
