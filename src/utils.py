import torch


def detach_hidden(hidden):
    """ Wraps hidden states in new Variables, to detach them from their history. Prevent OOM.
        After detach, the hidden's requires_grad=False and grad_fn=None.
    Issues:
    - Memory leak problem in LSTM and RNN: https://github.com/pytorch/pytorch/issues/2198
    - https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    - https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
    - https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
    -
    """
    for h in hidden:
        h.detach_()


def translate(model, src_text, max_seq_len, device, replace_unk=True):
    with torch.no_grad():
        # -------------------------------------
        # Prepare input and output placeholders
        # -------------------------------------
        # Like dataset's `__getitem__()` and data loader's `collate_fn()`.
        src_seqs = torch.tensor([[model.src_vocab.stoi[tok] for tok in src_text]], dtype=torch.long)
        batch_size = src_seqs.size(0)
        src_lens = torch.tensor([src_seqs.size(1)], dtype=torch.long)
        # Decoder's input
        input_seq = torch.tensor([model.src_vocab.sos_id] * batch_size, dtype=torch.long)
        # Store output words and attention states
        out_sent = []
        all_attention_weights = torch.zeros(max_seq_len, src_seqs.size(1))

        # Move variables from CPU to GPU.
        src_seqs = src_seqs.to(device)
        src_lens = src_lens.to(device)
        input_seq = input_seq.to(device)

        model.eval()
        encoder_outputs, encoder_hidden = model.encoder(src_seqs, src_lens.data.tolist())

        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden

        # Run through decoder one time step at a time.
        attention_mask = src_seqs.eq(model.src_vocab.pad_id).unsqueeze(1).data
        for t in range(max_seq_len):

            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            decoder_output, decoder_hidden, attention_weights = model.decoder.forward_step(
                input_seq, decoder_hidden, encoder_outputs, attention_mask)

            # Store attention weights.
            # .squeeze(0): remove `batch_size` dimension since batch_size=1

            all_attention_weights[t] = attention_weights.squeeze(0).cpu().data

            # Choose top word from decoder's output
            prob, token_id = decoder_output.data.topk(1)
            token_id = token_id[0].item()  # get value
            if token_id == model.src_vocab.eos_id:
                break
            else:
                if token_id == model.src_vocab.unk_id and replace_unk:
                    # Replace unk by selecting the source token with the highest attention score.
                    score, idx = all_attention_weights[t].max(0)
                    token = src_text[idx.item()]
                else:
                    token = model.tgt_vocab.itos[token_id]
            out_sent.append(token)

            # Next input is chosen word
            input_seq = torch.tensor([token_id], dtype=torch.long)
            input_seq = input_seq.to(device)

            # Repackage hidden state (may not need this, since no BPTT)
            detach_hidden(decoder_hidden)

        out_text = ''.join(out_sent)

        # all_attention_weights: (out_len, src_len)
        return out_text, all_attention_weights
