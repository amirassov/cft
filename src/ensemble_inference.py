import argparse

import torch
from torchtext.data import Dataset, Example, RawField
from torchtext.data import Iterator
from tqdm import tqdm

from glob import glob
from src.data import TranslationField
from youtrain.factory import Factory
from src.utils import detach_hidden
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


def translate(models, vocabulary, data, max_seq_len, device):
    with torch.no_grad():
        # -------------------------------------
        # Prepare input and output placeholders
        # -------------------------------------
        # Like dataset's `__getitem__()` and data loader's `collate_fn()`.
        src_seqs, src_lens = data.src
        batch_size = src_seqs.size(0)
        # Decoder's input
        input_seq = torch.tensor([vocabulary.sos_id] * batch_size, dtype=torch.long).to(device)
        sequence_symbols = []
        lengths = np.array([max_seq_len] * batch_size)

        for model in models:
            model.eval()

        outputs, hiddens = [], []
        for model in models:
            output, hidden = model.encoder(src_seqs, src_lens.data.tolist())
            outputs.append(output)
            hiddens.append(hidden)

        encoder_outputs = torch.stack(outputs).mean(0)
        encoder_hidden = (
            torch.stack([x[0] for x in hiddens]).mean(0),
            torch.stack([x[1] for x in hiddens]).mean(0))

        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden

        # Run through decoder one time step at a time.
        attention_mask = src_seqs.eq(vocabulary.pad_id).unsqueeze(1).data
        for t in range(max_seq_len):
            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            outputs, hiddens = [], []
            for model in models:
                output, hidden, _ = model.decoder.forward_step(
                    input_seq=input_seq,
                    decoder_hidden=decoder_hidden,
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask)
                outputs.append(output)
                hiddens.append(hidden)

            decoder_output = torch.stack(outputs).mean(0)
            decoder_hidden = (
                torch.stack([x[0] for x in hiddens]).mean(0),
                torch.stack([x[1] for x in hiddens]).mean(0))
            # Choose top word from decoder's output
            input_seq = decoder_output.data.topk(1)[1].squeeze(1)
            sequence_symbols.append(input_seq.cpu().tolist())
            eos_batches = input_seq.data.eq(vocabulary.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > t) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols) - 1

            # Repackage hidden state (may not need this, since no BPTT)
            detach_hidden(decoder_hidden)

            if max(lengths) < max_seq_len:
                break

        output = []
        for i in range(batch_size):
            output.append(''.join([vocabulary.itos[sequence_symbols[j][i]] for j in range(lengths[i])]))

        return output


def main():
    args = parse_args()

    models = []
    vocabulary = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in glob(args.checkpoint):
        checkpoint = torch.load(name)
        config = checkpoint['meta_data']['config']
        config['train_params']['weights'] = name
        vocabulary = checkpoint['meta_data']['vocabulary']
        print(name, vocabulary.itos)
        factory = Factory(config['train_params'])
        model = factory.make_model().to(device)
        models.append(model)

    src_field = TranslationField()
    index_field = RawField()
    test = pd.read_csv(args.test)
    texts = test['fullname'].tolist()
    examples = [Example.fromlist([x, i], [('src', src_field), ('index', index_field)]) for i, x in enumerate(texts)]
    dataset = Dataset(
        examples=examples,
        fields=[('src', src_field), ('index', index_field)])
    src_field.vocab = vocabulary
    iterator = Iterator(
        dataset=dataset,
        batch_size=2048,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False, shuffle=False)

    prediction_texts = []
    prediction_indices = []
    for data in tqdm(iterator):
        prediction_texts.extend(translate(
            models=models,
            vocabulary=vocabulary,
            data=data,
            max_seq_len=100,
            device=device))
        prediction_indices.extend(data.index)

    prediction = pd.DataFrame(
        [prediction_texts, prediction_indices]).T.rename(
        columns={0: 'fullname_prediction', 1: 'index'})
    test['fullname_true'] = prediction.sort_values('index')['fullname_prediction'].astype(str).tolist()
    test.loc[test['fullname_true'] == test['fullname'], 'target'] = 0
    test.loc[test['fullname_true'] != test['fullname'], 'target'] = 1
    test.loc[test['fullname_true'] == 'nan', 'target'] = 2
    test.loc[test['target'] != 1, 'fullname_true'] = ''
    test['target'] = test['target'].astype(int)
    test[['id', 'target', 'fullname_true']].to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
