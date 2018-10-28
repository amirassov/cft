import torch
import argparse

from src.lm.utils import translate
from src.lm.data import TranslationFactory
from src.youtrain.utils import get_config
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.config)
    paths = get_config(args.paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factory = TranslationFactory(config['data_params'], paths=paths['data'], device=device)
    dataset = factory.val_dataset

    model = torch.load(args.weights).to(device)
    tgt_texts, out_texts = [], []
    for pair in tqdm(dataset):
        src, tgt = pair.src, pair.tgt
        out_text, all_attention_weights = translate(
            model, src + ['<eos>'],
            config['data_params']['max_seq_len'],
            device)
        tgt_texts.append(''.join(tgt))
        out_texts.append(out_text)

    print(f'ACCURACY: {sum(x == y for x, y in zip(out_texts, tgt_texts)) / len(tgt_texts)}')
    with open(Path(paths['data']['path']) / paths['data']['pre_test'], "w") as f:
        f.write('\n'.join(out_texts))


if __name__ == '__main__':
    main()
