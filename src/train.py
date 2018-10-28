import torch
import argparse
from pathlib import Path

from src.youtrain.factory import Factory
from src.youtrain.runner import Runner
from src.youtrain.callbacks import ModelSaver, TensorBoard, Callbacks, Logger
from src.youtrain.utils import set_global_seeds, get_config

from src.lm.data import TranslationFactory

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    return parser.parse_args()


def create_callbacks(name, dumps):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    save_dir = Path(dumps['path']) / dumps['weights'] / name
    callbacks = Callbacks([
        Logger(log_dir),
        ModelSaver(
            save_dir=save_dir,
            save_every=1,
            save_name=f"best.pt",
            best_only=True),
        TensorBoard(log_dir)])
    return callbacks


class TranslationRunner(Runner):
    def _make_step(self, data, is_train):
        report = {}
        src_seqs, src_lens = data.src
        tgt_seqs, tgt_lens = data.tgt
        _, max_tgt_len = tgt_seqs.shape

        if is_train:
            self.optimizer.zero_grad()

        decoder_outputs, decoder_hidden = self.model(
            src_seqs=src_seqs,
            tgt_seqs=tgt_seqs,
            src_lens=src_lens)

        loss = self.loss(decoder_outputs[:max_tgt_len].transpose(0, 1).contiguous(), tgt_seqs)

        report['loss'] = loss.data
        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            report['grad'] = grad_norm
            self.optimizer.step()
        return report


def main():
    args = parse_args()
    set_global_seeds(666)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(args.config)
    paths = get_config(args.paths)
    data_factory = TranslationFactory(config['data_params'], paths['data'], device=device)

    config['train_params']['model_params']['src_vocab'] = data_factory.field.vocab
    config['train_params']['model_params']['tgt_vocab'] = data_factory.field.vocab
    config['train_params']['loss_params']['pad_id'] = data_factory.field.vocab.pad_id
    factory = Factory(config['train_params'])

    callbacks = create_callbacks(config['train_params']['name'], paths['dumps'])
    runner = TranslationRunner(
        stages=config['stages'],
        factory=factory,
        callbacks=callbacks,
        device=device)

    runner.fit(data_factory)


if __name__ == '__main__':
    main()
