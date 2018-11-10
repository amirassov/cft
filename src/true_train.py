import torch
import argparse
from pathlib import Path

from youtrain.factory import Factory
from youtrain.runner import Runner
from youtrain.callbacks import TensorBoard, Callbacks, Logger, CheckpointSaver
from youtrain.utils import set_global_seeds, get_config

from src.data import TranslationFactory

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
        CheckpointSaver(
            metric_name='loss',
            save_dir=save_dir,
            save_name="best_{epoch}_{metric}.pt",
            num_checkpoints=6,
            mode='min'),
        TensorBoard(log_dir)])
    return callbacks


class TranslationRunner(Runner):
    def __init__(self, factory, callbacks, stages, device, meta_data: dict = None):
        super().__init__(factory, callbacks, stages, device, meta_data)
        self.teacher_forcing_ratio = 1.0

    def fit(self, data_factory):
        self.callbacks.on_train_begin()
        for stage in self.stages:
            self.current_stage = stage

            train_loader = data_factory.make_loader(stage, is_train=True)
            val_loader = data_factory.make_loader(stage, is_train=False)

            self.optimizer = self.factory.make_optimizer(self.model, stage)
            self.scheduler = self.factory.make_scheduler(self.optimizer, stage)
            self.teacher_forcing_ratio = stage['teacher_forcing_ratio']

            self.callbacks.on_stage_begin()
            self._run_one_stage(train_loader, val_loader)
            self.callbacks.on_stage_end()

        self.callbacks.on_train_end()

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
            src_lens=src_lens,
            teacher_forcing_ratio=self.teacher_forcing_ratio if is_train else 0.0)

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

    config['train_params']['model_params']['vocabulary_size'] = len(data_factory.field.vocab)
    config['train_params']['model_params']['pad_id'] = data_factory.field.vocab.pad_id
    config['train_params']['model_params']['sos_id'] = data_factory.field.vocab.sos_id
    config['train_params']['loss_params']['pad_id'] = data_factory.field.vocab.pad_id

    factory = Factory(config['train_params'])

    callbacks = create_callbacks(config['train_params']['name'], paths['dumps'])
    runner = TranslationRunner(
        stages=config['stages'],
        factory=factory,
        callbacks=callbacks,
        device=device,
        meta_data={
            'vocabulary': data_factory.field.vocab,
            'config': config})

    runner.fit(data_factory)


if __name__ == '__main__':
    main()
