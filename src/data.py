from collections import Counter, OrderedDict
from itertools import chain

from torchtext.data import Field, Iterator, Dataset
from torchtext.vocab import Vocab
from torchtext.datasets import TranslationDataset

from src.youtrain.factory import DataFactory


class TranslationField(Field):
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    UNK_TOKEN = '<unk>'
    vocab_cls = Vocab

    def __init__(self, **kwargs):
        kwargs['batch_first'] = True
        kwargs['include_lengths'] = True
        kwargs['tokenize'] = (lambda s: list(s))
        kwargs['eos_token'] = self.EOS_TOKEN
        kwargs['pad_token'] = self.PAD_TOKEN
        kwargs['unk_token'] = self.UNK_TOKEN
        self.vocab = None
        super().__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token, self.SOS_TOKEN]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
        self.vocab.pad_id = self.vocab.stoi[self.PAD_TOKEN]
        self.vocab.eos_id = self.vocab.stoi[self.EOS_TOKEN]
        self.vocab.sos_id = self.vocab.stoi[self.SOS_TOKEN]
        self.vocab.unk_id = self.vocab.stoi[self.UNK_TOKEN]


class TranslationFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.field = TranslationField()
        self.max_seq_len = self.params['max_seq_len']
        self.train_dataset = self.make_dataset([self.paths['src_train'], self.paths['tgt_train']])
        self.val_dataset = self.make_dataset([self.paths['src_test'], self.paths['tgt_test']])
        self.field.build_vocab(self.train_dataset, max_size=50000)
        self.device = kwargs['device']

    def make_dataset(self, exts):
        return TranslationDataset(
            path=self.paths['path'],
            exts=exts,
            fields=[('src', self.field), ('tgt', self.field)],
            filter_pred=lambda x: len(x.src) <= self.max_seq_len and len(x.tgt) <= self.max_seq_len)

    def make_loader(self, stage, is_train=False):
        dataset = self.train_dataset if is_train else self.val_dataset
        return Iterator(
            dataset=dataset, batch_size=self.params['batch_size'],
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device, repeat=False, shuffle=is_train)
