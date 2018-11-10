import random


def add_letter(s, l):
    if s is not None:
        pos = random.randint(0, len(s))
        return s[:pos] + [l] + s[pos:]
    return s


def replace_letter(s, l):
    if len(s) > 1:
        pos = random.randint(0, len(s) - 1)
        return s[:pos] + [l] + s[pos + 1:]
    else:
        return s


def remove_word(s1, s2):
    words1 = ''.join(s1).split(' ')
    words2 = ''.join(s2).split(' ')
    if len(words1) > 1 and len(words1) == len(words2):
        pos = random.randint(0, len(words1) - 1)
        return list(' '.join(words1[:pos] + words1[pos + 1:])), list(' '.join(words2[:pos] + words2[pos + 1:]))
    return s1, s2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        for t in self.transforms:
            example = t(example)
        return example


class AddLetter:
    def __init__(self, letters, p=0.5):
        self.p = p
        self.letters = letters

    def __call__(self, example):
        if random.random() < self.p:
            pos = random.choice(self.letters)
            example.src = add_letter(example.src, pos)
        return example


class RemoveWord:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, example):
        if random.random() < self.p:
            example.src, example.tgt = remove_word(example.src, example.tgt)
        return example


class ReplaceLetter:
    def __init__(self, letters, p=0.5):
        self.p = p
        self.letters = letters

    def __call__(self, example):
        if random.random() < self.p:
            pos = random.choice(self.letters)
            example.src = replace_letter(example.src, pos)
        return example
