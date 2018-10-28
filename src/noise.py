import random
import string


def remove_space(words):
    pos = random.randint(0, len(words) - 1)
    words[pos:pos + 2] = [''.join(words[pos:pos + 2])]
    return words


def remove(words):
    if len(words) > 1:
        pos = random.randint(0, len(words) - 1)
        return words[:pos] + words[pos + 1:]
    return words


def repeat_word(words):
    pos = random.randint(0, len(words) - 1)
    return words[:pos] + [words[pos]] + words[pos:]


def repeat_letter(s):
    pos = random.randint(0, len(s) - 1)
    return s[:pos] + s[pos] + s[pos:]


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def add_letter(s, letters):
    pos = random.randint(0, len(s))
    addition = random.choice(letters)
    return s[:pos] + addition + s[pos:]


def add_word(words, letters, max_word_size=10):
    pos = random.randint(0, len(words) - 1)
    addition = ''.join(random.choices(letters, k=random.randint(0, max_word_size)))
    return words[:pos] + [addition] + words[pos:]


def add_noise(sentence, letters):
    words = sentence.split()
    if random.random() < 0.2:
        words = remove_space(words)
    if random.random() < 0.1:
        words = remove(words)
    if random.random() < 0.1:
        words = repeat_word(words)
    if random.random() < 0.1:
        words = add_word(words, letters=letters)
    for i, s in enumerate(words):
        if random.random() < 0.2:
            words[i] = repeat_letter(words[i])
        if random.random() < 0.1:
            words[i] = remove_punctuation(words[i])
        if random.random() < 0.05:
            words[i] = add_letter(words[i], letters)
        if random.random() < 0.05:
            words[i] = remove(words[i])
    return ' '.join(words)
