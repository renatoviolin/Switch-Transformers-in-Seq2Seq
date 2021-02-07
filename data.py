import config
import spacy
import torch
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

train_data.examples = train_data.examples[:10000]


SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE)


def get_train_iterator():
    return train_iterator


def get_valid_iterator():
    return valid_iterator


def get_test_iterator():
    return test_iterator
