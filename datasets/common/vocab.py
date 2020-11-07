#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vocab.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.
from collections import Counter
import tqdm
import six
from utils import io
from utils.logger import get_logger
import nltk

logger = get_logger(__file__)

EBD_PAD = EBD_ALL_ZEROS = '<pad>'
EBD_UNKNOWN = '<unk>'
EBD_BOS = '<bos>'
EBD_EOS = '<eos>'


class Vocab(object):
    def __init__(self, word2idx=None):
        self.word2idx = word2idx if word2idx is not None else dict()
        self._idx2word = None

    @classmethod
    def from_json(cls, json_file):
        return cls(io.load_json(json_file))

    # @classmethod
    # def from_text(cls, tokens, tokenizer=None):
    #     """
    #     :param tokens: ['im ok is there', 'This is good']
    #     :param tokenizer: default None will tokenize by split space
    #     """
    #     text_field = torchtext.data.Field(init_token=EBD_BOS, eos_token=EBD_EOS,
    #                                       pad_token=EBD_PAD, unk_token=EBD_UNKNOWN)
    #     if tokenizer is None:
    #         tokenizer = Tokenizer(f=lambda x: x.split())
    #     tokens = [tokenizer.tokenize(t) for t in tokens]
    #     text_field.build_vocab(tokens)
    #     return cls(text_field.vocab.stoi)

    def dump_json(self, json_file):
        io.dump_json(json_file, self.word2idx)

    def check_json_consistency(self, json_file):
        rhs = io.load_json(json_file)
        for k, v in self.word2idx.items():
            if not (k in rhs and rhs[k] == v):
                return False
        return True

    def words(self):
        return self.word2idx.keys()

    @property
    def idx2word(self):
        if self._idx2word is None or len(self.word2idx) != len(self._idx2word):
            self._idx2word = {v: k for k, v in self.word2idx.items()}
        return self._idx2word

    def __len__(self):
        return len(self.word2idx)

    def add(self, word):
        self.add_word(word)

    def add_word(self, word):
        self.word2idx[word] = len(self.word2idx)

    def map(self, word):
        return self.word2idx.get(
            word,
            self.word2idx.get(EBD_UNKNOWN, -1)
        )

    def map_sequence(self, sequence, add_bos=True, add_eos=True):
        if isinstance(sequence, six.string_types):
            sequence = sequence.split()
        if add_bos:
            sequence = [EBD_BOS] + sequence
        if add_eos:
            sequence = sequence + [EBD_EOS]
        return [self.map(w) for w in sequence]

    def map_fields(self, feed_dict, fields):
        feed_dict = feed_dict.copy()
        for k in fields:
            if k in feed_dict:
                feed_dict[k] = self.map(feed_dict[k])
        return feed_dict

def get_sentences_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()
    return sentences

def build_vocab(file_paths, compile_functions, vocab_path=None, threshold=0):
    logger.critical(f"Start build vocab")
    assert len(file_paths) == len(compile_functions)
    counter = Counter()
    for file_path, funcs in zip(file_paths, compile_functions):
        if funcs is None:
            sentences = get_sentences_from_txt(file_path)
        else:
            sentences = funcs(file_path)
        for caption in tqdm.tqdm(sentences, desc=f'Tokenizing'):
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            counter.update(tokens)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in list(counter.items()) if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocab()
    vocab.add_word(EBD_PAD)
    vocab.add_word(EBD_BOS)
    vocab.add_word(EBD_EOS)
    vocab.add_word(EBD_UNKNOWN)

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    if vocab_path is not None:
        vocab.dump_json(vocab_path)
    logger.critical(f"Success build vocab, saved in {vocab_path}")
    return vocab