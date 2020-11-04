#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vocab.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import six
from utils import io
import torchtext

__all__ = ['Vocab']

EBD_PAD = EBD_ALL_ZEROS = '<pad>'
EBD_UNKNOWN = '<unk>'
EBD_BOS = '<bos>'
EBD_EOS = '<eos>'

class Tokenizer(object):
    def __init__(self, f=None):
        self.tokenizer = f

    def registry_method(self, f):
        self.tokenizer = f

    def tokenize(self, s):
        self.tokenizer(s)


class Vocab(object):
    def __init__(self, word2idx=None):
        self.word2idx = word2idx if word2idx is not None else dict()
        self._idx2word = None

    @classmethod
    def from_json(cls, json_file):
        return cls(io.load_json(json_file))

    @classmethod
    def from_text(cls, tokens, tokenizer=None):
        """
        :param tokens: ['im ok is there', 'This is good']
        :param tokenizer: default None will tokenize by split space
        """
        text_field = torchtext.data.Field(init_token=EBD_BOS, eos_token=EBD_EOS,
                                          pad_token=EBD_PAD, unk_token=EBD_UNKNOWN)
        if tokenizer is None:
            tokenizer = Tokenizer(f=lambda x: x.split())
        tokens = [tokenizer.tokenize(t) for t in tokens]
        text_field.build_vocab(tokens)
        return cls(text_field.vocab.stoi)

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

    def map_sequence(self, sequence):
        if isinstance(sequence, six.string_types):
            sequence = sequence.split()
        return [self.map(w) for w in sequence]

    def map_fields(self, feed_dict, fields):
        feed_dict = feed_dict.copy()
        for k in fields:
            if k in feed_dict:
                feed_dict[k] = self.map(feed_dict[k])
        return feed_dict
