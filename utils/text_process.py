import torch


def pad_sentences_batch(sentences_batch):
    lengths = [len(sentence) for sentence in sentences_batch]
    targets = torch.zeros(len(sentences_batch), max(lengths)).long()
    for i, sentence in enumerate(sentences_batch):
        sentence = torch.LongTensor(sentence)
        end = lengths[i]
        targets[i, :end] = sentence[:end]
    return targets, lengths
