import torch
import pandas as pd


def pad_sentences_batch(sentences_batch):
    lengths = [len(sentence) for sentence in sentences_batch]
    targets = torch.zeros(len(sentences_batch), max(lengths)).long()
    for i, sentence in enumerate(sentences_batch):
        sentence = torch.LongTensor(sentence)
        end = lengths[i]
        targets[i, :end] = sentence[:end]
    return targets, lengths

def multi_label_flat(df, label_col, split_seg=' '):
    res = {col: [] for col in df.columns if col != label_col}
    label_res = []
    for index, row in df.iterrows():
        labels = row[label_col].split(split_seg)
        for label in labels:
            label_res.append(label)
            for col in res:
                res[col].append(row[col])
    res[label_col] = label_res
    res_df = pd.DataFrame(data=res)
    return res_df

