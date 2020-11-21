from datasets.common.transforms import *
from torch.nn import functional as F
from transformers import pipeline

pipeline()

# train_text_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'
# classes_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/classes.txt'
# vocab_path = '/home/ubuntu/likun/vocab/dbpedia_vocab.json'
# sentence_bert_path = '/home/ubuntu/likun/nlp_pretrained/bert-base-nli-mean-tokens'
# transformer = SentenceBertTransformer(sentence_bert_path)
# dataset = DBpediaCencept(text_path=train_text_path, classes_path=classes_path, vocab_json=vocab_path,
#                          desc_transformer=transformer, label_name_transformer=lambda x: x, data_nums=1000)
# label_name_embeddings = {label_name: transformer(label_name) for label_name in dataset.idx2classes}
#
# desc, item, label, label_name = dataset.__getitem__(3)
# print(f'{label_name}')
# for class_name, class_embedding in label_name_embeddings.items():
#     sim = F.cosine_similarity(desc.unsqueeze(0), class_embedding.unsqueeze(0))
#     print(f'{class_name}: {sim}')


# for c in classes:
#     p = os.path.join(path, c)
#     file_names = os.listdir(p)
#     for n in file_names:
#         print(c + n)
#         try:
#             with open(os.path.join(p, n), 'r') as f:
#                 text.append(f.read())
#                 label.append(c)
#         except Exception as e:
#             print(e)
