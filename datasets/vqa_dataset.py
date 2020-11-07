from utils import io
from datasets.common.vocab import build_vocab
from torch.utils.data.dataset import Dataset

def get_questions_from_json(json_file):
    question_json = io.load_json(json_file)
    questions = []
    for question in question_json['questions']:
        questions.append(question['question'])
    return questions

def get_answers_from_json(json_file):
    question_json = io.load_json(json_file)
    answers = []
    for question in question_json['questions']:
        answers.append(question['answer'])
    return answers

class VQADataset(Dataset):
    def __init__(self, image_root_path, json_path, question_vocab_path, answer_vocab_path,
                 transforms, question_per_image=None):
        super(VQADataset, self).__init__()

if __name__ == '__main__':
    # build vocab
    clevr_json_files = ('/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json',
                        '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_val_questions.json',
                        '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_test_questions.json')
    question_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_question_vocab.json'
    answer_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_answer_vocab.json'
    build_vocab(file_paths=clevr_json_files, compile_functions=(get_questions_from_json, ) * 3, vocab_path=question_vocab_path)
    build_vocab(file_paths=clevr_json_files, compile_functions=(get_answers_from_json, ) * 3, vocab_path=answer_vocab_path)