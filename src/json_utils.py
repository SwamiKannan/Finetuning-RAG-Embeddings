import json
# from sentence_transformers import SentenceTransformer, losses, InputExample
# from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import pickle


def get_id():
    a = [8, 4, 4, 4, 12]  # Structure of ids provided in llama_index.finetuning.generate_qa_embedding_pairs()
    x = 'abcdefghijklmnopqrstuvwxyz0123456789'
    w = '-'.join([''.join(random.choices(x, weights=[1 for _ in range(len(x))], k=v)) for v in a])
    return w


def create_corpus(qa_items, negative_size, all_ids=[]):
    corpus_check = {}
    content = []
    index = list(qa_items.keys())
    print('Index length', len(index))
    for i in index:
        uiud_q, uiud_r = 0, 0
        while uiud_q in all_ids:
            uiud_q = get_id()
        all_ids.append(uiud_q)
        question = qa_items[i]['Question']
        answer = qa_items[i]['Reference']
        corpora_id = random.sample(index[:int(i)]+index[int(i)+1:], negative_size)
        corpora_list = [qa_items[id]['Reference'] for id in corpora_id] + [answer]
        random.shuffle(corpora_list)
        corpus = ''.join(corpora_list)
        if corpus in corpus_check.keys():
            corpus_check[corpus] = uiud_r
            uiud_r = corpus_check[corpus]
        else:
            while uiud_r in all_ids:
                uiud_r = get_id()
            all_ids.append(uiud_r)
        content.append((uiud_q, question, uiud_r, corpus))
    del corpus_check
    return content


def create_json(content_list, output_file):
    final_dict = {}
    queries = {}
    response_docs = {}
    corpus = {}
    for content in content_list:
        queries[content[0]] = content[1]
        corpus[content[2]] = content[3]
        response_docs[content[0]] = [content[2]]
    final_dict['queries'] = queries
    final_dict['relevant_docs'] = response_docs
    final_dict['corpus'] = corpus
    final_dict['mode'] = 'text'
    with open(output_file, 'w') as f:
        json.dump(final_dict, f)


if __name__ == "__main__":
    NUMBER_OF_CORPUS_ANSWERS = 10
    fa = open('all_used_keys.pkl', 'wb+')
    try:
        all_ids = pickle.load(fa)
    except Exception:
        all_ids = [0]

    with open("D:\\Falcon_projects\\problems.json", 'r') as f:
        qas = json.load(f)

    data = [qa[1] for qa in list(qas.items())]
    final_qas = {}
    topics = ['units-and-measurement', 'science-and-engineering-practices', 'physics']
    i = 0
    for d in data:
        if d['topic'] in topics:
            question = d['question']
            reference = d['lecture'] + d['solution']
            final_qas.update({i: {'Question': question, 'Reference': reference}})
            i += 1


# model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# for row in data:
#     i =
