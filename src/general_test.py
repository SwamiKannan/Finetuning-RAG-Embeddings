# import json
import uuid
import random

def get_id():
    a =[8, 4, 4, 4, 12]
    x='abcdefghijklmnopqrstuvwxyz0123456789'
    w = '-'.join([''.join(random.choices(x,weights=[1 for _ in range(len(x))],k=v)) for v in a])
    return w

print(get_id())

# x=random.sample(x,)

# index = [0,1,2,3,4,5]
# id1=2
# print(index[id1])
# print(index[:id1]+index[id1+1:])
# print(index)
# set_topics=set()
# set_subject=set()
# with open("D:\\Falcon_projects\\finetune_physics_embeddings\\train_dataset.json",'r') as f:
#     qas = json.load(f)

# # print(qas.keys())
# queries,corpora,rel_docs,mode = qas['queries'], qas['corpus'], qas['relevant_docs'], qas['mode']
# print('Mode', mode)
# for q in queries:
#     query = queries[q]
#     rel_doc = rel_docs[q]
#     corpus_all =[]
#     corpus_all=[corpora[r] for r in rel_doc]
#     break
# count = 0
# total = 0
# benchmark = corpora[list(corpora.keys())[0]]

# print(len(list(corpora.keys())))

# for key in corpora.keys():
#     if corpora[key] != benchmark:
#         count+=1
#         print('Mismatch Corpora')
#         print(corpora[key])
#         print('Mismatch benchmark')
#         print(benchmark)
#     total+=1

# print(count)
# print(total)
# print('Query: ',query)
# print('Relevant documents: ', rel_doc)
# print('Corpus: ', corpus_all)

# for d in data:
#     set_subject.add(d['subject'])
#     set_topics.add(d['topic'])

# print('Subjects: ', set_subject)
# print('Topics: ', set_topics)
