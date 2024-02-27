from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
import pandas as pd


def create_dataset(json_file_path):
    dataset = EmbeddingQAFinetuneDataset.from_json(json_file_path)
    return dataset


def evaluate_hit_rate(dataset, model_id, name, top_k=5):
    results_list = []
    scores = []
    corpus = dataset.corpus
    relevant_docs = dataset.relevant_docs
    queries = dataset.queries

    nodes = [TextNode(id=id, text=text) for id, text in corpus.items()]
    vs = VectorStoreIndex(nodes, embed_model=model_id, show_progress=True)
    retriever = vs.as_retriever(similarity_top_k=top_k)

    for (query_id, query) in queries.items():
        answer_nodes = retriever.retrieve(query)
        predicted_ids = [node.node.node_id for node in answer_nodes]
        actual_ids = relevant_docs[query_id][0]
        is_hit = actual_ids in predicted_ids

        results = {'query_id': query_id, 'query': query, 'score': is_hit, 
                   'retrieved_ids': None if is_hit else actual_ids, 'predicted_ids' : None if is_hit else predicted_ids}

        results_list.append(results)
        scores.append(1 if is_hit else 0)
        total_score = sum(scores)
        accuracy = total_score * 100 / len(list(queries))

    return results_list, scores, total_score, accuracy

def evaluate_IRE(dataset):
    