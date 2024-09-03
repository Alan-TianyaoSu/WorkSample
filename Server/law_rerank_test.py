import os
import time
import json

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from prompt.utils import extract_query, extract_answer_query
from score.utils import llm_score
from pg.models import AbtestQuery, AbtestRef, AbtestRefV2, ABTestBatch, abtest_ref_table, abtest_batch_table
from pg.utils import pg_pool
from es.utils import get_refs

from score.utils import update_unscored_hanlder, clone_ref_v2_to_ref
from prepare.data import h
from es.utils import bm25search
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import BGE_RERANK_PATH, BGE_RERANK_FT_PATH
from law_uuid_100_0102 import LAW_UUID_100_DICT
from pdf_uuid_1222_02 import dict_uuid
from tokenizer.jieba_tokenizer import tokenizer, law_titles_list
from tokenizer.legal_terminology import law_explain_dict, law_mapping_dict
from tokenizer.law_titles_dict import law_title_mapping_dict


from logger import get_logger

logger = get_logger(__name__)

import os



def supplement_input(input):
    words = tokenizer(input)
    law_title = ''
    for word in words:
        if word in law_mapping_dict.keys():
            input = input.replace(word, law_mapping_dict[word])
        if word in law_title_mapping_dict.keys():
            input = input.replace(word, law_title_mapping_dict[word])
            law_title = law_title_mapping_dict[word]
        if word in law_explain_dict.keys():
            input += law_explain_dict[word]
        if word in law_titles_list:
            law_title = law_title_mapping_dict[word]
    return [input, law_title]


def filter_handle(input, es_filter, law_title):
    filter_list = ['拟上市公司', '拟上市企业', 'IPO', 'ipo', '报告期', '首次公开发行并上市', '首发上市', '首发申报',
                   '申请上市', '上市申请']
    # for el in filter_list:
    #     if el in input:
    #         es_filter.append({"term": {"metadata.priority.keyword": "0"}})
    if law_title:
        es_filter.append({"term": {"metadata.title.keyword": law_title}})
    return es_filter


def question_to_query(input: str, query_kind: int, batch_id: int, index, sub_query):
    ## 根据法律专有名词词典补充描述问题
    input = input.replace("请问根据法律法规的规定", "")
    input = input.replace("请问根据法律法规", "")
    input = input.replace("根据法律法规", "")
    input = input.replace("拟上市公司", "")
    input = input.replace("，", "")
    results = supplement_input(input)
    query = results[0]
    law_title = results[1]
    es_filter = []
    es_filter = filter_handle(input, es_filter, law_title)
    print(f'query:{query}')
    query_to_ref(input, query, query_kind, batch_id, index, sub_query, law_title, es_filter)

"""
根据问题编号和doc_id，返回是否击中 0:否 1:是
"""
def get_scored(index, doc_id):
    result = 0
    if index != 0:
        true_result_id = dict_uuid.get(str(index))
        # print(f'doc_id:{doc_id},true_id:{true_result_id}')
        if doc_id == true_result_id:
            result = 1
    return result


def get_law_scored(index, doc_id):
    result = 0
    if index != 0:
        true_result_id = LAW_UUID_100_DICT.get(str(index))
        # print(f'doc_id:{doc_id},true_id:{true_result_id}')
        if doc_id in true_result_id:
            result = 1
    return result


def insert_table(docs: list, input, index, query, query_kind, batch_id, query_answer):
    law_id_list = [doc.metadata.get('law_id') for doc in docs]
    true_result_list = LAW_UUID_100_DICT.get(str(index))
    total_ture_num = len(true_result_list)
    ture_num = 0
    for law_id in true_result_list:
        if law_id in law_id_list:
            ture_num += 1
    true_score = round(ture_num / total_ture_num, 1)
    query = query_answer if query_answer is not None and query_answer != '' else query
    for doc in docs:
        try:
            title = doc.metadata.get('title')
            _id = doc.metadata.get('_id')
            law_id = doc.metadata.get('law_id')
            scored = get_law_scored(index, law_id)
            score_bm25 = doc.metadata.get('score_bm25')
            rerank_score = doc.metadata.get('rerank_score')
            score = 0
            if score_bm25 is None:
                score = doc.metadata.get('_score')
            page_content = doc.page_content.strip()
        except Exception as e:
            logger.error(f"get_refs异常: {e}")
            continue

        dt = datetime.now()
        try:
            ref_model = AbtestRefV2(
                batch_id=batch_id,
                question=input,
                query_text=query,
                query_kind=query_kind,
                query_type=index,
                es_id=_id,
                ref_title=title,
                ref_text=page_content,
                create_time=dt,
                update_time=dt,
                ref_score_gpt1=true_score,
                scored=scored,
                ref_score_gpt2=score,
                ref_score_gpt3=score_bm25,
                ref_rerank_score=rerank_score
            )
        except:
            ref_model = AbtestRefV2(
                batch_id=batch_id,
                question=input,
                query_text=query,
                query_kind=query_kind,
                query_type=index,
                es_id=_id,
                ref_title=title,
                ref_text=page_content,
                create_time=dt,
                update_time=dt,
                ref_score_gpt1=true_score,
                scored=scored,
                ref_score_gpt2=score,
                ref_score_gpt3=score_bm25,
                ref_rerank_score=rerank_score
            )

        pg_pool.execute_insert(abtest_ref_table, ref_model)

def query_to_ref(input: str, query: str, query_kind: int, batch_id: int, index: int, sub_query: list, law_title,
                 es_filter: list = list()):
    bm25_list = bm25search(query, top_k, query_kind, es_filter)
    eb_list = get_refs(query, query_kind, top_k, es_filter)
    
    query_answer = extract_answer_query(query, query_kind)
    print(f'query_answer:{query_answer}')

    bm25_query_list = bm25search(query_answer, top_k, query_kind, es_filter)
    eb_query_list = get_refs(query_answer, query_kind, top_k, es_filter)
    docs = bm25_list + eb_list + bm25_query_list + eb_query_list
    # doc_lists = []
    # doc_lists.append(bm25_list)
    # doc_lists.append(eb_list)
    # doc_lists.append(bm25_query_list)
    # doc_lists.append(eb_query_list)
    # rrf_docs = rerank(doc_lists, batch_id, input, index, query, query_kind, top_k, 'rrf')

    insert_table(docs, input, index, query, query_kind, batch_id, query_answer)


def read_file_lines(file_path):
    inputs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            input = line.strip()
            if input:
                inputs.append(line.strip())
    return inputs


def create_batch(filename, batch_name, user_id):
    """
    获取batchID时未考虑并发写入batch的情况
    """
    dt = datetime.now()
    model = ABTestBatch(
        file_name=filename,
        batch_name=batch_name,
        create_time=dt,
        update_time=dt,
        user_id=user_id
    )
    pg_pool.execute_insert(abtest_batch_table, model)

    sql = 'select max(id) from {}'.format(abtest_batch_table)
    r = pg_pool.execute_query(sql, ())
    return r[0][0]


def rerank(doc_lists, batch_id, input, index, query, query_kind, top_k, rerank_strategy):
    
    # weights = [round(1/subquery_set_size,2) for i in range(subquery_set_size)]
    # print(f'weights:{weights}')
    match rerank_strategy:
        case "bge_rerank":
            all_documents = []
            for doc_list in doc_lists:
                for doc in doc_list:
                    all_documents.append(doc)
            sort_list = bge_ranking(input, all_documents, False, 10)
            insert_table(sort_list, input, index, query, query_kind)
        case "rrf":
            ######## rrf rerank start#########
            all_documents = set()
            for doc_list in doc_lists:
                for doc in doc_list:
                    all_documents.add(doc.page_content)
            subquery_set_size = len(doc_lists)
            print(f'subquery_set_size:{subquery_set_size}')
            weights = [round(1/subquery_set_size,2) for i in range(subquery_set_size)]
            rrf_score_dic = {doc: 0.0 for doc in all_documents}
            for doc_list, weight in zip(doc_lists, weights):
                for rank, doc in enumerate(doc_list, start=1):
                    rrf_score = weight * (1 / (rank + subquery_set_size * 30))
                    rrf_score_dic[doc.page_content] += rrf_score

            # Sort documents by their RRF scores in descending order
            sorted_documents = sorted(
                rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
            )
            # Map the sorted page_content back to the original document objects
            page_content_to_doc_map = {
                doc.page_content: doc for doc_list in doc_lists for doc in doc_list
            }
            sorted_docs = [
                page_content_to_doc_map[page_content] for page_content in sorted_documents
            ]
            print("sortlen:" + str(len(sorted_docs)))
            return sorted_docs[:60]
            ######## rrf rerank end#########
        case _:
            pass



def bge_ranking(query: str, doc_lists: list, is_ft: bool = True, top_k: int = 30):
    docs = []
    for doc in doc_lists:
        # print(len(doc_list))
        # for doc in doc_list:
        docs.append(doc)
    if is_ft:
        tokenizer = AutoTokenizer.from_pretrained(BGE_RERANK_FT_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANK_FT_PATH)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(BGE_RERANK_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANK_PATH)
        model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    model.to(device)

    pairs = []
    for doc in docs:
        pair = [query, doc.page_content]
        pairs.append(pair)
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs.to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = scores.cpu()
        float_scores = scores.numpy().tolist()
    if len(float_scores) == len(docs):
        sorted_data = sorted(zip(docs, float_scores), key=lambda x: x[1], reverse=True)
        for i in sorted_data:
            doc_i = i[0]
            doc_i.metadata['rerank_score'] = i[1]
        sorted_docs, _ = zip(*sorted_data)
        return sorted_docs[:top_k]


if __name__ == '__main__':
    start = int(time.time())
    query_kind = 1  # 1 law, 2 sec
    filename = 'law_1201_100queries.txt'
    user_id = 'test_lsl'
    top_k = 15

    batch_id = create_batch(filename, 'law_test_0125', user_id)
    print("batch_id:", batch_id)

    input_file = "../data/{}".format(filename)
    inputs = read_file_lines(input_file)
    i = 0
    # index_list = [32,69,60,94,17,22,75,18,82,99,66,9,89,80]
    index_list = [4,6,9,12,32]
    for input in inputs:
        i += 1 # index
        # if i in index_list:
        # if i == 1:
        #     continue
        question_to_query(input, query_kind, batch_id, i, '')
        logger.info(f"finish i - {i}, input - {input}")

    logger.info(("Finished!!!!!!"))
    end = int(time.time())
    logger.info(f"耗时:{end - start}s")
