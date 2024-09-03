import os
import time
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from prompt.utils import extract_query,extract_answer_query
from score.utils import llm_score
from pg.models import AbtestQuery, AbtestRef, AbtestRefV2, ABTestBatch, AbtestRefNew, abtest_ref_table, abtest_batch_table, abtest_ref_table_new
from pg.utils import pg_pool
from es.utils import get_refs

from score.utils import update_unscored_hanlder, clone_ref_v2_to_ref
from prepare.data import h
from es.utils import bm25search
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import BGE_RERANK_PATH, BGE_RERANK_FT_PATH, PEG_RERANK_PATH, ALIME_RERANK_PATH
from law_uuid_100_0102 import LAW_UUID_100_DICT
from pdf_uuid_1222_02 import dict_uuid
from tokenizer.jieba_tokenizer import tokenizer, tokenizer_sec
from tokenizer.legal_terminology import sec_explain_dict, sec_mapping_dict
from general.query_utils import extract_base_question_sheet
from utils_v2_keyword import output_bm25_top_k_docs,output_emb_top_k_docs,question_output_bm25_top_k_docs
from v202_structured_query_recursion import custom_search_v4
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from config import EMBEDDING_PATH


from logger import get_logger


logger = get_logger(__name__)

import os


def supplement_input(input):
    input = input.replace('“','').replace('”','')
    words = tokenizer_sec(input)
    for word in words:
        if word in sec_mapping_dict.keys():
            input = input.replace(word, sec_mapping_dict[word])
        if word in sec_explain_dict.keys():
            input += sec_explain_dict[word]
    return input

def cal_cosine(input, content):
    model = SentenceTransformer(EMBEDDING_PATH)
    # 获取句子向量
    vector1 = model.encode(input, convert_to_tensor=True)
    vector2 = model.encode(content, convert_to_tensor=True)
    # 计算余弦相似度
    cosine_similarity = 1 - cosine(vector1, vector2)
    return cosine_similarity

def question_to_query(input: str, query_kind: int, batch_id: int, index, base_dict):
    ## 根据案例词典补充描述问题
    query = supplement_input(input)
    print(f'query:{query}')
    # if input_sup == input:
    #     query = extract_answer_query(input, query_kind)
    #     print(query)
    #     query_to_ref(input, query, query_kind, batch_id, index, sub_query, es_filter)
    # else:
    #     query_to_ref(input, input_sup, query_kind, batch_id, index, sub_query, es_filter)
    # query = extract_answer_query(input, query_kind)
    # print(query)
    query_to_ref(input, query, query_kind, batch_id, index, base_dict, es_filter)
    """
    input表示原始问题
    query表示用于获取ref的输入
    """
    # query_to_ref(input, input, query_kind, batch_id, es_filter)
    # question入库
    # print('--------------------------------------------')
    # query = extract_query(input, query_kind)
    # question衍生query入库
    # query_to_ref(input, query, query_kind, batch_id, es_filter)
    # query_to_ref(input, query, query_kind, batch_id, es_filter)

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

def insert_table(docs:list, input, index, query, query_kind, batch_id):
    # law_id_list = [doc.metadata.get('law_id') for doc in docs]
    # true_result_list = LAW_UUID_100_DICT.get(str(index))
    # total_ture_num = len(true_result_list)
    # ture_num = 0
    # for law_id in true_result_list:
    #     if law_id in law_id_list:
    #         ture_num+=1
    # true_score = round(ture_num/total_ture_num, 1)
    for doc in docs:
        try:
            title = doc.metadata.get('title')
            _id = doc.metadata.get('_id')
            doc_id = doc.metadata.get('doc_id')
            scored = get_scored(index, doc_id)
            # law_id = doc.metadata.get('law_id')
            # scored = get_law_scored(index, law_id)
            score_bm25 = doc.metadata.get('score_bm25')
            rerank_score = doc.metadata.get('rerank_score')
            score = 0
            if score_bm25 is None:
                score = doc.metadata.get('_score')
            page_content = doc.page_content.strip()
            # jw_id = doc.metadata.get('jw_id')
        except Exception as e:
            logger.error(f"get_refs异常: {e}")
            continue

        dt = datetime.now()
        try:
            # ref_score_gpt1 = llm_score(query, page_content, 1)
            # if jw_id is not None:
            #     page_content = query_parent_context(jw_id)
            ref_score_gpt1 = 0
            ref_model = AbtestRefNew(
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
                ref_score_gpt1=0,
                scored=scored,
                ref_score_gpt2=score,
                ref_score_gpt3=score_bm25,
                ref_rerank_score=rerank_score
            )
        except:
            ref_model = AbtestRefNew(
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
                ref_score_gpt1=0,
                scored=scored,
                ref_score_gpt2=score,
                ref_score_gpt3=score_bm25,
                ref_rerank_score=rerank_score
            )

        pg_pool.execute_insert(abtest_ref_table_new, ref_model)

def filter_handle(input, es_filter):
    es_filter = []
    filter_part = ''
    if "科创板" in input:
        if filter_part == '':
            filter_part += '(688)\\d{3}'
        else:
            filter_part += '|(688)\\d{3}'
    if "创业板" in input:
        if filter_part == '':
            filter_part += '(300|301)\\d{3}'
        else:
            filter_part += '|(300|301)\\d{3}'
    if "上交所" in input:
        if filter_part == '':
            filter_part += '(600|601|603|605)\\d{3}'
        else:
            filter_part += '|(600|601|603|605)\\d{3}'
    if "深交所" in input:
        if filter_part == '':
            filter_part += '(000|001|002|003)\\d{3}'
        else:
            filter_part += '|(000|001|002|003)\\d{3}'
    if "北交所" in input:
        if filter_part == '':
            filter_part += '(43|83|87)\\d{4}'
        else:
            filter_part += '|(43|83|87)\\d{4}'
    if filter_part != '':
        es_filter.append({"regexp": {"metadata.stock_code.keyword": "(" + filter_part + "|预披露)"}})
    # es_filter.append({"term": {"_id": "5a550abf-0038-4e7e-900f-ec690a4f8a7e"}})
    return es_filter

def query_to_ref(input: str, query, query_kind: int, batch_id: int, index: int, base_dict: dict,  es_filter: list = list()):
    # query = supplement_input(input)
    # print(f'input:{input}')
    # bm25_list = bm25search(input,top_k,query_kind,es_filter)
    # bm25_list = question_output_bm25_top_k_docs(input, top_k, query_kind, batch_id, index)
    bm25_list = bm25_mutiple_search(query, base_dict, top_k, index, query_kind, batch_id)
    bm25_list = [doc for doc in bm25_list if len(doc.page_content)>=120]
    bm25_rerank_list = bge_ranking(query, bm25_list, False, top_k)
    bm25_rerank_list = [d for d in bm25_rerank_list if d.metadata['rerank_score'] > 0]
    # insert_table(bm25_list, input, index, query, query_kind)
    # question = base_dict['question']
    structured_query = base_dict['eb_structured_query']
    # input = supplement_input(input)
    # print(f'input:{input}')
    # eb_dict = custom_search_v4(query, 2, structured_query, batch_id, index, False)
    # eb_list = []
    # for eb_recall in eb_dict:
    #     eb_list.append(eb_recall['chunk_list'][0])
    eb_list = output_emb_top_k_docs(input, structured_query, top_k, index, query_kind, batch_id, log_flg=False)
    eb_list = [doc for doc in eb_list if len(doc.page_content)>=120]
    eb_rerank_list = bge_ranking(query, eb_list, False, top_k)
    eb_rerank_list = [d for d in eb_rerank_list if d.metadata['rerank_score'] > 0]
    
    rerank(bm25_rerank_list, eb_rerank_list, batch_id, input, index, query, query_kind, 10, 'rrf')


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


def rerank(bm25_list:list, eb_list:list, batch_id, input, index, query, query_kind, top_k, rerank_strategy):
    # if bm25_list is None or len(bm25_list) < 1:
    #     print('bm25数据为空')
    #     return
    # elif eb_list is None or len(eb_list) < 1:
    #     print('eb数据为空')
    #     return
    # subquery_set_size = 1 + len(eb_list)
    # print(f'subquery_set_size:{subquery_set_size}')
    doc_lists = []
    # weights = [round(1/subquery_set_size,2) for i in range(subquery_set_size)]
    # print(f'weights:{weights}')
    doc_lists.append(bm25_list)
    # for eb_list_i in eb_list:
    doc_lists.append(eb_list) 


    # rerank_strategy = "bge_rerank"
    print("Current rerank strategy: ", rerank_strategy)
    # match rerank_strategy:
    #   case "none":
    #     # 直接记录所有召回的文档,并忽略传入的参数top_k
    #     insert_table(doc_lists, input, index, query, query_kind, batch_id)
    #
    #   case "bge_rerank":
    #     all_documents = []
    #     for doc_list in doc_lists:
    #         print(len(doc_list))
    #         for doc in doc_list:
    #             all_documents.append(doc)
    #     sort_list = bge_ranking(input, all_documents, False, top_k)
    #     insert_table(sort_list, input, index, query, query_kind, batch_id)
    #
    #   case "rrf":
    #     ######### rrf rerank start#########
    #     all_documents = set()
    #     for doc_list in doc_lists:
    #         for doc in doc_list:
    #             all_documents.add(doc.page_content)
    #
    #     rrf_score_dic = {doc: 0.0 for doc in all_documents}
    #     weights = [0.5, 0.5]
    #     subquery_set_size = 2
    #
    #     for doc_list, weight in zip(doc_lists, weights):
    #         for rank, doc in enumerate(doc_list, start=1):
    #             rrf_score = weight * (1 / (rank + subquery_set_size*30))
    #             rrf_score_dic[doc.page_content] += rrf_score
    #     # for key,value in rrf_score_dic.items():
    #     #     print(f'key:{key},value:{value}')
    #
    #     # Sort documents by their RRF scores in descending order
    #     sorted_documents = sorted(
    #         rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
    #     )
    #     # Map the sorted page_content back to the original document objects
    #     page_content_to_doc_map = {
    #         doc.page_content: doc for doc_list in doc_lists for doc in doc_list
    #     }
    #     sorted_docs = [
    #         page_content_to_doc_map[page_content] for page_content in sorted_documents
    #     ]
    #     print("sortlen:"+str(len(sorted_docs)))
    #     insert_table(sorted_docs[:top_k], input, index, query, query_kind, batch_id)
    #     ######### rrf rerank end#########
    #
    #   case _:
    #     pass

    if rerank_strategy == "none":
        # 直接记录所有召回的文档,并忽略传入的参数top_k
        insert_table(doc_lists, input, index, query, query_kind, batch_id)
    elif rerank_strategy == "bge_rerank":
        all_documents = []
        for doc_list in doc_lists:
            print(len(doc_list))
            for doc in doc_list:
                all_documents.append(doc)
        sort_list = bge_ranking(input, all_documents, False, top_k)
        insert_table(sort_list, input, index, query, query_kind, batch_id)

    elif rerank_strategy == "rrf":
        ######### rrf rerank start#########
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc.page_content)

        rrf_score_dic = {doc: 0.0 for doc in all_documents}
        weights = [0.5, 0.5]
        subquery_set_size = 2

        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + subquery_set_size * 30))
                rrf_score_dic[doc.page_content] += rrf_score
        # for key,value in rrf_score_dic.items():
        #     print(f'key:{key},value:{value}')

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
        insert_table(sorted_docs[:top_k], input, index, query, query_kind, batch_id)
        ######### rrf rerank end#########

    else:
        pass

# doc_dict = {doc.page_content: doc for doc in all_documents}
    # all_documents_new = doc_dict.values()



def bge_ranking(query: str, doc_lists: list, is_ft: bool=True, top_k:int=30):
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
        for index, doc in enumerate(docs):
            doc.metadata['rerank_score'] = float_scores[index]
            
        # sorted_data = sorted(zip(docs, float_scores), key=lambda x: x[1], reverse=True)
        # for i in sorted_data:
        #     doc_i = i[0]
        #     doc_i.metadata['rerank_score'] = i[1]
        # sorted_docs, _ = zip(*sorted_data)
        return docs[:top_k]


def bm25_mutiple_search(query, base_dict, top_k, q_index, query_kind, batch_id):

    # input = base_dict['question']
    structure_query_dict = base_dict['bm_structured_query']
    # print(structure_query_dict)
    bm25_list = output_bm25_top_k_docs(query, structure_query_dict, top_k, q_index, query_kind, batch_id, False)
    return bm25_list

if __name__ == '__main__':
    start = int(time.time())
    query_kind = 2  # 1 law, 2 sec
    filename = 'sec_1201_1.txt'
    # filename = 'law_0103.txt'
    user_id = 'test_lsl'

    top_k = 30
    es_filter = []

    batch_id = create_batch(filename, 'sec_test_0128', user_id)
    print("batch_id:", batch_id)
    input_file = "../data/{}".format(filename)
    inputs = read_file_lines(input_file)
    index = 0
    # h = json.loads(h)
    # index_list = [32,69,60,94,17,22,75,18,82,99,66,9,89,80]
    index_list = [83, 85]
    dict = extract_base_question_sheet()
    for input in inputs:
        index += 1  # index
        base_dict = dict.get(index)
        # if i in index_list:
        if index == 54:
            question_to_query(input, query_kind, batch_id, index, base_dict)
            logger.info(f"finish i - {index}, input - {input}")
        # else:
        #     break
    

    """
    历史打分相关
    """
    # update_unscored_hanlder([batch_id])

    logger.info(("Finished!!!!!!"))

    end = int(time.time())
    logger.info(f"耗时:{end - start}s")
