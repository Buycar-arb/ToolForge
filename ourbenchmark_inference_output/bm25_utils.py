# RAG相关的工具函数
import bm25s
import jieba
import random
from typing import List, Union
from bm25s.tokenization import Tokenized
import bm25s
from tqdm.auto import tqdm
class BM25Processor:
    def __init__(self):
        pass
    def tokenize(
        self,
        texts,
        return_ids: bool = True,
        show_progress: bool = True,
        leave: bool = False,
    ) -> Union[List[List[str]], Tokenized]:
        if isinstance(texts, str):
            texts = [texts]

        corpus_ids = []
        token_to_index = {}

        for text in tqdm(
            texts, desc="Split strings", leave=leave, disable=not show_progress
        ):

            splitted = jieba.lcut(text)
            doc_ids = []

            for token in splitted:
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)

                token_id = token_to_index[token]
                doc_ids.append(token_id)

            corpus_ids.append(doc_ids)

        # Create a list of unique tokens that we will use to create the vocabulary
        unique_tokens = list(token_to_index.keys())

        vocab_dict = token_to_index

        # Return the tokenized IDs and the vocab dictionary or the tokenized strings
        if return_ids:
            return Tokenized(ids=corpus_ids, vocab=vocab_dict)

        else:
            # We need a reverse dictionary to convert the token IDs back to tokens
            reverse_dict = unique_tokens
            # We convert the token IDs back to tokens in-place
            for i, token_ids in enumerate(
                tqdm(
                    corpus_ids,
                    desc="Reconstructing token strings",
                    leave=leave,
                    disable=not show_progress,
                )
            ):
                corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

            return corpus_ids
    def bm25s_function(self, corpus: List[dict], query: str, top_k:int, language = 'english') -> List[dict]:
        """
        BM25检索函数，支持带元数据的文档
        """
        if language == 'chinese':
            bm25s.tokenizer = self.tokenize
        
        # 提取content用于索引，并建立内容到文档的映射
        content_list = [doc["content"] for doc in corpus]
        content_to_doc = {doc["content"]: doc for doc in corpus}
        
        # 对content进行分词
        corpus_tokens = bm25s.tokenize(content_list)
        
        # 构建检索器实例
        retriever = bm25s.BM25(corpus=content_list)
        
        # 构建倒排索引
        retriever.index(corpus_tokens)
        
        # 查询分词
        query_tokens = bm25s.tokenize(query)
        
        # 检索
        docs, scores = retriever.retrieve(query_tokens, k=top_k)
        
        # 根据检索结果的内容，快速匹配对应的完整文档
        retrieved_docs = []
        for doc_batch in docs:  # docs 是二维数组
            for doc_content in doc_batch:  # 每个元素是文档内容字符串
                if doc_content in content_to_doc:
                    retrieved_docs.append(content_to_doc[doc_content])
        
        return retrieved_docs