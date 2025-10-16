# data processor
import json
import random
from typing import List, Dict, Tuple
class DataProcessor:
    @staticmethod
    def load_multihop_data_from_jsonl(message: Dict) -> Tuple[str, List[Dict], List[Dict], str]:
        """Load dataset"""
        if type(message) == dict:
            data = message
        else:
            data = json.loads(message)
        
        query = data["question"]
        answer = data["answer"]
        supporting_facts = data["supporting_facts"]
        context = data["context"]
        case_type = data["route_select"]
        reasoning = data["reasoning"]
        
        # Construct the set of supporting facts
        support_set = set()
        for title, sent_id in supporting_facts:
            support_set.add((title, sent_id))
        
        gold_contents = []
        all_contents = []
        
        for title, sentences in context:
            for sent_id, sentence in enumerate(sentences):
                doc_dict = {
                    "title": title,
                    "content": sentence
                }
                
                if (title, sent_id) in support_set:
                    gold_contents.append(doc_dict)
                else:
                    all_contents.append(doc_dict)
        
        # Remove duplicates
        def deduplicate_docs(docs):
            seen_contents = set()
            unique_docs = []
            for doc in docs:
                if doc["content"] not in seen_contents:
                    seen_contents.add(doc["content"])
                    unique_docs.append(doc)
            return unique_docs
        
        unique_gold_contents = deduplicate_docs(gold_contents)
        unique_all_contents = deduplicate_docs(all_contents)
        
        return query, unique_gold_contents, unique_all_contents, answer, case_type ,reasoning

    @staticmethod
    def deduplicate_rag_results(nested_list):
        """Deduplicate RAG results"""
        if not nested_list:
            return []
        
        if isinstance(nested_list[0], list):
            flat_list = []
            for sublist in nested_list:
                flat_list.extend(sublist)
            nested_list = flat_list
        
        unique_items = {}
        for item in nested_list:
            if isinstance(item, dict) and 'content' in item:
                title = item['content']
                if title not in unique_items:
                    unique_items[title] = item
        result = list(unique_items.values())
        random.shuffle(result)
        return result
    @staticmethod
    def replace_tool_names_in_reasoning(reasoning, good_tool_mapping):

        if not reasoning or not good_tool_mapping:
            return reasoning
        

        tool_mapping_dict = {}
        def extract_mappings(data):
            if isinstance(data, dict):
                if 'original_tool' in data and 'diversity' in data:
                    tool_mapping_dict[data['original_tool']] = data['diversity']
            elif isinstance(data, list):
                for item in data:
                    extract_mappings(item)
        
        extract_mappings(good_tool_mapping)
        

        updated_reasoning = reasoning
        for original_tool, diversity in tool_mapping_dict.items():
            updated_reasoning = updated_reasoning.replace(original_tool, diversity)
        
        if reasoning != updated_reasoning:

            print("1.2:Replace tools in reasoning")
        return updated_reasoning
