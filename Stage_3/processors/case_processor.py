import copy
import ast
import aiohttp
from stage_2_generate.processors.base_processor import BaseProcessor
from stage_2_generate.prompts.conversation_generate_prompts import generate_tool_call_system_prompt_A, generate_tool_call_system_prompt_B, generate_tool_call_system_prompt_C, generate_tool_call_system_prompt_D, generate_tool_call_user_prompt, conversation_generate_system_prompt
from stage_2_generate.prompts.conversation_generate_prompts import A1_user_prompt, A2_user_prompt, A3_user_prompt, A4_user_prompt
from stage_2_generate.prompts.conversation_generate_prompts import B1_user_prompt, B2_user_prompt, B3_user_prompt, B4_user_prompt, B5_user_prompt, B6_user_prompt
from stage_2_generate.prompts.conversation_generate_prompts import C1_user_prompt, C3_user_prompt, C4_user_prompt, C5_user_prompt, C6_user_prompt, C7_user_prompt, C8_user_prompt, C9_user_prompt, C10_user_prompt
from stage_2_generate.prompts.conversation_generate_prompts import D1_user_prompt, D2_user_prompt, D3_user_prompt, D4_user_prompt, D5_user_prompt, D6_user_prompt, D7_user_prompt, D8_user_prompt, D9_user_prompt, D10_user_prompt
from stage_2_generate.prompts.flow_prompts import flow_A1, flow_A2, flow_A3, flow_A4
from stage_2_generate.prompts.flow_prompts import flow_B1, flow_B2, flow_B3, flow_B4, flow_B5, flow_B6
from stage_2_generate.prompts.flow_prompts import flow_C1, flow_C3, flow_C4, flow_C5, flow_C6, flow_C7, flow_C8, flow_C9, flow_C10
from stage_2_generate.prompts.flow_prompts import flow_D1, flow_D2, flow_D3, flow_D4, flow_D5, flow_D6, flow_D7, flow_D8, flow_D9, flow_D10
from config.settings import RAG_MIN_TOP_K, RAG_MAX_TOP_K, TOOL_BANK_DIR

import json
class CaseA1Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_A + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)

            tool_response1_good =[]
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)


            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_A1
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": A1_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,   
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated",flush=True)
            return full_messages 
class CaseA2Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_A + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            tool_response1_bad =[]
            tool_response1_good =[]
            
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_A2
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": A2_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,   
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages   
class CaseA3Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_A + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            tool_response1_bad =[]
            tool_response1_good =[]
            
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_A3
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": A3_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                tool_list = self.tool_list, 
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,   
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 
class CaseA4Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt_general},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_A + self.tool_prompt_general
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            
            tool_response1_bad_1 =[]
            tool_response1_bad_2 =[]
            tool_response1_bad_3 =[]
            tool_response1_good =[]
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K * 3, RAG_MAX_TOP_K * 3)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                third = len(rag) // 3
                tool_response1_bad_3.append(copy.deepcopy(rag[:third]))
                tool_response1_bad_2.append(copy.deepcopy(rag[third:2*third]))
                tool_response1_bad_1.append(copy.deepcopy(rag[2*third:])) 
                tool_response1_good.append(copy.deepcopy(rag[:third]))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad_1 = self.deduplicate_rag_results(tool_response1_bad_1)
            tool_response1_bad_2 = self.deduplicate_rag_results(tool_response1_bad_2)
            tool_response1_bad_3 = self.deduplicate_rag_results(tool_response1_bad_3)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)
            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_A4
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": A4_user_prompt.format(query = self.query,
                                                tool_list = self.tool_list_general, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad_1,
                                                error_content_2 = tool_response1_bad_2,
                                                error_content_3 = tool_response1_bad_3,
                                                general_tool = self.general_tool_name,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad_1,
                        tool_response1_bad_2,
                        tool_response1_bad_3,
                        tool_response1_good,   
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_general_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages     
class CaseB1Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_B + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)

            tool_response1_good =[]
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)


 

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_B1
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": B1_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,   
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 

class CaseB2Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_B + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            tool_response1_bad =[]
            tool_response1_good =[]
            
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_B2
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": B2_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,   
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 

class CaseB3Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_B + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            tool_response1_bad =[]
            tool_response1_good =[]
            
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_B3
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": B3_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,   
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 
class CaseB4Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_B + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            tool_response1_bad =[]
            tool_response1_good =[]
            
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_B4
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": B4_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                tool_list = self.tool_list,  
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,   
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 
        
class CaseB5Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_B + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            tool_response1_bad =[]
            tool_response1_good =[]
            
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_B5
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": B5_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                tool_list = self.tool_list,  
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,   
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 
class CaseB6Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt_general},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_B + self.tool_prompt_general
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            
            tool_response1_bad_1 =[]
            tool_response1_bad_2 =[]
            tool_response1_bad_3 =[]
            tool_response1_good =[]
            all_reference1_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K * 3, RAG_MAX_TOP_K * 3)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                third = len(rag) // 3
                tool_response1_bad_3.append(copy.deepcopy(rag[:third]))
                tool_response1_bad_2.append(copy.deepcopy(rag[third:2*third]))
                tool_response1_bad_1.append(copy.deepcopy(rag[2*third:])) 
                tool_response1_good.append(copy.deepcopy(rag[:third]))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad_1 = self.deduplicate_rag_results(tool_response1_bad_1)
            tool_response1_bad_2 = self.deduplicate_rag_results(tool_response1_bad_2)
            tool_response1_bad_3 = self.deduplicate_rag_results(tool_response1_bad_3)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)
            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_B6
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": B6_user_prompt.format(query = self.query,
                                                tool_list = self.tool_list_general, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad_1,
                                                error_content_2 = tool_response1_bad_2,
                                                error_content_3 = tool_response1_bad_3,
                                                general_tool = self.general_tool_name,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad_1,
                        tool_response1_bad_2,
                        tool_response1_bad_3,
                        tool_response1_good,   
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping, "argument_tool_bank": self.simulate_recall_tools_general_json},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_general_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 
class CaseC1Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)


            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                # reference2[i] = self.convert_reference_to_dict_list(reference2[i])
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            # all_reference1_data.extend(all_reference2_data)

            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

 

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_C1
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C1_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages 
class CaseC3Processor(BaseProcessor): 
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer,type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_C3
            # print(flow)
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C3_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages   
class CaseC4Processor(BaseProcessor):
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ] 
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)      
            
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(tool_call_1) ==0 or len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):

                    tool_response1_good[i].append(reference_data[j])
       
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference1_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])

            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)


            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_C4
            # print(flow)
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C4_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)
            save_rags = [
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
    
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
        
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseC5Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text":  generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(tool_call_1) ==0 or len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error:tool_call return 0 or reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = [] 
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])     
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])

            
            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_C5
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C5_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_1 = tool_response1_bad,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            result = self.parse_jsonl_string(response['content'])
            print("2.4:Parse the model outputs and construct the final JSON data")
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)
            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseC6Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = [] 
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
           
            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_C6
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C6_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_1 = tool_response1_bad,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            result = self.parse_jsonl_string(response['content'])
            print("2.4:Parse the model outputs and construct the final JSON data")
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)
            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseC7Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)

            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response1_good =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_C7
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C7_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseC8Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)

            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response1_good =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_C8
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C8_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseC9Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)

            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt_general},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt_general
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None

            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 
            tool_response1_good =[]
            tool_response2_bad_1 =[]
            tool_response2_bad_2 =[]
            tool_response2_bad_3 =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])     
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K * 3, RAG_MAX_TOP_K * 3)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                third = len(rag) // 3
                tool_response2_bad_3.append(copy.deepcopy(rag[:third]))
                tool_response2_bad_2.append(copy.deepcopy(rag[third:2*third]))
                tool_response2_bad_1.append(copy.deepcopy(rag[2*third:])) 
                tool_response2_good.append(copy.deepcopy(rag[:third]))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_bad_1 = self.deduplicate_rag_results(tool_response2_bad_1)
            tool_response2_bad_2 = self.deduplicate_rag_results(tool_response2_bad_2)
            tool_response2_bad_3 = self.deduplicate_rag_results(tool_response2_bad_3)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
            print("2.3:Start generating concrete data...")
            print(self.general_tool_name)
            system_prompt = conversation_generate_system_prompt
            flow = flow_C9
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C9_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list_general, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response2_bad_1,
                                                error_content_2 = tool_response2_bad_2,
                                                error_content_3 = tool_response2_bad_3,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                general_tool = self.general_tool_name,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,
                        tool_response2_bad_1,
                        tool_response2_bad_2,
                        tool_response2_bad_3,
                        tool_response2_good     
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_general_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseC10Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:

            # print("mammamammamama")
            # print(self.tool_prompt_general)
            # print("-----")
            # print(self.simulate_recall_tools_general_json)
            # print("mammamammamama")
            # print(f"通用工具\n{self.general_tool_name}")
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt_general},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_C + self.tool_prompt_general
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None

            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 
            tool_response1_good =[]
            tool_response1_bad_1 =[]
            tool_response1_bad_2 =[]
            tool_response1_bad_3 =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []

            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K * 3, RAG_MAX_TOP_K * 3)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                third = len(rag) // 3
                tool_response1_bad_3.append(copy.deepcopy(rag[:third]))
                tool_response1_bad_2.append(copy.deepcopy(rag[third:2*third]))
                tool_response1_bad_1.append(copy.deepcopy(rag[2*third:])) 
                tool_response1_good.append(copy.deepcopy(rag[:third]))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])
            tool_response1_bad_1 = self.deduplicate_rag_results(tool_response1_bad_1)
            tool_response1_bad_2 = self.deduplicate_rag_results(tool_response1_bad_2)
            tool_response1_bad_3 = self.deduplicate_rag_results(tool_response1_bad_3)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)
            
            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
            print("2.3:Start generating concrete data...")
            # print(self.general_tool_name)
            system_prompt = conversation_generate_system_prompt
            flow = flow_C10
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": C10_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list_general, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad_1,
                                                error_content_2 = tool_response1_bad_2,
                                                error_content_3 = tool_response1_bad_3,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                general_tool = self.general_tool_name,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad_1,
                        tool_response1_bad_2,
                        tool_response1_bad_3,
                        tool_response1_good,
                        tool_response2_good     
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_general_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseD1Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j]) 
            # To avoid duplicates in the original data, we add a deduplication step
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)


            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                # reference2[i] = self.convert_reference_to_dict_list(reference2[i])
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            # all_reference1_data.extend(all_reference2_data)

            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

 

            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_D1
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D1_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages    
class CaseD2Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_good =[]
            all_reference1_data = []  
            all_reference2_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])
            for i in range(len(tool_call_2)):
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])       
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            system_prompt = conversation_generate_system_prompt
            flow = flow_D2

            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D2_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                gold_content_1 = tool_response1_good,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages    
class CaseD3Processor(BaseProcessor): 
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer,type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    

            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])


            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_D3
            # print(flow)
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D3_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages  
class CaseD4Processor(BaseProcessor):
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ] 
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)      
            
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(tool_call_1) ==0 or len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 
            tool_response1_bad =[]
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = []  
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):

                    tool_response1_good[i].append(reference_data[j])
       
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference1_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])

            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)


            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_D4
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D4_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)
            save_rags = [
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
        
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags, "answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!", flush=True)
            return full_messages
class CaseD5Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text":  generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(tool_call_1) ==0 or len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                return None 

            tool_response1_bad =[]
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = [] 
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])     
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])

            
            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_D5
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D5_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_1 = tool_response1_bad,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            result = self.parse_jsonl_string(response['content'])
            print("2.4:Parse the model outputs and construct the final JSON data")
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)
            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseD6Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None
            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response2_bad =[]
            tool_response1_good =[]
            tool_response2_good = []
            all_reference1_data = []  
            all_reference2_data = [] 
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_bad.append(copy.deepcopy(rag))
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_bad = self.deduplicate_rag_results(tool_response2_bad)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
           
            print("2.3:Start generating concrete data...")

            system_prompt = conversation_generate_system_prompt
            flow = flow_D6
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D6_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                gold_content_2 = tool_response2_good,
                                                error_content_1 = tool_response1_bad,
                                                error_content_2 = tool_response2_bad,
                                                answer = self.answer,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            result = self.parse_jsonl_string(response['content'])
            print("2.4:Parse the model outputs and construct the final JSON data")
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)
            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_bad,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseD7Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)

            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response1_good =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_D7
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D7_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list,  
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseD8Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")

            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None


            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)

            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 

            tool_response1_bad =[]
            tool_response1_good =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_bad.append(copy.deepcopy(rag))
                tool_response1_good.append(copy.deepcopy(rag))

                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])      
            tool_response1_bad = self.deduplicate_rag_results(tool_response1_bad)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], 5, 10)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)

            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_D8
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D8_user_prompt.format(query = self.query, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad,
                        tool_response1_good,
                        tool_response2_good      
                        ]
            print("2.5:In this case, the construction of the tool_call section's judge is not required, so this step is skipped")
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": "Don't need to check"},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseD9Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt_general},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt_general
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None

            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 
            tool_response1_good =[]
            tool_response2_bad_1 =[]
            tool_response2_bad_2 =[]
            tool_response2_bad_3 =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []
            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                tool_response1_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])     
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)

            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K * 3, RAG_MAX_TOP_K * 3)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                third = len(rag) // 3
                tool_response2_bad_3.append(copy.deepcopy(rag[:third]))
                tool_response2_bad_2.append(copy.deepcopy(rag[third:2*third]))
                tool_response2_bad_1.append(copy.deepcopy(rag[2*third:])) 
                tool_response2_good.append(copy.deepcopy(rag[:third]))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_bad_1 = self.deduplicate_rag_results(tool_response2_bad_1)
            tool_response2_bad_2 = self.deduplicate_rag_results(tool_response2_bad_2)
            tool_response2_bad_3 = self.deduplicate_rag_results(tool_response2_bad_3)
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
            print("2.3:Start generating concrete data...")
            # print(self.general_tool_name)
            system_prompt = conversation_generate_system_prompt
            flow = flow_D9
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D9_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list_general, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response2_bad_1,
                                                error_content_2 = tool_response2_bad_2,
                                                error_content_3 = tool_response2_bad_3,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                general_tool = self.general_tool_name,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_good,
                        tool_response2_bad_1,
                        tool_response2_bad_2,
                        tool_response2_bad_3,
                        tool_response2_good     
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_general_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages
class CaseD10Processor(BaseProcessor):        
    async def process(self):
        async with aiohttp.ClientSession() as session:
            user_prompt = self.user_prompt.format(self.query)
            save_messages = [
                {"role": "system", "content": self.system_prompt + self.tool_prompt_general},
                {"role": "user", "content": user_prompt}
            ]
            print("2.1:Generate required tool_calls and golden contexts for each round based on reasoning guidance")
            system_prompt = generate_tool_call_system_prompt_D + self.tool_prompt_general
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": generate_tool_call_user_prompt.format(query = self.query, tools = self.good_tool_content, reference = self.gold_contents, answer = self.answer, type = self.wheel_type, reasoning = self.reasoning)
                }]}
            ]    
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            trace = response['content']
            if not trace or trace.strip() == "":
                return None

            print("2.2:Extract results from Step 2.1 and integrate them with RAG to construct tool contents for each round")
            turn_1 = self.extract_tags_as_str_list(trace,"turn_1",False)
            turn_2 = self.extract_tags_as_str_list(trace,"turn_2",False)
            tool_call_1 = self.extract_tags_as_str_list(turn_1,"tool_call",True)
            tool_call_2 = self.extract_tags_as_str_list(turn_2,"tool_call",True)
            reference1 = self.extract_tags_as_str_list(turn_1,"reference",True)
            reference2 = self.extract_tags_as_str_list(turn_2,"reference",True)
            if len(reference1) < len(tool_call_1) or len(reference2) < len(tool_call_2):
                print(f"Judge error：reference1({len(reference1)}) < tool_call_1({len(tool_call_1)}) or reference2({len(reference2)}) < tool_call_2({len(tool_call_2)})")
                return None 
            tool_response1_good =[]
            tool_response1_bad_1 =[]
            tool_response1_bad_2 =[]
            tool_response1_bad_3 =[]
            tool_response2_good =[]
            all_reference1_data = []  
            all_reference2_data = []

            for i in range(len(tool_call_1)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_1[i])["arguments"]["query"], RAG_MIN_TOP_K * 3, RAG_MAX_TOP_K * 3)
                reference_data = ast.literal_eval(reference1[i])
                all_reference1_data.extend(reference_data)
                third = len(rag) // 3
                tool_response1_bad_3.append(copy.deepcopy(rag[:third]))
                tool_response1_bad_2.append(copy.deepcopy(rag[third:2*third]))
                tool_response1_bad_1.append(copy.deepcopy(rag[2*third:])) 
                tool_response1_good.append(copy.deepcopy(rag[:third]))
                for j in range(len(reference_data)):
                    tool_response1_good[i].append(reference_data[j])
            tool_response1_bad_1 = self.deduplicate_rag_results(tool_response1_bad_1)
            tool_response1_bad_2 = self.deduplicate_rag_results(tool_response1_bad_2)
            tool_response1_bad_3 = self.deduplicate_rag_results(tool_response1_bad_3)
            tool_response1_good = self.deduplicate_rag_results(tool_response1_good)
            
            for i in range(len(tool_call_2)):
                rag = self.bm25s_function(self.all_contents, json.loads(tool_call_2[i])["arguments"]["query"], RAG_MIN_TOP_K, RAG_MAX_TOP_K)
                reference_data = ast.literal_eval(reference2[i])
                all_reference2_data.extend(reference_data)
                tool_response2_good.append(copy.deepcopy(rag))
                for j in range(len(reference_data)):
                    tool_response2_good[i].append(reference_data[j])
            tool_response2_good = self.deduplicate_rag_results(tool_response2_good)
            print("2.3:Start generating concrete data...")
            system_prompt = conversation_generate_system_prompt
            flow = flow_D10
            messages = [
                {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": D10_user_prompt.format(query = self.query, 
                                                tool_list = self.tool_list_general, 
                                                right_response = self.reasoning,
                                                right_tool_1 = tool_call_1,
                                                right_tool_2 = tool_call_2,
                                                gold_content_1 = tool_response1_good,
                                                error_content_1 = tool_response1_bad_1,
                                                error_content_2 = tool_response1_bad_2,
                                                error_content_3 = tool_response1_bad_3,
                                                gold_content_2 = tool_response2_good,
                                                answer = self.answer,
                                                general_tool = self.general_tool_name,
                                                flow=flow,
                                                )
                }]}
            ]
            response = await self.call_claude_api(session, self.model, self.temperature, self.max_tokens, messages, system_prompt)
            print("2.4:Parse the model outputs and construct the final JSON data")
            result = self.parse_jsonl_string(response['content'])
            new_list = []
            for item in result:
                if item['role'] != 'user':
                    new_list.append(item)
            save_messages.extend(new_list)

            save_rags = [
                        tool_response1_bad_1,
                        tool_response1_bad_2,
                        tool_response1_bad_3,
                        tool_response1_good,
                        tool_response2_good     
                        ]
            print("2.5:Begin constructing the judge for the tool_call section")
            tools_dir = TOOL_BANK_DIR
    
            argument_check = self.get_grouped_tool_calls_hybrid(save_messages, tools_dir)
            
            full_messages = [
                {"messages": save_messages},
                {"rags": save_rags,"answer": self.answer, "reasoning": self.reasoning, "good_tool_mapping" : self.good_tool_mapping},
                {"argument_check": argument_check},
                {
                "argument_all_reference": [
                        {"turn": 1, "data": all_reference1_data},
                        {"turn": 2, "data": all_reference2_data}
                    ]
                },
                {"argument_tool_bank": self.simulate_recall_tools_general_json}
            ]
            print("2.6:This JSON record has been successfully generated!",flush=True)
            return full_messages