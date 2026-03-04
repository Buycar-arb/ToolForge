from openai import OpenAI
import json
import re
import uuid
from typing import Dict, List, Any, Optional
from bm25_utils import BM25Processor
bm25_processor = BM25Processor()
# ======== 配置区 ========
input_path = "your test dataset path"
output_path = "your output path" # 输出文件路径

client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="EMPTY"
)


# ======== 步骤2: 解析 tool_call ========
def parse_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    从模型输出的 content 中解析出所有的 tool_call
    
    Args:
        content: 模型输出的文本内容，可能包含多个 <tool_call>...</tool_call>
    
    Returns:
        List[Dict]: 解析出的 tool_call 列表，每个元素包含 id, name, arguments
    """
    tool_calls = []
    
    # 使用正则表达式匹配所有 <tool_call>...</tool_call>
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # 解析 JSON 格式的 tool_call 内容
            call_data = json.loads(match.strip())
            tool_calls.append({
                'id': f'call_{uuid.uuid4().hex[:8]}',  # 生成唯一ID
                'name': call_data['name'],
                'arguments': call_data['arguments']
            })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse tool_call: {e}")
            print(f"Content: {match.strip()}")
            continue
    
    return tool_calls


# ======== 步骤4: 将 arguments 转换为 query 字符串 ========
def arguments_to_query(arguments: Dict[str, Any]) -> str:
    """
    将 tool_call 的 arguments 转换为格式化的 query 字符串
    
    Args:
        arguments: 函数参数字典
    
    Returns:
        str: 格式化后的 query 字符串
    """
    query_parts = []
    for param_name, param_value in arguments.items():
        if param_value is not None:
            if isinstance(param_value, list):
                # 列表类型：用逗号连接
                formatted_value = ', '.join(str(item) for item in param_value)
            else:
                # 其他类型：直接转为字符串
                formatted_value = str(param_value)
            query_parts.append(f"{param_name}: {formatted_value}")
    
    query = '\n'.join(query_parts) if query_parts else ''
    return query

# ======== 工具: 格式化检索结果 ========
def format_docs(docs: List[Dict[str, Any]], start_idx: int = 1) -> str:
    """
    将检索到的文档列表格式化为模型可读的字符串
    start_idx 用于跨多次合并时保持编号连续
    """
    formatted = []
    for idx, doc in enumerate(docs, start_idx):
        title = doc.get("title", "")
        text = doc.get("content", "")
        formatted.append(f"**{idx}**\ntitle: {title}\ncontent: {text}")
    # 以空行分隔每条文档，形如 "**1**\ntitle: ...\ncontent: ...\n\n**2**..."
    return "\n\n".join(formatted)

# 检测 <answer>...</answer>
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# ======== 主逻辑 ========
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        # 构建bm25语料库
        all_contents = []
        original_query = data[6]["question"]
        golden_answer = data[6]["answer"]
        
        # 初始化 messages（用于保存最终对话历史）
        initial_messages = [
            {"role": "system", "content": data[1]["messages"][0]["content"]},
            {"role": "user", "content": data[1]["messages"][1]["content"]}
        ]
        for title, sentences in data[6]["context"]:
            for sent_id, sentence in enumerate(sentences):
                doc_dict = {
                    "title": title,
                    "content": sentence
                }
                all_contents.append(doc_dict)
        resp = client.chat.completions.create(
            model="history-8B",
            messages=[
                {"role": "system", "content": data[1]["messages"][0]["content"]},
                {"role": "user", "content": data[1]["messages"][1]["content"]}
            ],
            max_tokens=8192,
        )
        assistant_round = 1
        tool_round = 1
        llm_result = resp.choices[0].message.content

        print(f"\n{'='*60}")
        print(f"===== Sample {i} =====")
        print(f"{'='*60}\n")
        print(f"[Assistant 第{assistant_round}轮输出]\n{llm_result}\n")
        
        # 步骤2: 解析 tool_call
        tool_calls = parse_tool_calls(llm_result)
        
        # 初始化最终 messages（包含初始对话和首轮 assistant 输出）
        final_messages = initial_messages + [{"role": "assistant", "content": llm_result}]
        
        if tool_calls:
            # ===== 追加对话并继续生成 =====
            messages = final_messages.copy()  # 使用已初始化的 messages

            # 把检索结果喂回去：多个 tool_call 的检索结果先合并，再作为每个 tool_call 的回复
            per_results = []
            running_idx = 1
            for call_idx, call in enumerate(tool_calls):
                per_query = arguments_to_query(call["arguments"])
                per_docs = bm25_processor.bm25s_function(
                    corpus=all_contents,
                    query=per_query,
                    top_k=10,
                    language="english"
                )
                formatted_block = format_docs(per_docs, start_idx=running_idx)
                running_idx += len(per_docs)
                per_results.append(formatted_block)

            merged_results = "\n\n".join(per_results)
            print(f"[Tool 第{tool_round}轮合并结果]\n{merged_results}\n")
            # for call in tool_calls:
            messages.append({
                "role": "tool",
                "content": merged_results
            })
            tool_round += 1

            max_iterations = 5
            for step in range(max_iterations):
                follow_resp = client.chat.completions.create(
                    model="history-8B",
                    messages=messages,
                    max_tokens=2048,
                )
                follow_content = follow_resp.choices[0].message.content or ""
                assistant_round += 1
                print(f"[Assistant 第{assistant_round}轮输出]\n{follow_content}\n")
                messages.append({"role": "assistant", "content": follow_content})

                # 检查答案
                ans = ANSWER_PATTERN.search(follow_content)
                if ans:
                    print(f"[找到答案] {ans.group(1).strip()}")
                    if golden_answer == ans.group(1).strip():
                        print("此次回答正确哦😄")
                    else:
                        print("这个模型差点意思没有回答正确哦😢")
                    break

                # 如果还有新的 tool_call，则继续检索并追加
                new_calls = parse_tool_calls(follow_content)
                if not new_calls:
                    print("[结束] 无更多 tool_call 且未发现答案")
                    break
                new_results = []
                running_idx = 1
                for call in new_calls:
                    new_query = arguments_to_query(call["arguments"])
                    print(f"new_query{new_query}")
                    new_docs = bm25_processor.bm25s_function(
                        corpus=all_contents,
                        query=new_query,
                        top_k=10,
                        language="english"
                    )
                    # 保持编号连续：从已有 messages 的 tool 内容数估算起始编号
                    # 这里简单从1重新编号即可，如需严格连续，可额外维护计数器
                    formatted_new = format_docs(new_docs, start_idx=running_idx)
                    running_idx += len(new_docs)
                    new_results.append(formatted_new)
                    # print(f"[Tool 第{tool_round}轮结果 对应 {call['name']}]")
                    # print(formatted_new + "\n")
                merged_new_results = "\n\n".join(new_results)
                print(f"[Tool 第{tool_round}轮合并结果]\n{merged_new_results}\n")
                messages.append({
                    "role": "tool",
                    "content": merged_new_results
                })
                # print(f"第{tool_round}轮结果{messages}")
                tool_round += 1
            
            # 更新最终 messages（包含所有对话历史）
            final_messages = messages
        else:
            print("[解析] 未找到 tool_call")
            # 如果没有 tool_call，final_messages 已经包含初始对话和首轮 assistant 输出
        
        # 保存结果到 jsonl 文件
        result_record = {
            "sample_id": i,
            "original_query": original_query,
            "golden_answer": golden_answer,
            "messages": final_messages
        }
        fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        print(f"[已保存] Sample {i} 的结果到 {output_path}")
       
        # break  # 只处理第一个样本
