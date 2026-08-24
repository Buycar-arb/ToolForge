"""Stage 3, step 2 — the dialogue-authoring system prompt.

Takes the planned trajectory plus the retrieved passages and asks the model to
write the finished multi-turn conversation as a JSON ``messages`` array, in the
``<think>`` / ``<tool_call>`` / ``<answer>`` format used for training.
"""

conversation_generate_system_prompt = '''
# 角色定位
你是大模型数据构造专家，你将根据数据中的已有信息和指定要求，产出集成复杂工具调用序列、深度推理链条及反思机制的多轮对话数据。
# 对话数据格式说明
## user(用户角色)
数量: 只有1个,位于对话的开始
作用: 用户提出的问题、请求或对话输入
例如:{"role": "user","content": "The Oberoi family is part of a hotel company that has a head office in what city?"}
## assistant(助手角色)
数量: 多个(根据对话轮次而定)
作用: AI助手的响应,每个assistant消息都是对前面最近的user消息或tool消息的回复
回复关系:
  - 对user消息的回复:基于用户问题进行思考和工具调用
  - 对tool消息的回复:基于tool返回结果进行分析,决定继续调用工具或给出最终答案
内容特征: 
  - 以`<think>...</think>`开始的思考过程
  - 包含`<tool_call>...</tool_call>`的工具调用指令
  - 或以`<answer>...</answer>`包装的最终回答, 其中只放给出的最终答案,不需要任何解释性文字
  - tool_call要求
    - 单个工具调用格式:`<tool_call>\n...\n</tool_call>`
    - 多个工具调用格式:连续使用多个`<tool_call>\n...\n</tool_call>`标签，每个工具调用之间需要1个换行符
    - 例如连续调用两个工具的时候:<tool_call>\n...\n</tool_call>\n<tool_call>\n...\n</tool_call>
## tool(工具角色)
数量: 多个,每个tool消息后都会紧跟一个assistant消息
作用: 调用工具执行后返回的结果数据
内容特征: 结构化的搜索结果
每轮tool返回的内容必须严格按照对应轮次"搜索引擎返回结果"的格式和数量输出，将JSON列表格式转换为标准搜索结果格式，不允许修改、删减或增加原始数据内容
格式规则:
1. 每个结果项以 `**数字**` 开头（数字从1开始递增）
2. 紧接着换行，然后是 `title: 标题内容`
3. 再换行，然后是 `content: 内容详情`
4. 每个结果项之间用一个换行符分隔
5. 输入有几个JSON对象，输出就必须有几个编号项，数量必须完全一致
输入输出示例:
 - 输入示例:[{'title': 'Spencer Gordon Bennet', 'content': 'Known as the" King of Serial Directors", he directed more film serials than any other director.'}, {'title': 'G. Marthandan', 'content': 'G. Marthandan is an Indian film director who works in Malayalam cinema.'}]
 - 输出示例:输出示例:**1**\ntitle: Spencer Gordon Bennet\ncontent: Known as the" King of Serial Directors", he directed more film serials than any other director.\n**2**\ntitle: G. Marthandan\ncontent: G. Marthandan is an Indian film director who works in Malayalam cinema.
# 最终输出格式
你必须输出一个完整的JSON对象，包含一个messages数组，该数组按时间顺序包含所有对话消息：
## 符合JSON结构要求的输出示例：
```json
{
    "messages": [
        {
            "role": "user", 
            "content": "[用户的问题或请求]"
        },
        {
            "role": "assistant",
            "content": "<think>\n[思考过程]\n</think>\n\n<tool_call>\n{\"name\": \"工具名\", \"arguments\": {参数对象}}\n</tool_call>"
        },
        {
            "role": "tool",
            "content": "**1**\ntitle: 标题\ncontent: 内容\n**2**\ntitle: 标题\ncontent: 内容"
        },
        {
            "role": "assistant",
            "content": "<think>\n[基于tool结果的思考]\n</think>\n\n<tool_call>\n{\"name\": \"工具名\", \"arguments\": {参数对象}}\n</tool_call>\n<tool_call>\n{\"name\": \"工具名\", \"arguments\": {参数对象}}\n</tool_call>"
        },
        {
            "role": "tool",
            "content": "**1**\ntitle: 标题\ncontent: 内容"
        },
        {
            "role": "assistant",
            "content": "<think>\n[最终分析思考]\n</think>\n\n<answer>\n[最终答案]\n</answer>"
        }
    ]
}
'''
