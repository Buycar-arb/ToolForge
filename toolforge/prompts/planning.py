"""Stage 3, step 1 — planning prompts.

Given a source record (question, gold answer, supporting passages, gold tool,
reasoning trace), these prompts ask the model to lay out the *skeleton* of the
tool-calling trajectory: which tool to call in each turn, with which arguments,
and which supporting passages that call should surface.

One system prompt per case family:

======  ====================================================================
family  turn structure
======  ====================================================================
``A``   one turn, one tool call
``B``   one turn, several tool calls (same tool repeated, or several tools)
``C``   two turns, one tool call each
``D``   two turns, several tool calls per turn
======  ====================================================================

The output is parsed out of ``<turn_N>`` / ``<tool_call>`` / ``<reference>`` tags
by :mod:`toolforge.stages.dialogue`.
"""

generate_tool_call_user_prompt = '''
# 已知信息
用户问题:{query}
关键上下文:{reference},关键上下文包含了回答问题的关键事实和背景
正确答案:{answer}
需要调用的工具集合:{tools}
问题类型:{type}
工具调用顺序的指导轨迹:{reasoning}
'''
generate_tool_call_system_prompt_A= '''
#角色定位
你是大模型数据构造专家，你将根据数据中的已有信息和指定要求，产出符合要求的高质量的工具调用数据。

# 你能获得的信息
你将会获得:
- 需要解决的问题
- 回答问题所需的关键上下文信息
- 问题的正确答案
- 需要调用的工具集合
- 问题所属的类型
- 工具调用顺序的指导轨迹

# 问题描述
给定的问题是需要经过1轮工具调用才能得到答案的问题。问题类型如下:
case1:回答这个问题只需要一轮工具调用,且只需要调用某一种工具一次

# 任务描述
你需要根据给定的问题、需要调用的工具集合、关键上下文、正确答案，按照给定的问题类型和工具调用顺序的指导轨迹，构造出符合要求的工具调用轨迹。

# 工具调用要求
1.构造的工具调用必须与指导轨迹保持严格一致
2.构造的工具调用中的参数必须严格按照指定的工具要求进行填充,必须拥有`required`数组中的所有字段,这些是回答这个问题的必选字段。

# 输出要求
- 输出分为1轮,分别放在<turn_1>\n...\n</turn_1>中
- 关键上下文信息是这轮工具调用所需的上下文片段,输出保持原本的格式:[{'title': '', 'content': ''},{'title': '', 'content': ''}],每个片段之间换行,不需要输出额外的内容,结果放在<reference>\n...\n</reference>中
- 这一轮的工具调用的内容需要根据实际选择的上下文片段确定
- 工具调用需要放在<tool_call>\n...\n</tool_call>中
# 输出样例
<turn_1>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n</turn_1>
'''

generate_tool_call_system_prompt_B= '''
#角色定位
你是大模型数据构造专家，你将根据数据中的已有信息和指定要求，产出符合要求的高质量的工具调用数据。

# 你能获得的信息
你将会获得:
- 需要解决的问题
- 回答问题所需的关键上下文信息
- 问题的正确答案
- 可调用的工具集合
- 问题所属的类型
- 工具调用顺序的指导轨迹

# 问题描述
给定的问题是需要经过1轮工具调用才能得到答案的问题。问题类型如下:
case2:回答这个问题只需要一轮工具调用,但在这样一轮中需要同时调用某一种工具两次或者两次以上,或者同时调用两种或者两种以上不同的工具

# 任务描述
你需要根据给定的问题、需要调用的工具集合、关键上下文、正确答案，按照给定的问题类型和工具调用顺序的指导轨迹，构造出符合要求的工具调用轨迹。

# 工具调用要求
1.构造的工具调用必须与指导轨迹保持严格一致
2.构造的工具调用中的参数必须严格按照指定的工具要求进行填充,必须拥有`required`数组中的所有字段,这些是回答这个问题的必选字段。

# 输出要求
- 输出分为1轮,分别放在<turn_1>\n...\n</turn_1>中
- 关键上下文信息是这轮工具调用所需的上下文片段,输出保持原本的格式:[{'title': '', 'content': ''},{'title': '', 'content': ''}],每个片段之间换行,不需要输出额外的内容,结果放在<reference>\n...\n</reference>中
- 这一轮的工具调用的内容需要根据实际选择的上下文片段确定
- 工具调用需要放在<tool_call>\n...\n</tool_call>中
# 输出样例
<turn_1>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n</turn_1>
'''

generate_tool_call_system_prompt_C= '''
#角色定位
你是大模型数据构造专家，你将根据数据中的已有信息和指定要求，产出符合要求的高质量的工具调用数据。

# 你能获得的信息
你将会获得:
- 需要解决的问题
- 回答问题所需的关键上下文信息
- 问题的正确答案
- 可调用的工具集合
- 问题所属的类型
- 工具调用顺序的指导轨迹

# 问题描述
给定的问题是需要经过2轮工具调用才能得到答案的复杂问题。问题类型如下:
case4:回答这个问题需要两轮工具调用,第一轮工具调用的结果为第二轮工具调用提供输入信息,但某轮中需要调用多个工具。

# 任务描述
你需要根据给定的问题、需要调用的工具集合、关键上下文、正确答案，按照给定的问题类型和工具调用顺序的指导轨迹，构造出符合要求的工具调用轨迹。

# 工具调用要求
1.构造的工具调用必须与指导轨迹保持严格一致
2.构造的工具调用中的参数必须严格按照指定的工具要求进行填充,必须拥有`required`数组中的所有字段,这些是回答这个问题的必选字段。

# 输出要求
- 输出分为两轮,分别放在<turn_1>\n...\n</turn_1>和<turn_2>\n...\n</turn_2>中
- 从关键上下文信息中合理的选择出每轮工具调用所需的上下文片段并保持原本的格式:[{'title': '', 'content': ''},{'title': '', 'content': ''}],每个片段之间换行,不需要输出额外的内容,结果放在<reference>\n...\n</reference>中
- 每一轮的工具调用的内容需要根据实际选择的上下文片段确定
- 每一轮的工具调用的数量根据实际情况确定
- 关键上下文信息中所有的上下文片段都要合理分配到每轮工具调用中,保证每轮的集合可以和关键上下文信息完全一致,每个上下文片段只能在一轮中使用,不能重复分配
- 工具调用需要放在<tool_call>\n...\n</tool_call>中
# 输出样例
<turn_1>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n</turn_1>
<turn_2>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n</turn_2>
'''
generate_tool_call_system_prompt_D= '''
#角色定位
你是大模型数据构造专家，你将根据数据中的已有信息和指定要求，产出符合要求的高质量的工具调用数据。

# 你能获得的信息
你将会获得:
- 需要解决的问题
- 回答问题所需的关键上下文信息
- 问题的正确答案
- 可调用的工具集合
- 问题所属的类型
- 工具调用顺序的指导轨迹

# 问题描述
给定的问题是需要经过2轮工具调用才能得到答案的复杂问题。问题类型如下:
case3:回答这个问题需要两轮工具调用,第一轮工具调用的结果为第二轮工具调用提供输入信息,每轮只调用一个工具

# 任务描述
你需要根据给定的问题、需要调用的工具集合、关键上下文、正确答案，按照给定的问题类型和工具调用顺序的指导轨迹，构造出符合要求的工具调用轨迹。

# 工具调用要求
1.构造的工具调用必须与指导轨迹保持严格一致
2.构造的工具调用中的参数必须严格按照指定的工具要求进行填充,必须拥有`required`数组中的所有字段,这些是回答这个问题的必选字段。

# 输出要求
- 输出分为两轮,分别放在<turn_1>\n...\n</turn_1>和<turn_2>\n...\n</turn_2>中
- 从关键上下文信息中合理的选择出每轮工具调用所需的上下文片段并保持原本的格式:[{'title': '', 'content': ''},{'title': '', 'content': ''}],每个片段之间换行,不需要输出额外的内容,结果放在<reference>\n...\n</reference>中
- 每一轮的工具调用的内容需要根据实际选择的上下文片段确定
- 每一轮的工具调用的数量根据实际情况确定
- 关键上下文信息中所有的上下文片段都要合理分配到每轮工具调用中,保证每轮的集合可以和关键上下文信息完全一致,每个上下文片段只能在一轮中使用,不能重复分配
- 工具调用需要放在<tool_call>\n...\n</tool_call>中
# 输出样例
<turn_1>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n</turn_1>
<turn_2>\n<tool_call>\n...\n</tool_call>\n<reference>\n...\n</reference>\n</turn_2>
'''
