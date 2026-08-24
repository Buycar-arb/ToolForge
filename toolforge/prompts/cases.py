"""Stage 3, step 2 — the per-case user prompts.

One template per dialogue case.  They differ only in which slots they expose
(gold passages per turn, distractor passages, the fallback tool, the candidate
tool list); :data:`CASE_USER_PROMPTS` maps a case id to its template and
:mod:`toolforge.stages.cases` declares which slots each one needs.
"""

A1_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

A2_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

A3_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

A4_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
通用搜索工具为：{general_tool}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用第1次错误的检索信息：{error_content_1}
 - 第1步工具调用第2次错误的检索信息：{error_content_2}
 - 第1步工具调用第3次不充分的检索信息：{error_content_3}
 - 第1步工具调用正确的检索信息：{gold_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

B1_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

B2_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

B3_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

B4_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

B5_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

B6_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
通用搜索工具为：{general_tool}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是这轮搜索引擎返回的结果:
 - 第1步工具调用第1次错误的检索信息：{error_content_1}
 - 第1步工具调用第2次错误的检索信息：{error_content_2}
 - 第1步工具调用第3次不充分的检索信息：{error_content_3}
 - 第1步工具调用正确的检索信息：{gold_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''


# 缺东西-ch
C1_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
D1_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

D2_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
# user_C2 = '''
# 现在，请帮我构造我需要的多轮对话数据，你需要的信息如下所示。
# 问题为：{query}
# 正确的工具调用顺序为：{right_response}
# 第1步正确的工具调用为：{right_tool_1}
# 第2步正确的工具调用为：{right_tool_2}
# 第1步工具调用正确的检索信息为：{gold_content_1}
# 最终的答案为：{answer}
# 模拟的对话流程为:{flow}
# '''
C3_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
D3_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
# user_C3 = '''
# 现在，请帮我构造我需要的多轮对话数据，你需要的信息如下所示。
# 问题为：{query}
# 正确的工具调用顺序为：{right_response}
# 工具列表为：{tool_list}
# 第1步正确的工具调用为：{right_tool_1}
# 第2步正确的工具调用为：{right_tool_2}
# 第1步工具调用正确的检索信息为：{gold_content_1}
# 第2步工具调用正确的检索信息为：{gold_content_2}
# 第2步工具调用错误的检索信息为：{error_content_2}
# 最终的答案为：{answer}
# 模拟的对话流程为:{flow}
# '''
C4_user_prompt = '''
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
D4_user_prompt = '''
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
# 缺东西-ch
C5_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

D5_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

C6_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

D6_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
 - 第2步工具调用错误的检索信息：{error_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

C7_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

D7_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

C8_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

D8_user_prompt = '''
# 参考信息
问题为：{query}
正确的工具调用顺序：{right_response}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第1步工具调用错误的检索信息：{error_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

# 改东西-ch
C9_user_prompt = '''
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
通用搜索工具为：{general_tool}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用第1次错误的检索信息：{error_content_1}  # 工具错误导致
 - 第2步工具调用第2次错误的检索信息：{error_content_2}  # 参数错误导致
 - 第2步工具调用第3次错误的检索信息：{error_content_3}  # 专业工具无法检索到导致
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''
D9_user_prompt = '''
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
通用搜索工具为：{general_tool}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用第1次错误的检索信息：{error_content_1}  # 工具错误导致
 - 第2步工具调用第2次错误的检索信息：{error_content_2}  # 参数错误导致
 - 第2步工具调用第3次错误的检索信息：{error_content_3}  # 专业工具无法检索到导致
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

C10_user_prompt = '''
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
通用搜索工具为：{general_tool}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用第1次错误的检索信息：{error_content_1}  # 工具错误导致
 - 第1步工具调用第2次错误的检索信息：{error_content_2}  # 参数错误导致
 - 第1步工具调用第3次错误的检索信息：{error_content_3}  # 专业工具无法检索到导致
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''

D10_user_prompt = '''
问题为：{query}
正确的工具调用顺序：{right_response}
模拟选用错误工具时可选的工具列表：{tool_list}
通用搜索工具为：{general_tool}
工具调用描述如下：'name'字段表示工具名称,是字符串类型,'arguments'字段表示工具参数,是对象类型
第1步正确的工具调用：{right_tool_1}
第2步正确的工具调用：{right_tool_2}
以下是每轮搜索引擎返回的结果:
 - 第1步工具调用第1次错误的检索信息：{error_content_1}  # 工具错误导致
 - 第1步工具调用第2次错误的检索信息：{error_content_2}  # 参数错误导致
 - 第1步工具调用第3次错误的检索信息：{error_content_3}  # 专业工具无法检索到导致
 - 第1步工具调用正确的检索信息：{gold_content_1}
 - 第2步工具调用正确的检索信息：{gold_content_2}
最终答案：{answer}
# 回答要求
请基于参考信息生成多轮对话，需要同时满足以下要求：
1.整体对话流程必须遵循"正确的工具调用顺序"中的工具调用顺序和逻辑
2.对话过程要符合以下推理模式：{flow}
'''


#: ``case id -> user prompt template`` for all 29 dialogue cases.
CASE_USER_PROMPTS: dict[str, str] = {
    "case_A1": A1_user_prompt,
    "case_A2": A2_user_prompt,
    "case_A3": A3_user_prompt,
    "case_A4": A4_user_prompt,
    "case_B1": B1_user_prompt,
    "case_B2": B2_user_prompt,
    "case_B3": B3_user_prompt,
    "case_B4": B4_user_prompt,
    "case_B5": B5_user_prompt,
    "case_B6": B6_user_prompt,
    "case_C1": C1_user_prompt,
    "case_C3": C3_user_prompt,
    "case_C4": C4_user_prompt,
    "case_C5": C5_user_prompt,
    "case_C6": C6_user_prompt,
    "case_C7": C7_user_prompt,
    "case_C8": C8_user_prompt,
    "case_C9": C9_user_prompt,
    "case_C10": C10_user_prompt,
    "case_D1": D1_user_prompt,
    "case_D2": D2_user_prompt,
    "case_D3": D3_user_prompt,
    "case_D4": D4_user_prompt,
    "case_D5": D5_user_prompt,
    "case_D6": D6_user_prompt,
    "case_D7": D7_user_prompt,
    "case_D8": D8_user_prompt,
    "case_D9": D9_user_prompt,
    "case_D10": D10_user_prompt,
}
