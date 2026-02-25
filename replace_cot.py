import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet("train.parquet")

# 定义需要替换的原始文本和目标文本
old_text = ("Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: ")

new_text = ("""Answer the given question. You must conduct reasoning inside **<think>** and **</think>** tag in the begin of your response. The reasoning is to plan how to answer the question correctly. After reasoning, if you find you lack some knowledge, you can call a search engine by writing query sentence to call search engine by output "<search> YOUR QUERY HERE </search>" and it will return the top searched results between **<information>** and **</information>** Tags. If you find no further external knowledge needed, you can directly provide the answer inside **<answer>** and **</answer>** tags, answer must be short and clear. For example, <answer>Beijing</answer>. Note, after think, you should either search or answer, and stop output after that.

# Output example IF you need search information
<think> YOUR REASONING HERE </think>
<search> YOUR QUERY HERE </search>

# Output example IF no further external knowledge needed
<think> YOUR REASONING HERE </think>
<answer> YOUR QUERY HERE </answer>

# Question
""")


# 定义替换函数
def replace_prompt_content(prompt_list):
    for item in prompt_list:
        if 'content' in item and isinstance(item['content'], str):
            item['content'] = item['content'].replace(old_text, new_text)
    return prompt_list

# 执行替换
df['prompt'] = df['prompt'].apply(replace_prompt_content)

# 可选：保存修改后的文件
df.to_parquet("train.parquet", index=False)

print(df.shape)
# 显示第一行，验证替换结果
print(df.iloc[0]['prompt'][0]['content'])

