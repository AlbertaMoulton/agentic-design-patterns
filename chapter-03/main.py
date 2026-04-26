import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

# 初始化语言模型（推荐使用 ChatOpenAI）
try:
   llm: Optional[ChatOpenAI] = ChatOpenAI(
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
except Exception as e:
   print(f"初始化语言模型出错：{e}")
   llm = None

# --- 定义独立链 ---
# 三个链分别执行不同任务，可并行运行

summarize_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请简明扼要地总结以下主题："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

questions_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请针对以下主题生成三个有趣的问题："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

terms_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "请从以下主题中提取 5-10 个关键词，用逗号分隔："),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

# --- 构建并行 + 汇总链 ---

# 1. 定义并行任务块，结果与原始 topic 一起传递到下一步
map_chain = RunnableParallel(
   {
       "summary": summarize_chain,
       "questions": questions_chain,
       "key_terms": terms_chain,
       "topic": RunnablePassthrough(),  # 传递原始 topic
   }
)

# 2. 定义最终汇总 prompt，整合并行结果
synthesis_prompt = ChatPromptTemplate.from_messages([
   ("system", """根据以下信息：
    摘要：{summary}
    相关问题：{questions}
    关键词：{key_terms}
    请综合生成完整答案。"""),
   ("user", "原始主题：{topic}")
])

# 3. 构建完整链，将并行结果直接传递给汇总 prompt，再由 LLM 和输出解析器处理
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- 运行链 ---
async def run_parallel_example(topic: str) -> None:
   """
   异步调用并行处理链，输出综合结果。

   Args:
       topic: 传递给 LangChain 的主题输入
   """
   if not llm:
       print("LLM 未初始化，无法运行示例。")
       return

   print(f"\n--- 并行 LangChain 示例，主题：'{topic}' ---")
   try:
       # `ainvoke` 的输入是单个 topic 字符串，
       # 会传递给 map_chain 中的每个 runnable
       response = await full_parallel_chain.ainvoke(topic)
       print("\n--- 最终响应 ---")
       print(response)
   except Exception as e:
       print(f"\n链执行出错：{e}")

if __name__ == "__main__":
   test_topic = "太空探索的历史"
   # Python 3.7+ 推荐用 asyncio.run 执行异步函数
   asyncio.run(run_parallel_example(test_topic))
