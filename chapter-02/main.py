# Copyright (c) 2025 Marco Fago
# https://www.linkedin.com/in/marco-fago/
#
# 本代码采用 MIT 许可证，详见仓库 LICENSE 文件。

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# --- 配置 ---
# 从项目目录的 .env 加载环境变量
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# 确保环境变量已设置 API 密钥（如 GOOGLE_API_KEY）
try:
   if not google_api_key or google_api_key == "your_google_api_key_here":
       raise ValueError("未检测到 GOOGLE_API_KEY，请在 .env 文件中配置后重试。")
   llm = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash",
       temperature=0,
       google_api_key=google_api_key,
   )
   print(f"语言模型初始化成功：{llm.model}")
except Exception as e:
   print(f"语言模型初始化失败：{e}")
   llm = None

# --- 定义模拟子智能体处理器（等同于 ADK sub_agents） ---

def booking_handler(request: str) -> str:
   """模拟预订智能体处理请求。"""
   print("\n--- 委托给预订处理器 ---")
   return f"预订处理器已处理请求：'{request}'。结果：模拟预订动作。"

def info_handler(request: str) -> str:
   """模拟信息智能体处理请求。"""
   print("\n--- 委托给信息处理器 ---")
   return f"信息处理器已处理请求：'{request}'。结果：模拟信息检索。"

def unclear_handler(request: str) -> str:
   """处理无法委托的请求。"""
   print("\n--- 处理不明确请求 ---")
   return f"协调者无法委托请求：'{request}'。请补充说明。"

# --- 定义协调者路由链（等同于 ADK 协调者指令） ---
coordinator_router_prompt = ChatPromptTemplate.from_messages([
   ("system", """分析用户请求，判断应由哪个专属处理器处理。
    - 若请求涉及预订机票或酒店，输出 'booker'。
    - 其他一般信息问题，输出 'info'。
    - 若请求不明确或不属于上述类别，输出 'unclear'。
    只输出一个词：'booker'、'info' 或 'unclear'。"""),
   ("user", "{request}")
])

if llm:
   coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()
else:
   coordinator_router_chain = None

# --- 定义委托逻辑（等同于 ADK 的 Auto-Flow） ---
branches = {
   "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
   "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
   "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
}

delegation_branch = RunnableBranch(
   (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
   (lambda x: x['decision'].strip() == 'info', branches["info"]),
   branches["unclear"] # 默认分支
)

if coordinator_router_chain:
   coordinator_agent = {
      "decision": coordinator_router_chain,
      "request": RunnablePassthrough()
   } | delegation_branch | (lambda x: x['output'])
else:
   coordinator_agent = None

# --- 示例用法 ---
def main():
   if not llm or not coordinator_agent:
       print("\n因 LLM 初始化失败，跳过执行。")
       return

   print("--- 预订请求示例 ---")
   request_a = "帮我预订飞往伦敦的机票。"
   result_a = coordinator_agent.invoke({"request": request_a})
   print(f"最终结果 A: {result_a}")

   print("\n--- 信息请求示例 ---")
   request_b = "意大利的首都是哪里？"
   result_b = coordinator_agent.invoke({"request": request_b})
   print(f"最终结果 B: {result_b}")

   print("\n--- 不明确请求示例 ---")
   request_c = "讲讲量子物理。"
   result_c = coordinator_agent.invoke({"request": request_c})
   print(f"最终结果 C: {result_c}")

if __name__ == "__main__":
   main()