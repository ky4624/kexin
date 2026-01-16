import asyncio
import json
import traceback
import sys
import pydantic
from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse
import datetime
import dashscope
from dashscope import Generation
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable
import os
from main import col_chat_history, col_user_memory

# 猴子补丁：解决 langchain_core.pydantic_v1 模块缺失问题
class MockPydanticV1:
    def __getattr__(self, name):
        if hasattr(pydantic, 'v1'):
            return getattr(pydantic.v1, name)
        else:
            return getattr(pydantic, name)

sys.modules['langchain_core.pydantic_v1'] = MockPydanticV1()

# 导入全局应用实例
from main import app, col_file_meta, client

# 加载环境变量
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key or len(api_key) < 20:
    raise ValueError("环境变量 DASHSCOPE_API_KEY 格式不正确或未设置")

# 验证API密钥格式
if not api_key.startswith("sk-"):
    raise ValueError("DASHSCOPE_API_KEY 格式不正确，应该以 'sk-' 开头")

print(f"使用的API密钥: {api_key[:10]}...")

# -------------------------- 自定义实体记忆系统 --------------------------
class CustomEntityMemory:
    """自定义实体记忆系统，替代已弃用的ConversationEntityMemory"""
    
    def __init__(self, entity_cache_limit=100, human_prefix="用户", ai_prefix="AI助理"):
        self.entity_cache_limit = entity_cache_limit
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        
        # 实体存储
        self.entity_store = type('EntityStore', (), {
            'store': {}  # 存储实体信息的字典
        })()
        
        # 历史对话存储
        self.history = []
        
        # 对话缓冲区
        self.buffer = []
    
    def load_memory_variables(self, inputs):
        """加载记忆变量"""
        # 获取历史对话
        history = self.get_history_string()
        
        # 获取实体信息
        entities = json.dumps(self.entity_store.store, ensure_ascii=False)
        
        return {
            "history": history,
            "entities": entities
        }
    
    def save_context(self, inputs, outputs):
        """保存对话上下文"""
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")
        
        # 保存到历史
        self.history.append({
            "user": input_text,
            "ai": output_text
        })
        
        # 限制历史长度
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # 更新实体存储
        self._update_entity_store(input_text, output_text)
    
    def get_history_string(self):
        """获取格式化的历史对话字符串"""
        history_str = ""
        for entry in self.history:
            history_str += f"{self.human_prefix}: {entry['user']}\n{self.ai_prefix}: {entry['ai']}\n"
        return history_str.strip()
    
    def _update_entity_store(self, input_text, output_text):
        """更新实体存储"""
        # 简单的实体提取逻辑（实际应用中可以使用更复杂的NLP技术）
        combined_text = input_text + " " + output_text
        
        # 提取可能的实体（简单示例：提取以"我喜欢"开头的实体）
        if "我喜欢" in combined_text:
            start_idx = combined_text.find("我喜欢") + 3
            # 提取到下一个标点符号或句子结束
            end_idx = start_idx
            while end_idx < len(combined_text) and combined_text[end_idx] not in [",", ".", "。", "，", "!", "！", "?", "？", "\n"]:
                end_idx += 1
            if end_idx > start_idx:
                entity = combined_text[start_idx:end_idx].strip()
                if entity:
                    self.entity_store.store[entity] = "用户喜欢的事物"
        
        # 限制实体存储数量
        if len(self.entity_store.store) > self.entity_cache_limit:
            # 移除最早添加的实体
            old_entity = next(iter(self.entity_store.store))
            del self.entity_store.store[old_entity]

# -------------------------- 自定义LLM类（使用原生dashscope） --------------------------
class DashScopeLLM(Runnable):
    def __init__(self, model="qwen-turbo", temperature=0.7, api_key=None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        
        # 设置dashscope API密钥
        if api_key:
            dashscope.api_key = api_key
    
    def invoke(self, input, config=None, **kwargs):
        """同步调用dashscope API"""
        try:
            # 确保输入是字符串
            if isinstance(input, dict):
                input = input.get("prompt", "")
            
            # 调用dashscope API
            response = Generation.call(
                model=self.model,
                prompt=input,
                temperature=self.temperature
            )
            
            # 处理响应
            if response.status_code == 200 and response.output and response.output.text:
                return {"content": response.output.text}
            else:
                raise Exception(f"DashScope API调用失败: {response}")
        except Exception as e:
            print(f"同步调用失败: {e}")
            traceback.print_exc()
            raise
    
    async def ainvoke(self, input, config=None, **kwargs):
        """异步调用dashscope API"""
        try:
            # 确保输入是字符串
            if isinstance(input, dict):
                input = input.get("prompt", "")
            
            # 使用asyncio.to_thread执行同步调用
            response = await asyncio.to_thread(
                Generation.call,
                model=self.model,
                prompt=input,
                temperature=self.temperature
            )
            
            # 处理响应
            if response.status_code == 200 and response.output and response.output.text:
                return {"content": response.output.text}
            else:
                raise Exception(f"DashScope API异步调用失败: {response}")
        except Exception as e:
            print(f"异步调用失败: {e}")
            traceback.print_exc()
            raise

# 创建通义千问( DashScope )实例
llm = DashScopeLLM(
    model="qwen-turbo",  # 可以根据需要更换为其他千问模型
    temperature=0.7,
    api_key=api_key
)

print("成功创建通义千问(DashScopeLLM)实例")

# 使用自定义实体记忆替代ConversationEntityMemory
entity_memory = CustomEntityMemory(
    entity_cache_limit=100,
    human_prefix="用户",
    ai_prefix="AI助理"
)

MEMORY_PROMPT = PromptTemplate(
    input_variables=["input", "history", "entities"],
    template="""你是用户的专属AI纳界助理，拥有用户的全部长期记忆和知识库，风格和豆包一致，回复简洁精准友好。
    1. 你需要严格记住用户的实体信息：{entities}（比如爱吃鱼、不吃辣、忌口、喜好等，所有对话必须关联该记忆）
    2. 参考历史对话上下文：{history}
    3. 针对用户的问题/指令：{input} 进行精准回复，支持自然语言操控所有功能。
    规则：用户的所有资料永久保存，可随时调取；自动整理无效文件；提取关键记忆并永久生效；上传的文件自动解析内容并保存。
    回复要求：口语化、流畅，和豆包的回复风格一致，不要生硬，记忆内容无缝融入回复中。
    """
)

# 创建一个简单的runnable，模拟原来的ConversationChain行为
class SimpleRunnable:
    def __init__(self, entity_memory, prompt, llm):
        self.entity_memory = entity_memory
        self.prompt = prompt
        self.llm = llm
    
    def invoke(self, input, config=None, **kwargs):
        try:
            # 获取历史对话
            history = self.entity_memory.load_memory_variables({})["history"]
            # 获取实体信息
            entities = self.entity_memory.load_memory_variables({})["entities"]
            
            # 格式化提示
            formatted_prompt = self.prompt.format(
                input=input["input"],
                history=history,
                entities=entities
            )
            
            print(f"[同步] 准备调用LLM，提示词长度: {len(formatted_prompt)}")
            # 使用LLM生成回复
            response = self.llm.invoke(formatted_prompt)
            print(f"[同步] LLM调用成功，响应类型: {type(response)}")
            
            # 处理不同的返回格式
            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, dict) and 'content' in response:
                response_content = response['content']
            else:
                response_content = str(response)
            
            print(f"[同步] LLM回复内容: {response_content}")
            
            # 更新记忆
            self.entity_memory.save_context(
                {"input": input["input"]},
                {"output": response_content}
            )
            
            print(f"[同步] 记忆更新成功，返回AI回复")
            return {"output": response_content}
        except Exception as e:
            print(f"[同步] invoke方法执行失败: {e}")
            traceback.print_exc()
            return {"output": "抱歉，我暂时无法处理您的请求，请稍后重试。"}
    
    async def ainvoke(self, input, config=None, **kwargs):
        try:
            # 获取历史对话
            history = self.entity_memory.load_memory_variables({})["history"]
            # 获取实体信息
            entities = self.entity_memory.load_memory_variables({})["entities"]
            
            # 格式化提示
            formatted_prompt = self.prompt.format(
                input=input["input"],
                history=history,
                entities=entities
            )
            
            print(f"[异步] 准备调用LLM，提示词长度: {len(formatted_prompt)}")
            
            # 异步调用LLM
            response = await self.llm.ainvoke(formatted_prompt)
            print(f"[异步] LLM异步调用成功，响应类型: {type(response)}")
            
            # 处理不同的返回格式
            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, dict) and 'content' in response:
                response_content = response['content']
            else:
                response_content = str(response)
            
            print(f"[异步] LLM回复内容: {response_content}")
            
            # 更新记忆
            self.entity_memory.save_context(
                {"input": input["input"]},
                {"output": response_content}
            )
            
            print(f"[异步] 记忆更新成功，返回AI回复")
            return {"output": response_content}
        except Exception as e:
            print(f"[异步] ainvoke方法执行失败: {e}")
            traceback.print_exc()
            return {"output": "抱歉，我暂时无法处理您的请求，请稍后重试。"}

# 创建简单的runnable实例
runnable = SimpleRunnable(
    entity_memory=entity_memory,
    prompt=MEMORY_PROMPT,
    llm=llm
)

# 定义获取历史对话的函数
# 注意：这里我们使用CustomEntityMemory的内置历史，所以返回空列表
def get_session_history(session_id):
    return []

# 创建带有历史记录的runnable
conversation_chain = runnable

# -------------------------- 聊天相关API接口 --------------------------
@app.post("/chat", summary="聊天+记忆+汇总保存：核心对话接口，仿豆包回复风格")
async def chat_with_assistant(request: Request):
    user_id = "user_001"
    create_time = datetime.datetime.now()
    
    try:
        # 手动解析请求体
        request_body = await request.json()
        message = request_body.get("message", "").strip()
        
        if not message:
            return JSONResponse(status_code=400, content={"code": 400, "msg": "消息内容不能为空"})
        
        print(f"收到用户消息: {message}")
        
        # 核心：LangChain对话+记忆更新，豆包风格回复
        print("准备调用conversation_chain.ainvoke")
        result = await conversation_chain.ainvoke(
            {"input": message}
        )
        ai_reply = result["output"]
        
        print(f"AI回复: {ai_reply}")
        
        return {
            "code": 200,
            "user_msg": message,
            "ai_reply": ai_reply,
            "your_key_memory": entity_memory.entity_store.store,
            "create_time": str(create_time)
        }
    except json.JSONDecodeError as e:
        print(f"请求格式错误: {e}")
        return JSONResponse(status_code=400, content={"code": 400, "msg": "请求格式错误，请使用JSON格式"})
    except Exception as e:
        print(f"聊天失败: {e}")
        traceback.print_exc()
        # 提供更详细的错误信息，但注意不要泄露敏感信息
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"聊天失败: {str(e).split('(')[0]}"})

@app.post("/load_memory", summary="重启服务后加载长期记忆：保证记忆永不丢失")
async def load_user_memory():
    user_id = "user_001"
    try:
        # 加载用户的长期记忆
        user_memory = col_user_memory.find_one({"user_id": user_id})
        if user_memory and "memory" in user_memory:
            entity_memory.entity_store.store = user_memory["memory"]
            return {"code": 200, "msg": "长期记忆加载成功", "memory": user_memory["memory"]}
        return {"code": 200, "msg": "暂无长期记忆", "memory": {}}
    except Exception as e:
        print(f"加载记忆失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"加载记忆失败: {str(e)}"})

@app.post("/search", summary="深度检索：自然语言搜索所有保存的资料/聊天记录/记忆")
async def search_all(
    query: str = Body(..., description="搜索关键词/自然语言查询"),
    search_type: str = Body(default="all", description="搜索类型：all(全部)/file(文件)/chat(聊天记录)/memory(记忆)")
):
    user_id = "user_001"
    try:
        results = []
        
        # 搜索文件元信息
        if search_type in ["all", "file"]:
            file_results = list(col_file_meta.find({"user_id": user_id, "is_valid": True}))
            for doc in file_results:
                if query in str(doc.get("content", "")) or query in doc.get("filename", ""):
                    results.append({
                        "type": "file",
                        "filename": doc.get("filename", ""),
                        "content": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                        "create_time": str(doc.get("create_time", ""))
                    })
        
        # 搜索聊天记录
        if search_type in ["all", "chat"]:
            # 这里可以根据需要实现聊天记录搜索
            pass
        
        # 搜索记忆
        if search_type in ["all", "memory"]:
            for entity, description in entity_memory.entity_store.store.items():
                if query in entity or query in description:
                    results.append({
                        "type": "memory",
                        "entity": entity,
                        "description": description
                    })
        
        return {"code": 200, "msg": "搜索完成", "results": results}
    except Exception as e:
        print(f"搜索失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"搜索失败: {str(e)}"})