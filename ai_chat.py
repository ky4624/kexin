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
import re
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
from main import app, col_file_meta, client, fs

# 导入文件操作相关函数
from file_operations import save_everything, list_files, download_file

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
            'store': {},
            'get': lambda self, key: self.store.get(key),
            'set': lambda self, key, value: self.store.__setitem__(key, value),
            'delete': lambda self, key: self.store.__delitem__(key),
            'keys': lambda self: self.store.keys(),
            'values': lambda self: self.store.values(),
            'items': lambda self: self.store.items(),
            'clear': lambda self: self.store.clear()
        })()
        
        # 对话历史
        self.chat_history = []
        
        # 实体提取器
        self.entity_extractor = type('EntityExtractor', (), {
            'extract_entities': lambda self, text: []
        })()
        
        # 实体记忆的键
        self.memory_key = "entities"
        
        # 输入键
        self.input_key = "input"
        
        # 输出键
        self.output_key = "output"
    
    def add_memory(self, input_text, output_text):
        """添加记忆到实体存储"""
        # 将输入输出对保存到对话历史
        self.chat_history.append(f"{self.human_prefix}: {input_text}")
        self.chat_history.append(f"{self.ai_prefix}: {output_text}")
        
        # 模拟实体提取
        entities = self._extract_entities_from_text(input_text + " " + output_text)
        
        # 更新实体存储
        for entity, description in entities.items():
            self.entity_store.set(entity, description)
        
        # 限制实体存储大小
        if len(self.entity_store.store) > self.entity_cache_limit:
            # 移除最早添加的实体
            oldest_entity = next(iter(self.entity_store.keys()))
            self.entity_store.delete(oldest_entity)
    
    def _extract_entities_from_text(self, text):
        """从文本中提取实体（简单实现）"""
        # 这里可以实现更复杂的实体提取逻辑
        entities = {}
        
        # 示例：提取包含"喜欢"、"讨厌"等关键词的实体
        if "喜欢" in text:
            # 简单的模式匹配，实际应用中可以使用更复杂的NLP技术
            parts = text.split("喜欢")
            if len(parts) > 1:
                entities["喜欢的事物"] = parts[1].strip()
        
        if "讨厌" in text:
            parts = text.split("讨厌")
            if len(parts) > 1:
                entities["讨厌的事物"] = parts[1].strip()
        
        return entities
    
    def save_context(self, input_values, output_values):
        """保存上下文到记忆"""
        input_text = input_values.get(self.input_key, "")
        output_text = output_values.get(self.output_key, "")
        
        if input_text and output_text:
            self.add_memory(input_text, output_text)
    
    def load_memory_variables(self, inputs):
        """加载记忆变量"""
        return {self.memory_key: self.entity_store.store}
    
    def clear(self):
        """清除所有记忆"""
        self.entity_store.clear()
        self.chat_history.clear()

# -------------------------- 初始化记忆系统 --------------------------
# 创建实体记忆实例
entity_memory = CustomEntityMemory(
    entity_cache_limit=100,
    human_prefix="用户",
    ai_prefix="AI助理"
)

# -------------------------- 提示词模板定义 --------------------------
# 豆包风格的记忆提示词模板
MEMORY_PROMPT = PromptTemplate(
    input_variables=["entities", "chat_history", "input"],
    template="""你是AI纳界助理，一个具有智能记忆的助手。

## 用户关键记忆：
{entities}

## 对话历史：
{chat_history}

## 当前用户输入：
{input}

请用简洁、友好的语言回复用户，保持自然的对话风格。"""
)

# -------------------------- 大模型调用 --------------------------
# 定义一个简单的Runnable实现，用于调用DashScope API
class SimpleRunnable(Runnable):
    """简单的Runnable实现，直接调用DashScope API"""
    
    def __init__(self, entity_memory, prompt, llm):
        self.entity_memory = entity_memory
        self.prompt = prompt
        self.llm = llm
    
    def invoke(self, input_data, config=None):
        raise NotImplementedError("请使用异步方法 ainvoke")
    
    async def ainvoke(self, input_data, config=None, **kwargs):
        # 加载记忆变量
        memory_vars = self.entity_memory.load_memory_variables(input_data)
        
        # 获取对话历史
        chat_history = input_data.get("chat_history", [])
        
        # 格式化对话历史
        chat_history_str = "\n".join(chat_history)
        
        # 格式化提示词
        prompt_text = self.prompt.format(
            input=input_data.get("input", ""),
            entities=json.dumps(memory_vars.get("entities", {}), ensure_ascii=False),
            chat_history=chat_history_str
        )
        
        # 调用大模型
        response = Generation.call(
            model="qwen-plus",
            prompt=prompt_text,
            result_format="message"
        )
        
        # 获取回复
        output = response.output.choices[0].message.content
        
        # 保存上下文到记忆
        self.entity_memory.save_context(
            input_data,
            {"output": output}
        )
        
        return {"output": output}

# 初始化大模型
llm = None  # 这里不需要实际的LLM实例，因为我们在SimpleRunnable中直接调用API

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

# -------------------------- 辅助函数 --------------------------
async def save_chat_history(user_id, message, ai_reply, create_time):
    """保存聊天记录到数据库"""
    try:
        chat_record = {
            "user_id": user_id,
            "user_msg": message,
            "ai_reply": ai_reply,
            "create_time": create_time
        }
        col_chat_history.insert_one(chat_record)
        print(f"聊天记录保存成功: {chat_record}")
    except Exception as e:
        print(f"保存聊天记录失败: {e}")
        traceback.print_exc()

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
        
        # 指令解析：识别文件操作请求
        lower_message = message.lower()
        
        # 1. 保存文件/内容指令
        if any(keyword in lower_message for keyword in ["保存", "上传", "存储"]):
            if "文件" in lower_message:
                ai_reply = "请将要传的文件拖拽到上传指定区域"
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=200, content={
                    "code": 200, 
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
            else:
                # 保存文本内容
                try:
                    # 调用save_everything函数保存文本
                    result = await save_everything(content=message, files=[])
                    ai_reply = f"内容已保存成功！{result['msg']}"
                    await save_chat_history(user_id, message, ai_reply, create_time)
                    return JSONResponse(status_code=200, content={
                        "code": 200,
                        "user_msg": message,
                        "ai_reply": ai_reply,
                        "your_key_memory": entity_memory.entity_store.store,
                        "create_time": str(create_time)
                    })
                except Exception as e:
                    print(f"保存内容失败: {e}")
                    ai_reply = f"保存内容失败: {str(e)}"
                    await save_chat_history(user_id, message, ai_reply, create_time)
                    return JSONResponse(status_code=500, content={
                        "code": 500, 
                        "user_msg": message,
                        "ai_reply": ai_reply,
                        "your_key_memory": entity_memory.entity_store.store,
                        "create_time": str(create_time)
                    })
        
        # 2. 列出文件指令（支持按类型过滤）
        elif any(keyword in lower_message for keyword in ["列出文件", "文件列表", "查看文件"]):
            try:
                # 检查是否需要按类型过滤
                file_type = None
                if "pdf" in lower_message:
                    file_type = "pdf"
                elif "excel" in lower_message or "xlsx" in lower_message or "xls" in lower_message:
                    file_type = "xlsx"
                elif "word" in lower_message or "docx" in lower_message or "doc" in lower_message:
                    file_type = "docx"
                elif "txt" in lower_message:
                    file_type = "txt"
                elif "image" in lower_message or "jpg" in lower_message or "png" in lower_message:
                    file_type = "jpg"
                
                # 调用list_files函数获取文件列表
                result = await list_files(user_id=user_id, page=1, page_size=10, file_type=file_type)
                files = result["data"]["files"]
                
                if files:
                    file_list = "\n".join([f"- {file['filename']} (ID: {file['file_id']})" for file in files])
                    type_desc = f" {file_type.upper()}" if file_type else ""
                    ai_reply = f"您上传的{type_desc}文件列表：\n{file_list}\n\n如果需要下载特定文件，请使用'下载文件 [文件ID]'指令。"
                else:
                    type_desc = f" {file_type.upper()}" if file_type else ""
                    ai_reply = f"当前没有保存的{type_desc}文件。"
                
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=200, content={
                    "code": 200,
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
            except Exception as e:
                print(f"获取文件列表失败: {e}")
                ai_reply = f"获取文件列表失败: {str(e)}"
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=500, content={
                    "code": 500, 
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
        
        # 3. 下载文件指令
        elif "下载文件" in lower_message:
            # 提取文件ID
            match = re.search(r'下载文件\s+([a-zA-Z0-9]+)', lower_message)
            if match:
                file_id = match.group(1)
                ai_reply = f"请使用/download_file/{file_id}接口下载文件。"
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=200, content={
                    "code": 200,
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
            else:
                ai_reply = "请提供正确的文件ID，格式：'下载文件 [文件ID]'"
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=200, content={
                    "code": 200,
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
        
        # 4. 查找/搜索文件指令
        elif any(keyword in lower_message for keyword in ["查找", "搜索", "找出"]):
            try:
                # 提取文件类型
                file_type = None
                if "pdf" in lower_message:
                    file_type = "pdf"
                elif "excel" in lower_message or "xlsx" in lower_message or "xls" in lower_message:
                    file_type = "xlsx"
                elif "word" in lower_message or "docx" in lower_message or "doc" in lower_message:
                    file_type = "docx"
                elif "txt" in lower_message:
                    file_type = "txt"
                elif "image" in lower_message or "jpg" in lower_message or "png" in lower_message:
                    file_type = "jpg"
                
                if not file_type:
                    ai_reply = "请明确您要查找的文件类型，例如：查找我上传的所有PDF文件"
                    await save_chat_history(user_id, message, ai_reply, create_time)
                    return JSONResponse(status_code=200, content={
                        "code": 200,
                        "user_msg": message,
                        "ai_reply": ai_reply,
                        "your_key_memory": entity_memory.entity_store.store,
                        "create_time": str(create_time)
                    })
                
                # 调用list_files函数获取文件列表
                result = await list_files(user_id=user_id, page=1, page_size=10, file_type=file_type)
                files = result["data"]["files"]
                
                if files:
                    file_list = "\n".join([f"- {file['filename']} (ID: {file['file_id']})" for file in files])
                    ai_reply = f"找到您上传的{file_type.upper()}文件：\n{file_list}\n\n如果需要下载特定文件，请使用'下载文件 [文件ID]'指令。"
                else:
                    ai_reply = f"没有找到您上传的{file_type.upper()}文件。"
                
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=200, content={
                    "code": 200,
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
            except Exception as e:
                print(f"查找文件失败: {e}")
                ai_reply = f"查找文件失败: {str(e)}"
                await save_chat_history(user_id, message, ai_reply, create_time)
                return JSONResponse(status_code=500, content={
                    "code": 500, 
                    "user_msg": message,
                    "ai_reply": ai_reply,
                    "your_key_memory": entity_memory.entity_store.store,
                    "create_time": str(create_time)
                })
        
        # 5. 其他文件操作指令
        elif any(keyword in lower_message for keyword in ["文件", "保存", "下载", "上传"]):
            ai_reply = "我支持以下文件操作：\n1. 保存文本内容：直接输入要保存的内容即可\n2. 上传文件：请使用/save_all接口\n3. 列出文件：输入'列出文件'或'查看文件'\n4. 按类型查找文件：例如'查找我上传的所有PDF文件'\n5. 下载文件：输入'下载文件 [文件ID]'"
            await save_chat_history(user_id, message, ai_reply, create_time)
            return JSONResponse(status_code=200, content={
                "code": 200,
                "user_msg": message,
                "ai_reply": ai_reply,
                "your_key_memory": entity_memory.entity_store.store,
                "create_time": str(create_time)
            })
        
        # 核心：LangChain对话+记忆更新，豆包风格回复
        print("准备调用conversation_chain.ainvoke")
        result = await conversation_chain.ainvoke(
            {"input": message}
        )
        ai_reply = result["output"]
        
        print(f"AI回复: {ai_reply}")
        
        # 保存聊天记录
        await save_chat_history(user_id, message, ai_reply, create_time)
        
        return JSONResponse(status_code=200, content={
            "code": 200,
            "user_msg": message,
            "ai_reply": ai_reply,
            "your_key_memory": entity_memory.entity_store.store,
            "create_time": str(create_time)
        })
    except json.JSONDecodeError as e:
        print(f"请求格式错误: {e}")
        ai_reply = "请求格式错误，请使用JSON格式"
        await save_chat_history(user_id, message, ai_reply, create_time)
        return JSONResponse(status_code=400, content={
            "code": 400, 
            "user_msg": message,
            "ai_reply": ai_reply,
            "your_key_memory": entity_memory.entity_store.store,
            "create_time": str(create_time)
        })
    except Exception as e:
        print(f"聊天失败: {e}")
        traceback.print_exc()
        # 提供更详细的错误信息，但注意不要泄露敏感信息
        error_msg = f"聊天失败: {str(e).split('(')[0]}" if "(" in str(e) else str(e)
        await save_chat_history(user_id, message, error_msg, create_time)
        return JSONResponse(status_code=500, content={
            "code": 500, 
            "user_msg": message,
            "ai_reply": error_msg,
            "your_key_memory": entity_memory.entity_store.store,
            "create_time": str(create_time)
        })

@app.post("/load_memory", summary="重启服务后加载长期记忆：保证记忆永不丢失")
async def load_user_memory():
    user_id = "user_001"
    try:
        # 加载用户的长期记忆
        memory_data = col_user_memory.find_one({"user_id": user_id})
        if memory_data:
            # 恢复实体存储
            entity_memory.entity_store.store = memory_data.get("entities", {})
            return JSONResponse(status_code=200, content={"code": 200, "msg": "长期记忆加载成功"})
        else:
            return JSONResponse(status_code=200, content={"code": 200, "msg": "没有找到长期记忆"})
    except Exception as e:
        print(f"加载长期记忆失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"加载长期记忆失败: {str(e).split('(')[0]}"})

@app.post("/save_memory", summary="手动保存当前实体记忆到数据库")
async def save_memory():
    user_id = "user_001"
    try:
        # 保存当前实体记忆到数据库
        col_user_memory.update_one(
            {"user_id": user_id},
            {"$set": {"entities": entity_memory.entity_store.store, "update_time": datetime.datetime.now()}},
            upsert=True
        )
        return JSONResponse(status_code=200, content={"code": 200, "msg": "实体记忆保存成功"})
    except Exception as e:
        print(f"保存实体记忆失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"保存实体记忆失败: {str(e).split('(')[0]}"})

@app.post("/search", summary="搜索文件元信息+聊天记录+记忆：知识检索功能")
async def search_knowledge(request: Request):
    user_id = "user_001"
    try:
        # 手动解析请求体
        request_body = await request.json()
        query = request_body.get("query", "").strip()
        
        if not query:
            return JSONResponse(status_code=400, content={"code": 400, "msg": "搜索关键词不能为空"})
        
        print(f"收到搜索请求: {query}")
        
        # 搜索文件元信息
        files = list(col_file_meta.find(
            {"user_id": user_id, "is_valid": True},
            {"_id": 0, "file_id": 1, "filename": 1, "content": 1, "create_time": 1}
        ))
        
        # 简单的文本匹配
        search_results = []
        for file in files:
            if query in file.get("filename", "") or query in file.get("content", ""):
                search_results.append(file)
        
        return JSONResponse(status_code=200, content={
            "code": 200,
            "query": query,
            "results": search_results,
            "count": len(search_results)
        })
    except json.JSONDecodeError as e:
        print(f"请求格式错误: {e}")
        return JSONResponse(status_code=400, content={"code": 400, "msg": "请求格式错误，请使用JSON格式"})
    except Exception as e:
        print(f"搜索失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"搜索失败: {str(e).split('(')[0]}"})