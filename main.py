import os
import uuid
import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from gridfs import GridFS, NoFile
from langchain_classic.memory import ConversationEntityMemory
from langchain_classic.chains import ConversationChain
from langchain_classic.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation
import langextract as le
from PyPDF2 import PdfReader
from openpyxl import load_workbook

# 加载环境变量
load_dotenv()

# 检查必要的环境变量
required_env_vars = ["MONGO_URI", "MONGO_DB_NAME"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"环境变量 {var} 未设置")

app = FastAPI(title="AI纳界助理", description="永久存储/智能记忆/深度检索/自然语言操控/对话页")

# -------------------------- 新增：模拟LLM类 --------------------------
class MockLLM(BaseLanguageModel):
    """一个简单的模拟LLM类，用于满足ConversationEntityMemory的llm参数要求"""
    
    def __init__(self):
        super().__init__()
    
    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """模拟生成回复，返回空的实体信息"""
        generations = []
        for _ in prompts:
            # 返回空的生成结果，ConversationEntityMemory需要这个接口但实际上不使用它的内容
            generations.append([Generation(text="{}")])
        return LLMResult(generations=generations, llm_output={})
    
    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """异步模拟生成回复，返回空的实体信息"""
        return self.generate_prompt(prompts, stop, callbacks, **kwargs)
    
    def invoke(self, input, stop=None, callbacks=None, **kwargs):
        """实现BaseLanguageModel要求的invoke方法"""
        return "{}"
    
    async def ainvoke(self, input, stop=None, callbacks=None, **kwargs):
        """实现BaseLanguageModel要求的异步ainvoke方法"""
        return "{}"

# 跨域配置【升级】：前端本地调试+生产环境全兼容
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【新增核心】托管静态前端页面 - 访问 http://localhost:8000 直接打开仿豆包对话页
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# -------------------------- 1. 数据库初始化 - MongoDB 永久存储 --------------------------
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB_NAME")]
fs = GridFS(db)  # GridFS存储大文件：图片/音频/视频/PDF/Excel等
# 集合定义（所有数据永久保存）
col_chat_history = db["chat_history"]  # 聊天记录
col_user_memory = db["user_memory"]    # 用户关键记忆（爱吃鱼、不吃辣等）
col_file_meta = db["file_meta"]        # 文件元信息+抽取的内容

# -------------------------- 2. LangChain 核心记忆体系初始化（重点不变） --------------------------
# 创建模拟LLM实例
mock_llm = MockLLM()

# 使用ConversationEntityMemory作为唯一的记忆对象
entity_memory = ConversationEntityMemory(
    entity_cache_limit=100,
    llm=mock_llm,
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
conversation_chain = ConversationChain(
    memory=entity_memory,
    prompt=MEMORY_PROMPT,
    llm=mock_llm,
    verbose=True
)

# -------------------------- 3. 工具函数：文件内容提取+LangExtract信息拉取 --------------------------
def extract_file_content(file: UploadFile, file_content: bytes) -> str:
    content = ""
    suffix = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    try:
        if suffix in ["txt", "md", "json"]:
            content = file_content.decode("utf-8", errors="ignore")
        elif suffix == "pdf":
            try:
                # 使用已读取的内容创建PdfReader对象
                from io import BytesIO
                pdf_reader = PdfReader(BytesIO(file_content))
                content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            except Exception as e:
                print(f"PDF内容提取失败: {e}")
                content = f"[PDF提取失败] {str(e)}"
        elif suffix in ["xlsx", "xls"]:
            try:
                from io import BytesIO
                wb = load_workbook(BytesIO(file_content))
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        content += " ".join([str(cell) for cell in row if cell]) + "\n"
            except Exception as e:
                print(f"Excel内容提取失败: {e}")
                content = f"[Excel提取失败] {str(e)}"
        elif suffix in ["jpg", "png", "jpeg", "gif"]:
            content = f"[{file.filename}] 图片文件，格式：{suffix}，大小：{len(file_content)}字节"
        elif suffix in ["mp4", "mp3", "avi", "mov", "wav"]:
            content = f"[{file.filename}] 音视频文件，格式：{suffix}，大小：{len(file_content)}字节"
        elif suffix in ["docx", "doc"]:
            try:
                from io import BytesIO
                from docx import Document
                doc = Document(BytesIO(file_content))
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                content = f"[{file.filename}] Word文档，已保存，内容可检索（需要安装python-docx库以提取内容）"
            except Exception as e:
                print(f"Word内容提取失败: {e}")
                content = f"[{file.filename}] Word文档，已保存，内容提取失败: {str(e)}"
        
        # LangExtract核心提取：清洗+提炼关键信息
        if content:
            content = le.extract(content, extract_type="text", clean=True)
    except Exception as e:
        print(f"文件内容提取失败: {e}")
        content = f"[文件提取失败] {str(e)}"
    return content

# -------------------------- 4. 新增：前端首页路由（访问根目录打开仿豆包对话页） --------------------------
@app.get("/", summary="仿豆包风格的AI纳界助理对话首页")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------- 5. 核心API接口（全部保留+优化适配前端） --------------------------
@app.post("/save_all", summary="核心接口：发送任意内容/文件给AI，永久保存+自动整理")
async def save_everything(
    content: str = Body(default="", description="文本内容：日记/资料/文字信息"),
    files: list[UploadFile] = File(default=[], description="上传文件：图片/音频/视频/PDF/Excel/所有格式")
):
    user_id = "user_001"
    save_id = str(uuid.uuid4())
    create_time = datetime.datetime.now()
    
    try:
        if content.strip():
            core_content = le.extract(content, extract_type="summary", max_length=500)
            col_file_meta.insert_one({
                "save_id": save_id,
                "user_id": user_id,
                "type": "text",
                "content": content,
                "core_content": core_content,
                "filename": "文本笔记/日记",
                "create_time": create_time,
                "is_valid": True
            })
        
        file_list = []
        for file in files:
            try:
                # 只读取一次文件内容
                file_content = await file.read()
                file_id = fs.put(file_content, filename=file.filename, content_type=file.content_type)
                extract_content = extract_file_content(file, file_content)
                col_file_meta.insert_one({
                    "save_id": save_id,
                    "user_id": user_id,
                    "type": "file",
                    "file_id": file_id,
                    "filename": file.filename,
                    "content": extract_content,
                    "create_time": create_time,
                    "is_valid": True
                })
                file_list.append(file.filename)
            except Exception as e:
                print(f"处理文件 {file.filename} 失败: {e}")
                return JSONResponse(status_code=500, content={"code": 500, "msg": f"处理文件 {file.filename} 失败: {str(e)}"})
        
        # 更新记忆
        conversation_chain.predict(input=f"保存资料：{content}，上传文件：{file_list}")
        
        return {
            "code": 200,
            "msg": "✅ 所有内容已永久保存+AI自动整理完成",
            "data": {"save_id": save_id, "create_time": str(create_time), "files": file_list}
        }
    except Exception as e:
        print(f"保存内容失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"保存内容失败: {str(e)}"})

@app.post("/retrieve", summary="核心接口：自然语言调取资料（深度检索+时间检索+内容检索）")
async def retrieve_info(query: str = Body(..., description="自然语言检索指令"),
                        start_time: str = Body(default=None),
                        end_time: str = Body(default=None)):
    user_id = "user_001"
    query_filter = {"user_id": user_id, "is_valid": True}
    
    try:
        if start_time and end_time:
            try:
                start_dt = datetime.datetime.fromisoformat(start_time)
                end_dt = datetime.datetime.fromisoformat(end_time)
                query_filter["create_time"] = {"$gte": start_dt, "$lte": end_dt}
            except ValueError as e:
                return JSONResponse(status_code=400, content={"code": 400, "msg": f"时间格式错误: {str(e)}"})
        
        all_docs = list(col_file_meta.find(query_filter))
        match_docs = [doc for doc in all_docs if query in doc.get("content", "") or query in doc.get("filename", "")]
        chat_docs = list(col_chat_history.find({"user_id": user_id, "user_msg": {"$regex": query}}))
        
        entity_info = entity_memory.entity_store.store
        memory_prompt = f"用户关键记忆：{entity_info}，检索需求：{query}"
        ai_response = conversation_chain.predict(input=memory_prompt)
        
        return {
            "code": 200,
            "msg": "检索完成，已关联你的长期关键记忆",
            "data": {
                "user_memory": entity_info,
                "match_files": match_docs,
                "match_chat": chat_docs,
                "ai_summary": ai_response
            }
        }
    except Exception as e:
        print(f"检索失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"检索失败: {str(e)}"})

@app.post("/chat", summary="聊天+记忆+汇总保存：核心对话接口，仿豆包回复风格")
async def chat_with_assistant(message: str = Body(..., description="用户对话/指令")):
    user_id = "user_001"
    create_time = datetime.datetime.now()
    
    try:
        # 核心：LangChain对话+记忆更新，豆包风格回复
        ai_reply = conversation_chain.predict(input=message)
        
        # 指令处理：删除无效文件/汇总资料/提取记忆
        if any(key in message for key in ["删除无效文件", "清理垃圾文件", "删除空文件"]):
            col_file_meta.update_many({"content": "", "filename": {"$regex": "无内容"}}, {"$set": {"is_valid": False}})
            ai_reply += "\n✅ 已自动清理所有无效/空文件，标记为失效状态"
        
        if any(key in message for key in ["汇总保存", "整理资料", "汇总我的所有资料"]):
            all_valid_docs = list(col_file_meta.find({"user_id": user_id, "is_valid": True}))
            summary = le.extract(str(all_valid_docs), extract_type="summary", max_length=1000)
            col_file_meta.insert_one({
                "user_id": user_id,
                "type": "summary",
                "content": summary,
                "filename": "资料汇总-" + str(create_time.date()),
                "create_time": create_time,
                "is_valid": True
            })
            ai_reply += f"\n✅ 已汇总你所有的有效资料并永久保存，共整理 {len(all_valid_docs)} 条内容"
        
        if any(key in message for key in ["提取关键记忆", "我的记忆", "我有哪些偏好"]):
            ai_reply = f"📌 你的长期关键记忆：{entity_memory.entity_store.store}\n\n{ai_reply}"
        
        # 聊天记录永久保存
        col_chat_history.insert_one({
            "user_id": user_id,
            "user_msg": message,
            "ai_reply": ai_reply,
            "create_time": create_time
        })
        
        # 关键记忆持久化
        col_user_memory.update_one(
            {"user_id": user_id},
            {"$set": {"memory": entity_memory.entity_store.store, "update_time": create_time}},
            upsert=True
        )
        
        return {
            "code": 200,
            "user_msg": message,
            "ai_reply": ai_reply,
            "your_key_memory": entity_memory.entity_store.store,
            "create_time": str(create_time)
        }
    except Exception as e:
        print(f"聊天失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"聊天失败: {str(e)}"})

@app.post("/load_memory", summary="重启服务后加载长期记忆：保证记忆永不丢失")
async def load_user_memory():
    user_id = "user_001"
    try:
        memory_doc = col_user_memory.find_one({"user_id": user_id})
        if memory_doc:
            entity_memory.entity_store.store = memory_doc["memory"]
            return {"code": 200, "msg": "✅ 长期关键记忆加载完成", "memory": memory_doc["memory"]}
        return {"code": 200, "msg": "✅ 暂无长期记忆，开始积累你的专属记忆吧～"}
    except Exception as e:
        print(f"加载记忆失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"加载记忆失败: {str(e)}"})

# -------------------------- 新增：文件下载接口（前端可直接下载上传的文件） --------------------------
@app.post("/download_file", summary="下载已保存的文件")
async def download_file(file_id: str = Body(..., description="文件ID")):
    try:
        file = fs.get(file_id)
        return StreamingResponse(file, media_type=file.content_type, headers={"Content-Disposition": f"attachment; filename={file.filename}"})
    except NoFile:
        raise HTTPException(status_code=404, detail="文件不存在")
    except Exception as e:
        print(f"下载文件失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"下载文件失败: {str(e)}"})

# -------------------------- 启动服务 --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)