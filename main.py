import asyncio
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from gridfs import GridFS, NoFile
import uvicorn
from fastapi import Request, Depends, HTTPException, Form

# 加载环境变量
load_dotenv()

# 检查必要的环境变量
required_env_vars = ["MONGO_URI", "MONGO_DB_NAME", "DASHSCOPE_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"环境变量 {var} 未设置")

# 创建FastAPI应用实例，移除已弃用的请求体大小参数
app = FastAPI(title="AI纳界助理", description="永久存储/智能记忆/深度检索/自然语言操控/对话页")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管静态前端页面
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# 数据库初始化 - MongoDB 永久存储
client = MongoClient(os.getenv("MONGO_URI"))
try:
    # 验证MongoDB连接
    client.admin.command('ping')
    print("✅ MongoDB连接成功")
except Exception as e:
    print(f"❌ MongoDB连接失败: {e}")

# 数据库和集合定义
db = client[os.getenv("MONGO_DB_NAME")]
fs = GridFS(db)
col_chat_history = db["chat_history"]  # 聊天记录
col_user_memory = db["user_memory"]    # 用户关键记忆（爱吃鱼、不吃辣等）
col_file_meta = db["file_meta"]        # 文件元信息+抽取的内容

# 导入路由
from ai_chat import *
from file_operations import *

# 首页路由
@app.get("/", summary="仿豆包风格的AI纳界助理对话首页")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        # 使用正确的uvicorn参数设置请求体大小限制（100MB）
        max_request_size=100 * 1024 * 1024  # 100MB
    )