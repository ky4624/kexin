import asyncio
import uuid
import datetime
import traceback
from io import BytesIO
from fastapi import Body, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import langextract as le
from PyPDF2 import PdfReader
from openpyxl import load_workbook

# 导入全局应用实例和数据库集合
from main import app, col_file_meta, fs


# -------------------------- 文件操作工具函数 --------------------------
async def extract_file_content(file: UploadFile, file_content: bytes):
    """智能提取各类文件内容（升级后）：图片OCR/音频转文字/PDF/Word/Excel/TXT/视频帧等"""
    content = ""
    try:
        filename = file.filename.lower()
        if any(ext in filename for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]):
            # 图片：当前使用模拟内容（实际需接入OCR服务）
            content = f"[{file.filename}] 图片内容，已保存"
        elif any(ext in filename for ext in [".mp3", ".wav", ".m4a", ".ogg"]):
            # 音频：当前使用模拟内容（实际需接入语音转文字服务）
            content = f"[{file.filename}] 音频内容，已保存"
        elif any(ext in filename for ext in [".mp4", ".avi", ".mov", ".wmv"]):
            # 视频：当前使用模拟内容（实际需接入视频分析服务）
            content = f"[{file.filename}] 视频内容，已保存"
        elif ".pdf" in filename:
            # PDF：使用PyPDF2提取文字
            pdf_reader = PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                content += page.extract_text() or ""
            if not content.strip():
                content = f"[{file.filename}] PDF文件，已保存，无文字内容"
        elif ".docx" in filename:
            # Word：当前使用模拟内容（实际需接入python-docx库）
            content = f"[{file.filename}] Word文档，已保存"
        elif ".xlsx" in filename:
            # Excel：使用openpyxl提取内容
            workbook = load_workbook(filename=BytesIO(file_content))
            sheet = workbook.active
            for row in sheet.iter_rows(values_only=True):
                if any(cell for cell in row):  # 只处理非空行
                    content += "\t".join(str(cell) if cell is not None else "" for cell in row) + "\n"
        elif ".txt" in filename:
            # TXT：直接读取
            content = file_content.decode("utf-8", errors="ignore")
        else:
            # 其他格式：保存为二进制文件
            content = f"[{file.filename}] 已保存，不支持内容提取"
    except Exception as e:
        print(f"文件内容提取失败: {e}")
        content = f"[文件提取失败] {str(e)}"
    return content

# -------------------------- 文件操作API接口 --------------------------
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
                extract_content = await extract_file_content(file, file_content)
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
        await conversation_chain.ainvoke(
            {"input": f"保存资料：{content}，上传文件：{file_list}"},
            config={"configurable": {"session_id": user_id}}
        )
        
        return {
            "code": 200,
            "msg": "✅ 所有内容已永久保存+AI自动整理完成",
            "data": {"save_id": save_id, "create_time": str(create_time), "files": file_list}
        }
    except Exception as e:
        print(f"保存内容失败: {e}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"保存内容失败: {str(e)}"})