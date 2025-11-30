import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ..services import InferencePipeline, TaskExtractorService

app = FastAPI(title="Task Extraction API")
pipeline = None

@app.on_event("startup")
async def startup():
    global pipeline
    # In a real scenario, paths would come from env vars
    extractor = TaskExtractorService() 
    pipeline = InferencePipeline(extractor, os.environ.get("DEEPGRAM_API_KEY"))

class TextRequest(BaseModel):
    text: str

@app.post("/extract/text")
async def extract_text(req: TextRequest):
    return pipeline.process_text(req.text)

@app.post("/extract/audio")
async def extract_audio(file: UploadFile = File(...)):
    # Logic to save file and process
    return {"message": "Not implemented in this skeleton"}

@app.get("/health")
async def health():
    return {"status": "healthy"}