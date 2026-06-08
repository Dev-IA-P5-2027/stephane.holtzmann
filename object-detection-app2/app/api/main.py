from fastapi import FastAPI, UploadFile, File
from app.services.image_service import process_uploaded_file_for_api

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API Object Detection OK"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    result = process_uploaded_file_for_api(file)
    return result