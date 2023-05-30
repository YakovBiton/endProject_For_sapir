from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
from Find_Child import find_child
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows CORS from your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        with open(f"C:\\kobbi\\endProject\\images_from_server\\{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    # Now you can call your Python functions with the paths of the images
    
    result = find_child(f"C:\\kobbi\\endProject\\images_from_server\\{files[0].filename}", f"C:\\kobbi\\endProject\\images_from_server\\{files[1].filename}")
    print(result)
    # Assuming result is a dictionary with details you want to send back
    return result
