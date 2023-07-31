from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
from Find_Child import find_child
from PIL import Image
import io
import base64

########################     ########################################
# start server with FASTAPI
########################     ########################################
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
    
    result = find_child(f"C:\\kobbi\\endProject\\images_from_server\\{files[1].filename}", f"C:\\kobbi\\endProject\\images_from_server\\{files[0].filename}")
    print(result)
    # Assuming result is a dictionary with details you want to send back
   # Prepare response dict
    response = {}

    for i, (filename, points) in enumerate(result):
        # Load the image
        image_path = f"C:\\kobbi\\endProject\\TSKinFace_Data\\All_Data\\FMD_FMS_FMSD\\{filename}"
        image = Image.open(image_path)
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Add the image and info to the response
        response[filename] = {
            "image": img_str,
            "points": points,
            "strong_match": points >= 5
        }
    return response
