import os
import shutil
import pathlib
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Path, Query, File, UploadFile

app = FastAPI()

UPLOAD_DIR = pathlib.Path(r'src\uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/uploads')
async def upload_documents(file : UploadFile = File(...)):
    '''
    End points ot upload any documents
    '''
    
    try:
        file_path = UPLOAD_DIR / file.filename
        
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except:
        raise HTTPException(status_code=400, detail='Bad data try to upload this like file document')    
    
    finally:
        return {
            'message' : 'document upload successfully!',
            'Filename' : file.filename,
            'save_to' : file_path
        }
