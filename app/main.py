import torch
import io
from pydantic import BaseModel
from torchvision.models import ResNet 
from fastapi import FastAPI, File, UploadFile, Depends
from PIL import Image 
import torch.nn.functional as F

categories = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]

## We must define a __init__.py file in the app folder to make it a package
#from app.model import ...

# We are defining a subclass of BaseModel to structure the response data (our schema)
class Result(BaseModel):
    category: str
    confidence: float

app = FastAPI()

@app.get('/')
def read_root():
    return {"Message": "API is running. Visit /docs for the Swagger API documentation."}

@app.post('/predict/', response_model=Result)
async def predict() -> Result:
    return Result(category="freshapple", confidence=0.99)
