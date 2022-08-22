import torch
import numpy as np

from pydantic import BaseModel
from typing import List

class Photo(BaseModel):
    photo: List[float]
    width: int
    height: int

from fastapi import FastAPI

app = FastAPI()
'''
@app.on_event("startup")
def load_model():
    global midas
    global transform
    global device
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
'''
@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}

'''
@app.post('/predict')
def get_depth(data: Photo):
    received = data.dict()
    width = received['width']
    height = received['height']
    flattened_photo = np.array(received['photo'])
    img = flattened_photo.reshape((height, width, 3))

    input_batch = transform(img).to(device)

    with torch.no_grad():
      prediction = midas(input_batch)

      prediction = torch.nn.functional.interpolate(
          prediction.unsqueeze(1),
          size=img.shape[:2],
          mode="bicubic",
          align_corners=False,
      ).squeeze()

      output = prediction.cpu().numpy().tolist()

    return {'depth': output}
'''
