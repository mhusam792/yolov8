# -*- coding: utf-8 -*-
"""imageToText.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mCDYPzGQylDs9tOJGerUxQorskSof_cC
"""

# from transformers import pipeline
# captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
# print(captioner("/content/bag.jpg"))

# !pip install diffusers
# !pip install -q -U transformers==4.37.2
# !pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
# !git clone https://github.com/mhusam792/yolov8
# ! pip install ultralytics
# ! pip3 install torch torchvision torchaudio
# ! pip install fastapi

import torch
from transformers import BitsAndBytesConfig
import requests
from PIL import Image
from yolov8.api import yolo_model

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
from transformers import pipeline

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

def get_descrption(image_url):
  model_path = '/content/yolov8/last.pt'
  yolo_result = yolo_model(model_full_path=model_path, image=image_url)
  image = Image.open(requests.get(image_url, stream=True).raw)
  max_new_tokens = 200
  prompt = "USER: <image>\nGive me an  description of the basic element in the image with color and  pay attention to detail and ignor the background \nASSISTANT:"
  outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
  final_descrption=f"Description of the lost item : \n" +outputs[0]["generated_text"]
  index = final_descrption.index("ASSISTANT:") + len("ASSISTANT:")
  description = final_descrption[index:].strip()
  yolo_result["description"]=description
  return yolo_result

# print(get_descrption("https://m.media-amazon.com/images/I/61+r3+JstZL.jpg"))