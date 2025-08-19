from fastapi import FastAPI, UploadFile, File
from typing import List
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import io

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load SigLIP so400m model ---
model_id = "google/siglip-so400m-patch14-384"
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)  # chỉ xử lý ảnh
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

app = FastAPI()

def get_embedding(image_bytes: bytes):
    """Encode a single image to embedding"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return emb[0].cpu().numpy().tolist()

@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    """Single image embedding"""
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)
    return {"embedding": embedding}

@app.post("/embed-batch")
async def embed_images(files: List[UploadFile] = File(...)):
    """Batch image embedding"""
    images = []
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        images.append(image)

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    embs = embs / embs.norm(dim=-1, keepdim=True)

    embeddings = [emb.cpu().numpy().tolist() for emb in embs]
    return {"embeddings": embeddings, "count": len(embeddings)}
