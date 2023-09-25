from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import uvicorn
from starlette.requests import Request, HTMLResponse
from fastapi.templating import Jinja2Templates
import io
from train import CNNModel
import torchvision.transforms as transforms

templates = Jinja2Templates(directory="templates")

app = FastAPI()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load("mnodels/fashionModel.pth"), strict=False)
model.eval()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict/")
async def predict_fashion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = image.convert("L")
        image = image.resize((28, 28))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            predicted_class = torch.argmax(outputs[0]).item()
            class_name = class_names[predicted_class]
        
        response_data = {
            "predicted_class": class_name,
            "name": file.filename
        }

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
