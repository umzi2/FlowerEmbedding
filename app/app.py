from fastapi import FastAPI, File, UploadFile

from utils.download_dataset import download_flower
from .model import load_model, search_image, upload_pth_embedded
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI()

# Загружаем модель
model = load_model()
download_flower()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Сохраняем загруженное изображение в памяти
    image_bytes = await file.read()

    # Применяем поиск по тестовой выборке
    result = search_image(image_bytes, model)

    return JSONResponse(content=result)


@app.get("/upload")
async def upload_embedded():
    try:
        upload_pth_embedded(model)
        return HTMLResponse(content="<h1>Upload successful!</h1>", status_code=200)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Upload failed: {str(e)}</h1>", status_code=500
        )
