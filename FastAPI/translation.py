from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Translator API")

# "google/mt5-base"
# "facebook/mbart-large-50-many-to-many-mmt"

model_name = "google-t5/t5-small"
translator = pipeline("translation_en_to_fr", model=model_name, max_length=100)

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        text = " ".join(request.text.strip().split())
        if not text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        output = translator(text, max_length=100)
        translated = output[0]['translation_text'].strip()

        return TranslationResponse(
            original_text=text,
            translated_text=translated
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))