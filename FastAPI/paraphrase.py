from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Paraphraser API")

paraphraser_pipeline = pipeline("text2text-generation", model = "humarin/chatgpt_paraphraser_on_T5_base")

class ParaphraseRequest(BaseModel):
    text: str
    num_return_sequences: int = 4
    temperature: float = 1.1


@app.post("/paraphrase")
async def paraphrase_text(request: ParaphraseRequest):
    if len(request.text.split()) < 5:
        raise HTTPException(status_code=400, detail="Text must have at least 5 words.")

    try:
        results = paraphraser_pipeline(
            f"paraphrase: {request.text}",
            num_return_sequences=request.num_return_sequences,
            max_length=len(request.text) + 20,
            do_sample=True,
            temperature=request.temperature,
        )
        paraphrases = [result['generated_text'] for result in results]
        return {"original_text": request.text, "paraphrases": paraphrases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))