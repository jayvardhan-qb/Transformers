from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title = "Text Summarizer")

class ContentSummarizer():
    def __init__(self, model_name = "t5-small"):
        self.summarizer = pipeline("summarization", model = model_name)

    def summarize(self, text: str, max_length: int = 150) -> str:
        if not text:
            return "No text provided for summarization"
        
        try:
            output = self.summarizer(
                text,
                max_new_tokens = max_length,
                do_sample = False,
                truncation = True
            )

            return output[0]["summary_text"]
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Summary generation failed"

summarizer = ContentSummarizer()

class TextRequest(BaseModel):
    text: str
    max_length: int = 150

@app.post("/summarize")
async def summarize_text(request: TextRequest):
    try:
        request.text = input("Enter text for summarization: ")
        summary = summarizer.summarize(request.text, max_length = request.max_length)
        
        return JSONResponse(content = {
            "original_text": request.text,
            "summary": summary
        })
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))