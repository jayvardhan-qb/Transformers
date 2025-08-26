from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from transformers import pipeline
import PyPDF2
from typing import List
import tempfile

app = FastAPI(title = "PDF Summarizer")

class PDFSummarizer():
    def __init__(self, model_name = "t5-small"):
        self.summarizer = pipeline("summarization", model = model_name)

    def extract_text(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                text = "\n".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""
        
    def chunk_text(self, text: str, chunk_size: int = 600) -> List[str]:
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    
    def summarize(self, pdf_path: str, max_length: int = 150) -> str:
        text = self.extract_text(pdf_path)
        if not text:
            return "No text extracted from PDF"
        
        chunks = self.chunk_text(text)
        print(f"Processing {len(chunks)} chunks...")

        summaries = []
        for i, chunk in enumerate(chunks, 1):
            try:
                output = self.summarizer(
                    chunk,
                    max_new_tokens = max_length,
                    do_sample = False,
                    truncation = True
                )

                summaries.append(output[0]['summary_text'])
            except Exception as e:
                print(f"Error in chunk {i}: {str(e)}")
                continue

        return " ".join(summaries) if summaries else "Summary generation failed!"
    
summarizer = PDFSummarizer()

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), max_length: int = 150):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code = 400, detail = "Only PDF files allowed to be uploaded.")
    
    try:
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        summary = summarizer.summarize(temp_file_path, max_length = max_length)
        
        return JSONResponse(content = {"summary": summary})
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))