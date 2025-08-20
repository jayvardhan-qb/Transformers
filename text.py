import PyPDF2
from transformers import pipeline
from typing import List

class PDFSummarizer:
    def __init__(self, model_name: str = "t5-small"):
        self.summarizer = pipeline("summarization", model=model_name)
    
    def extract_text(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                text = "\n".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
                return text if text.strip() else ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 600) -> List[str]:
        sentences = text.split('. ')
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
                if i % 10 == 0:
                    print(f"Processing chunk {i}/{len(chunks)}...")
                
                output = self.summarizer(
                    chunk,
                    max_new_tokens=max_length,
                    do_sample = False,
                    truncation = True
                )
                summaries.append(output[0]['summary_text'])
            except Exception as e:
                print(f"Error in chunk {i}: {str(e)}")
                continue
    
        return " ".join(summaries) if summaries else "Summary generation failed"

if __name__ == "__main__":
    summarizer = PDFSummarizer()
    pdf_file = r"C:\Users\chuda\OneDrive\Desktop\TICKET TO USA\Transformers\data\Lease Deed.pdf"
    print("PDF Summary:", summarizer.summarize(pdf_file))