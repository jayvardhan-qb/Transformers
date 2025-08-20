import PyPDF2
from transformers import pipeline
import re

class PDFSummarizer:
    def __init__(self, model_name: str = "t5-small"):
        self.summarizer = pipeline("summarization", model=model_name)
        config = self.summarizer.model.config
        self.model_max_length = getattr(config, "max_position_embeddings", 
                                      getattr(config, "n_positions", 512))
        print(f"Using model max length: {self.model_max_length}")
    
    def extract_text(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                text = "\n".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
                return text if text.strip() else ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def chunk_text(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split on sentence boundaries
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_token_count = len(sentence.split()) * 1.4  # Approximate ratio
            
            if current_token_count + sentence_token_count > self.model_max_length * 0.8:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
                
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
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
                    do_sample=False,
                    truncation=True
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