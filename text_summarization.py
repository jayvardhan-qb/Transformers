import PyPDF2
from transformers import pipeline
from typing import List

class PDFSummarizer:
    # def __init__(self, model_name: str = "facebook/bart-large-cnn"):
    def __init__(self, model_name: str = "t5-small"):
        # Other model: "google/pegasus-xsum"
        self.summarizer = pipeline("summarization", model=model_name)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                return "\n".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
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
    
    def summarize(self, pdf_path: str, max_length: int = 90) -> str:
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            return "No text could be extracted from the PDF."
        
        # summaries = []
        # for i, chunk in enumerate(chunks, 1):
        #     try:
        #         # Print progress every 10 chunks
        #         if i % 10 == 0:
        #             print(f"Processing chunk {i}/{len(chunks)}...")
                
        #         output = self.summarizer(
        #             chunk,
        #             max_new_tokens=max_length,
        #             do_sample=False,
        #             truncation=True
        #         )
        #         summaries.append(output[0]['summary_text'])
        #     except Exception as e:
        #         print(f"Error in chunk {i}: {str(e)}")
        #         continue
    
        #     return " ".join(summaries) if summaries else "Summary generation failed"

        summaries = [
            self.summarizer(chunk, max_length=max_length, do_sample=False)[0]['summary_text']
            for chunk in self.chunk_text(text)
        ]

        return " ".join(summaries)

if __name__ == "__main__":
    summarizer = PDFSummarizer()
    
    pdf_file = r"C:\Users\chuda\OneDrive\Desktop\TICKET TO USA\Transformers\data\fines.pdf"
    
    summary = summarizer.summarize(pdf_file, max_length=90)
    
    print("PDF Summary:")
    print(summary)