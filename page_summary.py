from transformers import pipeline

class ContentSummarizer():
    def __init__(self, model_name: str = "t5-small"):
    # def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("paraphrasing", model = model_name)

    def user_input(self) -> str:
        user = input("Enter the text to summarize: ").strip()
        
        return user
    
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

            return output[0]['summary_text']
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Summary generation failed"

if __name__ == "__main__":
    summarizer = ContentSummarizer()
    text = summarizer.user_input()
    summary = summarizer.summarize(text)
    print("\nSummary:", summary)