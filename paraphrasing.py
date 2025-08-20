from transformers import pipeline

class TextParaphraser:
    def __init__(self):
        self.paraphraser = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")
        # self.paraphraser = pipeline("text2text-generation", model="AventIQ-AI/t5-paraphrase-generation")

    def user_input(self):
        text = input("Enter text to paraphrase: ").strip()
        if len(text.split()) >= 5: 
            return text
        print("Please enter at least 5 words")

    def paraphrase(self, text: str):
        try:
            results = self.paraphraser(
                f"paraphrase: {text}",
                # num_beams = 5,
                num_return_sequences = 4,
                max_length = len(text) + 20,  
                do_sample = True,
                temperature = 1.1,
                # top_k = 50
            )
            return [result['generated_text'] for result in results]
        except Exception as e:
            print(f"Error: {e}")
            return []

if __name__ == "__main__":
    paraphraser = TextParaphraser()
    text = paraphraser.user_input()

    outputs = paraphraser.paraphrase(text)
    print("\nParaphrases:")
    for i, output in enumerate(outputs, 1):
        print(f"{i}. {output}")
    print()