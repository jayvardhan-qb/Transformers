from transformers import pipeline

# google-t5/t5-base
# google/mt5-base
# facebook/mbart-large-50-many-to-many-mmt

class Translator():
    def __init__(self, model_name = "google-t5/t5-small"):
    # def __init__(self, model_name = "google/mt5-base"):
    # def __init__(self, model_name = "google-t5/t5-base"):
    # def __init__(self, model_name = "ai4bharat/indictrans2-en-indic-dist-200M"):
        self.translator = pipeline("translation_en_to_fr", model = model_name, max_length = 100)

    def user_input(self):
        text = input("Enter text to translate: ").strip()
        text = " ".join(text.split())
        return text
    
    def translate(self, text: str):
        try:
            output = self.translator(text, max_length = 100)

            return output[0]['translation_text'].strip()
        except Exception as e:
            print(f"Error: {e}")
            return ""
        
if __name__ == "__main__":
    translation = Translator()
    text = translation.user_input()
    output = translation.translate(text)

    print(output)