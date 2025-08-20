import speech_recognition as sr
import numpy as np
from transformers import pipeline
import soundfile as sf

class AudioFile():
    def __init__(self, model_name = "openai/whisper-base"):
        self.transcriber = pipeline("automatic-speech-recognition", model = model_name)
        self.sample_rate = 16000

    def record_audio(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.5
        mic = sr.Microphone(sample_rate = self.sample_rate)

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration = 1)
            print("Start speaking. Recording will stop when silence is detected.")
            audio = recognizer.listen(source)

        print("Recording complete!")

        return audio
    
    def save_audio(self, audio, filename = "audio-file.wav"):
        audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0

        sf.write(filename, audio_data, self.sample_rate)
        print(f"Audio saved to {filename}.")

        return audio_data
    
    def transcribe(self, audio_array):
        result = self.transcriber(audio_array)

        return result["text"]
    
if __name__ == "__main__":
    stt = AudioFile()
    audio = stt.record_audio()
    audio_array = stt.save_audio(audio, "audio-file.wav")
    transcription = stt.transcribe(audio_array)

    print("\nTranscription:")
    print(transcription)