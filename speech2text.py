# AventIQ-AI/whisper-audio-to-text

import pyaudio
import numpy as np
from transformers import pipeline

# RATE is the sampling rate i.e. number of frames per second
# CHUNK_SIZE is the (arbitrarily chosen) number of frames the (potentially very long) signals are split into in this example
# RATE * SECONDS is the number of frames that should be recorded. Since the for loop is not repeated for each frame but only for each chunk, the number of loops has to be divided by the chunk size CHUNK_SIZE. This has nothing to do with samples.
# In PyAudio, a sample represents a single discrete measurement of an audio signal's amplitude at a specific point in time.
# A chunk is a block of audio data, typically defined by a frames_per_buffer or CHUNK size in PyAudio. It contains multiple frames, and therefore multiple samples. PyAudio processes audio in these chunks.

class SpeechToText():
    def __init__(self, model_name="openai/whisper-base"):
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name)
        self.sample_rate = 16000
        self.chunk_size = 1024

    def record_and_transcribe(self, seconds=5): 
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
    
        print(f"Recording {seconds} seconds... Speak now!")
        frames = []
    
        chunks_needed = int(self.sample_rate / self.chunk_size * seconds)
        for _ in range(chunks_needed):
            data = stream.read(self.chunk_size)
            frames.append(data)
    
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
        print("Transcribing...")
    
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # To convert int16 -> float32 for 16-bit audio
    
        result = self.transcriber(audio_data)
        return result["text"]

if __name__ == "__main__":
    stt = SpeechToText()
    input("Press Enter to start recording...")
    text = stt.record_and_transcribe(seconds=5)
    print(f"\nTranscription: {text}")