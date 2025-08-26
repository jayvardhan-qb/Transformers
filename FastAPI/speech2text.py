from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
import numpy as np
import soundfile as sf
import io
import librosa

app = FastAPI(title="Speech-to-Text API")

asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_stream, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_stream.dtype != np.float32:
            audio_stream = audio_stream.astype(np.float32)

        if sample_rate != 16000:
            audio_stream = librosa.resample(audio_stream, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        result = asr_pipeline(audio_stream)
        return {"transcription": result["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")