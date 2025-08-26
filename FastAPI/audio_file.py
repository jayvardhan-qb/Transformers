from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
import soundfile as sf
import io

app = FastAPI(title="Audio Transcription API")

model_name = "openai/whisper-base"
transcriber = pipeline("automatic-speech-recognition", model=model_name)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

        result = transcriber(audio_array)
        return {"filename": file.filename, "Transcription": result["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")