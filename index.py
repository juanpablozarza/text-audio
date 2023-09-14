from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from transformers import AutoProcessor, BarkModel
import nltk
import numpy as np
import boto3
import torch
import logging

app = Flask(__name__)
model_size = "large-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
# Run on GPU with FP16
whisper = WhisperModel(model_size, device=device, compute_type="int8_float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Download NLTK Data
nltk.download("punkt")

kinesis = boto3.client("kinesis", region_name="us-west-1")
STREAM_NAME = "AudioEdGen"

# Load processor and model when server starts
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)

model.enable_cpu_offload()

def transcribe_audio(file):
    print('Starting transcription')
    text = ""
    segments, info = whisper.transcribe(file, beam_size=5)
    for segment in segments:
        text += segment.text
    print(text)
    return text


@app.route("/transcribe", methods=["POST"])
def upload_file():
    
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    transcribed_text = transcribe_audio(file)

    return jsonify({"transcribed_text": transcribed_text})


@app.route("/generateAudioFile", methods=["POST"])
def generateAudioFile():
    reqData = request.json
    textData = reqData.get("textData")
    sentences = nltk.sent_tokenize(textData)
    voice_preset = "v2/en_speaker_9"
    for sentence in sentences:
        # Tokenize the input
        inputs = processor(sentence, voice_preset=voice_preset, return_tensors="pt")
        audio_array = model.generate(
            **inputs,
        )
        audio_array = audio_array.cpu().numpy().squeeze()
        audio_bytes = audio_array.astype(np.float32).tobytes()
        put_to_kinesis(audio_bytes)
    return {"Status": "Completed"}


def put_to_kinesis(audio_bytes):
    kinesis.put_record(
        StreamName=STREAM_NAME, Data=audio_bytes, PartitionKey="partitionKey"
    )


port = 8080

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    print("App running on port", port)