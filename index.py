from flask import Flask, request, jsonify, send_file
from faster_whisper import WhisperModel
import nltk
import numpy as np
import boto3
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from datasets import load_dataset
import io
from scipy.io.wavfile import write
from datetime import datetime

app = Flask(__name__)
model_size = "large-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Run on GPU with FP16
whisper = WhisperModel(model_size, device=device, compute_type="float16")
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
print('Running on', device)
# processor = AutoProcessor.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

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


@app.route("/generateAudioFile/<uid>", methods=["POST"])
def generateAudioFile(uid):
    print('Generating audio file...')
    reqData = request.json
    textData = reqData.get("textData")
    sentences = nltk.sent_tokenize(textData)
    voice_preset = "v2/en_speaker_9"
    audio_files = []
    for sentence in sentences:
        inputs = processor(text=sentence, return_tensors="pt")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, 16000, speech.numpy())
        wav_bytes = byte_io.read()
        audio_files.append(upload_to_s3(wav_bytes, uid))
        byte_io.seek(0)
    return jsonify(audio_files)


def upload_to_s3(bytes,partition_key):
    # Format the datetime object to a string
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')    
    object_name = f'{partition_key}/{formatted_datetime}.wav'
    # Upload to S3
    s3 = boto3.client('s3')
    s3.put_object(Body=bytes, Bucket="audios-edgen", Key=object_name)
    # Generate Pre-signed URL
    presigned_url = s3.generate_presigned_url('get_object',
                                              Params={'Bucket': "audios-edgen",
                                                      'Key': object_name},
                                              ExpiresIn=3600) # URL expires in 1 hour
    print(presigned_url)
    return presigned_url
   
def split_and_upload(file_data, partition_key):
    max_size = 1048576  # 1 MB in bytes
    if len(file_data) <= max_size:
        upload_to_kinesis(file_data, partition_key)
    else:
        mid_index = len(file_data) // 2
        split_and_upload(file_data[:mid_index], partition_key)
        split_and_upload(file_data[mid_index:], partition_key)



port = 8080

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    print("App running on port", port)
