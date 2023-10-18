import uuid
from flask import Flask, request, jsonify, send_file
from faster_whisper import WhisperModel
import nltk
import numpy as np
import boto3
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, VitsModel, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset
import soundfile as sf
from datasets import load_dataset
import io
from scipy.io.wavfile import write
from datetime import datetime

from bark import SAMPLE_RATE, generate_audio, preload_models

preload_models()
mysp= __import__("my-voice-analysis")
# tts = CS_API()
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


# Text to speech model for spanish
\

# Text classifier 
textClassfierModelName = 'qanastek/51-languages-classifier'
textClassfierTokenizer = AutoTokenizer.from_pretrained(textClassfierModelName)
textClassfierModel = AutoModelForSequenceClassification.from_pretrained(textClassfierModelName)
text_classifier = TextClassificationPipeline(model=textClassfierModel, tokenizer=textClassfierTokenizer) 

# Download NLTK Data
nltk.download("punkt")

kinesis = boto3.client("kinesis", region_name="us-west-1")
STREAM_NAME = "AudioEdGen"
# Load processor and model when server starts
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)
# bark_preprocess = AutoProcessor.from_pretrained("suno/bark")
# bark = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
voice_preset = "v2/en_speaker_6"

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
    print(request.files)
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
    lang = textClassifier(textData)
    sampRate = 0
    if lang == 'es-ES':
        speech = spanishTTS(textData)
        sampRate = SAMPLE_RATE
    else:        
      inputs = processor(text=textData, return_tensors="pt")
      embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
      speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
      speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
      speech =speech.numpy()
      sampRate = 16000
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sampRate, speech)
    wav_bytes = byte_io.read()
    byte_io.seek(0)
    return upload_to_s3(wav_bytes, uid)

@app.route("/audioEval/", methods=['POST'])
def audioEval():
    audio_file = request.files['audio']
    random_uid = uuid.uuid4()
    with open(f'uploads/{random_uid}.wav', 'wb') as f:
     f.write(audio_file.content)
    with open(f'uploads/{random_uid}.wav', 'rb') as file:
        result = mysp.mysppron(file,f'uploads/{random_uid}.wav')
        print(result)
        return result
    

def spanishTTS(textData):
    audio_array = generate_audio(textData, history_prompt="v2/es_speaker_8")
    write("results/output.wav", rate=SAMPLE_RATE, data=audio_array)
    return audio_array

def textClassifier(textData):
    output = text_classifier(textData)
    print(output[0]['label'])
    return output[0]['label']

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
   

port = 8080

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    print("App running on port", port)
