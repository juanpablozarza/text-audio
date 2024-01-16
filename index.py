import uuid
from flask import Flask, request, jsonify

import numpy as np
import boto3
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, VitsModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM , AutoProcessor, TextClassificationPipeline, AutoModelForSpeechSeq2Seq,  pipeline
from datasets import load_dataset
import soundfile as sf
from datasets import load_dataset
import io
from pydub import AudioSegment
import librosa
from scipy.io.wavfile import write
from datetime import datetime
import scipy
import sys
import os
sys.path.insert(0, './bark')
from bark import SAMPLE_RATE, generate_audio, preload_models
from peft import PeftModel, PeftConfig
from werkzeug.utils import secure_filename
preload_models()
from peft import PeftModel, PeftConfig
import ast
import logging
from bark import SAMPLE_RATE, generate_audio, preload_models
import tempfile
import subprocess
logging.basicConfig(level=logging.INFO)
from pydub import AudioSegment
from pydub.playback import play
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import audio_effects as ae
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

mysp= __import__("my-voice-analysis")
# tts = CS_API()
app = Flask(__name__)
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
model_size = "large-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Run on GPU with FP16
# whisper = WhisperModel(model_size, device=device, compute_type="float16")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

whisper_model_id = "openai/whisper-large-v3"

whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper.to(device)

processor_whisper = AutoProcessor.from_pretrained(whisper_model_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper,
    tokenizer=processor_whisper.tokenizer,
    feature_extractor=processor_whisper.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
# Text to speech model for spanish
# Text classifier 
textClassfierModelName = 'papluca/xlm-roberta-base-language-detection'
textClassfierTokenizer = AutoTokenizer.from_pretrained(textClassfierModelName)
textClassfierModel = AutoModelForSequenceClassification.from_pretrained(textClassfierModelName)
text_classifier = TextClassificationPipeline(model=textClassfierModel, tokenizer=textClassfierTokenizer) 

# download and load all models
preload_models()

kinesis = boto3.client("kinesis", region_name="us-west-1")
STREAM_NAME = "AudioEdGen"
# Load processor and model when server starts
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
voice_preset = "v2/en_speaker_6"



# # Mini model used for lang separation

peft_model_id = "Juanpablozarza292/T5-lang-classifier-7b1-lora"
config = PeftConfig.from_pretrained(peft_model_id)
pipeline = pipeline('text2text-generation', model = "MBZUAI/LaMini-Flan-T5-783M")

lang_sep_model = pipeline.model
lang_sep_tokenizer = pipeline.tokenizer
# Load the Lora model
lang_sep_model = PeftModel.from_pretrained(lang_sep_model, peft_model_id)

def transcribe_audio(file):
    # Generate a unique filename with the original file extension
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join("./uploads", secure_filename(filename))
    # Save the original file
    file.save(file_path)
    # Check if the file is in CAF format and convert to WAV if necessary
    if ext.lower() == '.caf':
        audio = AudioSegment.from_file(file_path, format='caf')
        wav_path = file_path.replace('.caf', '.wav')
        audio.export(wav_path, format='wav')
        # Update file_path to the new WAV file
        file_path = wav_path
    # Process the file with Whisper
    result = whisper_pipe(file_path)
    print(result['text'])
    # Delete the file(s) after processing
    os.remove(file_path)
    # if ext.lower() == '.caf':
    #     os.remove(wav_path)
    return result['text']
    
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

def fix_string_literals(s):
    # Fix missing closing commas
    s = s.replace('", "', '", "').replace('",  "', '", "').replace('",   "', '", "')

    # Try to convert the fixed string to a list using ast.literal_eval
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except:
        pass

    # If the above fails, manually split and clean up the string
    chunks = s.split('", "')
    cleaned_chunks = []
    for chunk in chunks:
        cleaned_chunk = chunk.strip().strip('"')
        cleaned_chunks.append(cleaned_chunk)

    return cleaned_chunks


def slowdownAudio(uid: str, audio_array:np.array):
    # Check user's classId and slow down the audio accordingly
    userRef = db.collection("users").document(uid)
    userDoc = userRef.get().to_dict()
    lastLesson = userDoc["lessons"][-1]
    lessonRef = db.collection("lessons").document(lastLesson)
    lessonDoc = lessonRef.get().to_dict()
    classId = lessonDoc["classId"]
    if classId.startswith("A"): 
        playback_speed = 0.80
    elif classId.startswith("B"):
        playback_speed = 0.90
    elif classId.startswith("C"):
        playback_speed = 1.0
    # Load the audio file
    audio = AudioSegment.from_file(io.BytesIO(audio_array), format="wav")
    # Slow down the audio to half its speed
    slowed_audio = ae.speed_down(audio, playback_speed)
    # Save the slowed audio
    slowed_audio_bytes = io.BytesIO()
    slowed_audio.export(slowed_audio_bytes, format="wav")
    slowed_audio_bytes = slowed_audio_bytes.getvalue()
    return slowed_audio_bytes
     
@app.route("/generateAudioFile/<uid>", methods=["POST"])
def generateAudioFile(uid):
    print('Generating audio file...')
    reqData = request.json
    textData = reqData.get("textData")
    langs = textClassifier(textData)
    print(f"Langs: {langs}")
    combined_audio = [] 
    for chunk in langs: 
        print(f"Chunk: {chunk}")
        spanish_sim_langs = ["it","fr", "pt","ro", "es"] 
        if not langs[chunk] in spanish_sim_langs:
            inputs = processor(text=chunk, return_tensors="pt")
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            sampRate = 16000
            # Convert tensor to numpy array
            speech = speech.numpy()
            resampled_segment =librosa.resample(speech, orig_sr=sampRate, target_sr=SAMPLE_RATE)
            combined_audio.append(resampled_segment)
        else:
            # Generate audio from text
            speech = generate_audio(chunk, history_prompt="v2/es_speaker_9")
            sampRate = SAMPLE_RATE
            combined_audio.append(speech)

    # Export combined audio
    print(f"Sample rate: {SAMPLE_RATE}")
    resampled_speech = np.concatenate(combined_audio)
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, SAMPLE_RATE, resampled_speech)
    wav_bytes = byte_io.read()
    byte_io.seek(0)
    slow_down_wav_bytes = slowdownAudio(uid, wav_bytes)
    # Slow down the audio if necessary
    return upload_to_s3(slow_down_wav_bytes, uid)

@app.route("/audioEval", methods=['POST'])
def audioEval():
    audio_file = request.files['audio']
    text = request.form['text']
    random_uid = uuid.uuid4()
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)
    with open(f'uploads/{audio_file.filename}', 'rb') as file:
        try:
            user_phonetic = audio_to_phonetics(file)
        except:
            return jsonify({"error": "Audio file not supported"})    
        score = compare_phonetics_score(user_phonetic,text)
        print(f"Score: {score}")

        return score

def compare_phonetics_score(user_phonetic:list, text: str):
    # For every missmatch in the phonetics, substract 1 to the score
    phonetic_dict = cmudict.dict()
    score = 100
    words = text.split()
    phonetics = [phonetic_dict[word.lower()][0] for word in words if word.lower() in phonetic_dict]
    for i in range(len(user_phonetic)):
        if user_phonetic[i] != phonetics[i]:
            score -= 1
    return score

     
def audio_to_phonetics(audio_file): 
    phonetic_dict = cmudict.dict()
    text = transcribe_audio(audio_file) 
    words = text.split()
    phonetics = [phonetic_dict[word.lower()][0] for word in words if word.lower() in phonetic_dict]
    return phonetics

def spanishTTS(textData):
    audio_array = generate_audio(textData, history_prompt="v2/es_speaker_8")
    write("results/output.wav", rate=SAMPLE_RATE, data=audio_array)
    return audio_array

def textClassifier(textData):
    # inputs = lang_sep_tokenizer(f"### Instruction: Split the sentence into phrases according to language. sentence: {textData}", return_tensors='pt')
    # predictions = lang_sep_model.generate(**inputs)
    # pred = lang_sep_tokenizer.decode(predictions[0], skip_special_tokens=True)
    # clean_chunks = fix_string_literals(pred)
    lang_chunks = {}
    # for chunk in clean_chunks:
    output = text_classifier(textData)
    lang_chunks[textData] = output[0]['label']  
    return lang_chunks

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
                                              ExpiresIn=7200) # URL expires in 1 hour
    print(presigned_url)
    return presigned_url

port =8080
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    print("App running on port", port)

