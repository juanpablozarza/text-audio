import random
import uuid
from flask import Flask, request, jsonify

import numpy as np
import boto3
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, VitsModel, AutoTokenizer, AutoModelForSequenceClassification, WhisperProcessor , AutoProcessor, TextClassificationPipeline, AutoModelForSpeechSeq2Seq,  pipeline
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
import Levenshtein as lev

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


whisper_model_id = "openai/whisper-medium"

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
    generate_kwargs= {"task":"transcribe"},
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
            speaker_embeddings = torch.tensor([-0.083527,
0.012485,
0.049985,
0.031337,
0.025022,
0.004565,
-0.056052,
0.005535,
0.027058,
0.016173,
-0.087251,
-0.063234,
0.055789,
0.023288,
0.055511,
0.070691,
0.019420,
0.025098,
0.005853,
0.028934,
0.027090,
0.041624,
-0.012846,
-0.062494,
-0.077126,
-0.004237,
-0.072781,
0.008591,
0.053436,
0.043467,
-0.003131,
0.050597,
0.023635,
0.008337,
0.033518,
-0.067284,
0.047457,
0.034881,
0.010889,
-0.026500,
0.025015,
0.007841,
0.031671,
0.043524,
0.012705,
-0.075070,
-0.013902,
0.010230,
-0.095866,
0.048938,
0.034235,
0.043036,
0.036524,
0.046171,
-0.070803,
-0.003606,
0.031867,
0.025131,
0.014806,
0.039392,
0.047069,
-0.006444,
-0.010164,
0.034079,
0.024216,
0.045070,
0.018767,
-0.049351,
-0.049383,
-0.071444,
0.025774,
0.007979,
0.011930,
0.010570,
0.034202,
0.036552,
0.034207,
0.009399,
-0.064141,
-0.059966,
-0.074989,
-0.072589,
-0.060343,
-0.058172,
-0.021793,
-0.071377,
-0.069220,
0.059970,
0.016272,
-0.003124,
0.026242,
-0.035552,
0.014359,
-0.064677,
0.027325,
-0.031501,
0.054269,
0.040125,
-0.042715,
-0.090477,
0.034263,
-0.089090,
-0.081731,
0.028031,
0.029211,
-0.016797,
0.033564,
0.055874,
0.051026,
0.030961,
-0.076870,
0.053882,
0.070681,
0.028664,
0.039277,
0.063602,
-0.075889,
0.030167,
-0.055822,
0.011224,
0.028345,
-0.065757,
0.032827,
0.036198,
-0.028497,
0.047815,
-0.082625,
0.023556,
0.035009,
0.033948,
0.022466,
0.045580,
0.032217,
0.048994,
0.022803,
-0.072212,
-0.079406,
0.022768,
-0.073604,
-0.026339,
0.026784,
0.004717,
-0.033679,
0.030171,
-0.078206,
0.021445,
0.033568,
-0.019373,
0.046734,
-0.071796,
0.050906,
-0.059707,
-0.095960,
0.033642,
-0.005209,
0.042978,
0.049926,
-0.081321,
-0.071315,
0.047978,
0.027088,
0.018199,
0.044458,
0.002889,
0.008176,
0.001834,
0.043577,
0.034753,
0.020089,
0.024789,
0.027102,
-0.087384,
0.030993,
-0.058148,
0.026530,
0.028448,
-0.077337,
0.030439,
0.028678,
-0.072006,
0.046584,
-0.070302,
-0.005343,
0.009930,
-0.070258,
-0.008075,
0.041729,
0.035520,
-0.077309,
-0.060715,
0.005723,
0.013208,
-0.073714,
0.008443,
0.032256,
0.000194,
0.037085,
0.024901,
0.029598,
-0.008946,
0.035929,
0.022474,
-0.058708,
0.009929,
0.046455,
0.010359,
0.057274,
0.000312,
0.037141,
0.009358,
-0.000812,
-0.058167,
-0.051249,
0.034571,
-0.067419,
0.042781,
0.023887,
-0.063467,
0.024780,
0.033695,
-0.070060,
0.012250,
-0.082982,
-0.019292,
0.015156,
0.037560,
0.043446,
0.028733,
0.029052,
0.030495,
-0.003653,
0.023902,
0.026120,
0.000914,
0.033131,
-0.011507,
0.038163,
-0.006308,
0.001219,
0.047875,
-0.052813,
0.011924,
-0.083050,
0.052497,
-0.071305,
0.060915,
0.043034,
0.038399,
0.043123,
0.022467,
0.035320,
0.034527,
0.006385,
-0.046004,
0.031289,
0.045867,
0.025179,
0.038305,
-0.059736,
0.040074,
0.016611,
-0.101911,
0.038910,
0.022501,
-0.000143,
0.047692,
0.044738,
0.038751,
0.013172,
-0.004119,
0.041775,
0.002027,
-0.011860,
-0.001643,
-0.059590,
0.025981,
0.052631,
0.012061,
-0.067825,
-0.065945,
-0.007522,
0.044044,
0.039047,
-0.106284,
-0.001976,
0.048379,
0.015990,
0.042910,
-0.090759,
0.005120,
0.068759,
0.035644,
0.003328,
0.029793,
0.022931,
0.032854,
0.015728,
0.042218,
0.027850,
0.020177,
-0.008348,
0.056963,
0.015328,
-0.072268,
0.037211,
0.046201,
0.017590,
0.039419,
0.043165,
0.052134,
0.066284,
0.028173,
-0.011126,
0.031666,
-0.045541,
-0.077123,
-0.019736,
0.021426,
-0.066800,
0.051262,
-0.057731,
0.021933,
-0.048773,
0.006572,
0.036924,
-0.077349,
0.032372,
0.051847,
-0.065659,
-0.084866,
-0.059190,
0.028338,
0.046083,
0.028791,
-0.062331,
0.034886,
0.046865,
-0.011726,
-0.008135,
-0.006421,
0.008170,
-0.055913,
0.029758,
-0.012234,
-0.062997,
0.032476,
0.056739,
0.026697,
-0.068554,
-0.060150,
-0.058530,
0.001998,
-0.083140,
0.093509,
0.031174,
-0.039927,
-0.006498,
0.048920,
-0.051228,
0.010380,
-0.078304,
0.010913,
0.011288,
-0.004488,
0.054312,
-0.002013,
-0.061965,
0.002404,
0.034549,
0.042918,
0.007764,
0.030480,
-0.063842,
0.022779,
-0.009424,
-0.002088,
0.056823,
0.004248,
0.070791,
0.037839,
-0.094277,
-0.073128,
0.022976,
0.019566,
0.030013,
0.004661,
0.048601,
-0.065036,
0.041614,
0.046739,
0.009676,
0.034306,
-0.042776,
0.040868,
0.016598,
0.021178,
0.020336,
-0.047107,
-0.001957,
0.040344,
0.036630,
-0.039918,
0.006285,
0.011967,
0.029821,
0.007353,
0.017333,
-0.108762,
-0.058043,
-0.080115,
0.010887,
0.031525,
0.002281,
-0.011143,
0.041170,
-0.065877,
0.017928,
-0.077523,
-0.082128,
-0.019739,
-0.074544,
0.004092,
0.060362,
0.028258,
-0.059444,
0.018333,
0.035165,
0.034899,
0.049486,
-0.058890,
-0.057295,
-0.010472,
-0.018435,
0.028908,
0.027947,
0.025302,
0.052610,
0.032297,
-0.000414,
-0.082894,
0.008614,
0.046582,
0.029814,
0.002829,
0.018926,
0.041093,
0.035587,
0.011238,
0.036060,
-0.012005,
0.036352,
0.033535,
0.064181,
0.033083,
0.030058,
0.032414,
0.031097,
-0.064749,
-0.082713,
0.027735,
-0.006560,
0.020956,
0.030085,
0.040678,
-0.077488,
0.047033,
0.066472,
-0.010980,
0.028028,
0.022878,
-0.002134,
0.056454,
0.017356,
0.002007,
0.007171,
-0.064499,
0.014689,
-0.005971,
0.040507,
0.031065,
-0.074043,
-0.057378,
0.009843,
0.042178,
0.030118,
0.067661,
-0.066967,
0.030579,
-0.016076,
-0.015489,
-0.065437,
0.047424,
-0.036523,
0.032892,
0.026650,
0.001036,
-0.069630,
0.021078,
-0.046483,
-0.032812,
0.006256,
0.044380,
-0.075756,
0.047536,
0.041946,
-0.011155,
-0.000246,
0.033780,
0.037406,
0.015544,
0.030161,
-0.074358,
]).unsqueeze(0)
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            sampRate = 16000
            # Convert tensor to numpy array
            speech = speech.numpy()
            resampled_segment =librosa.resample(speech, orig_sr=sampRate, target_sr=SAMPLE_RATE)
            combined_audio.append(resampled_segment)
        else:
            # Generate audio from text
            speech = generate_audio(chunk, history_prompt="v2/en_speaker_9")
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
    # slow_down_wav_bytes = slowdownAudio(uid, wav_bytes)
    # Slow down the audio if necessary
    return upload_to_s3(wav_bytes, uid)

@app.route("/audioEval", methods=['POST'])
def audioEval():
    audio_file = request.files['audio']
    phonetic_dict = cmudict.dict()
    text = request.form['title']
    try:
        user_phonetic, user_text = audio_to_phonetics(audio_file)
    except Exception as e:
        print(e)    
        return jsonify({"error": "Audio file not supported"})    
    score = compare_phonetics_score(user_phonetic,text)
    words = text.split()
    correct_phonetics = [phonetic_dict[word.lower()][0] for word in words if word.lower() in phonetic_dict]
    print(f"Score: {score}")
    random_number = random.randint(0, 10)
    if score + random_number <= 100:
      accuracy = score + random_number
    else:
      accuracy = score - random_number
    return {"totalScore":score, "accuracy": accuracy}

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

def highlight_word_mismatches(user_phonetics, correct_phonetics, user_text):
    mismatch_indices = find_mismatch_indices(user_phonetics, correct_phonetics)
    highlighted_text = user_text
    for index in mismatch_indices:
        highlighted_text = highlighted_text.replace(highlighted_text.split()[index], f"<mark>{highlighted_text.split()[index]}</mark>")
    return highlighted_text

def find_mismatch_indices(user_phonetics, correct_phonetics):
    mismatch_indices = []

    for index, (user_word, correct_word) in enumerate(zip(user_phonetics, correct_phonetics)):
        user_str = ' '.join(user_word)
        correct_str = ' '.join(correct_word)

        # Calculate Levenshtein distance
        distance = lev.distance(user_str, correct_str)

        # If distance is not zero, there is a mismatch
        if distance != 0:
            mismatch_indices.append(index)

    return mismatch_indices

def audio_to_phonetics(audio_file): 
    phonetic_dict = cmudict.dict()
    text = transcribe_audio(audio_file) 
    print(f"Transcribed text: {text}")
    words = text.split()
    phonetics = [phonetic_dict[word.lower()][0] for word in words if word.lower() in phonetic_dict]
    return phonetics, text

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
    # output = text_classifier(textData)
    lang_chunks[textData] = "en" # Temporary fix
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
                                              ExpiresIn=172000) # URL expires in 48 hour
    print(presigned_url)
    return presigned_url

port =8080
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    print("App running on port", port)

