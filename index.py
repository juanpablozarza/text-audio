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
            speaker_embeddings = torch.tensor([-0.088287,0.016144,0.055017,0.029052,0.015096,0.014972,-0.067029,0.007009,0.033722,0.014678,-0.083072,-0.064236,0.047448,0.024139,0.052235,0.066601,0.031087,0.035723,0.006487,0.036682,0.025053,0.035013,-0.012637,-0.062425,-0.072185,-0.004461,-0.070950,0.022124,0.051367,0.042662,-0.002556,0.043912,0.028532,0.009907,0.033805,-0.071517,0.048401,0.019725,0.016676,-0.035781,0.034850,0.004887,0.031422,0.039423,0.020462,-0.072256,-0.014559,0.009038,-0.094089,0.053852,0.039721,0.040031,0.035306,0.042026,-0.083358,-0.003777,0.022852,0.027406,0.020202,0.035050,0.038536,-0.005857,-0.009354,0.033559,0.036002,0.046980,0.027620,-0.056997,
-0.055340,
-0.061274,
0.025169,
0.007439,
0.013477,
0.010445,
0.047740,
0.038641,
0.020623,
0.010106,
-0.060837,
-0.063589,
-0.071845,
-0.073924,
-0.062130,
-0.056032,
-0.021115,
-0.071226,
-0.076171,
0.074259,
0.014346,
-0.009105,
0.023388,
-0.034161,
0.014896,
-0.071031,
0.039456,
-0.040102,
0.054891,
0.033890,
-0.037332,
-0.079685,
0.028541,
-0.102200,
-0.062983,
0.039721,
0.028951,
-0.010553,
0.038840,
0.054151,
0.038804,
0.027573,
-0.077371,
0.048018,
0.059192,
0.028804,
0.035476,
0.056604,
-0.066005,
0.024404,
-0.067709,
0.004110,
0.031299,
-0.072559,
0.032258,
0.040822,
-0.027948,
0.034767,
-0.076688,
0.029099,
0.023901,
0.037045,
0.039570,
0.054408,
0.013346,
0.058780,
0.024654,
-0.077034,
-0.085250,
0.022883,
-0.058429,
-0.016147,
0.036793,
0.003444,
-0.010465,
0.025019,
-0.075915,
0.018534,
0.014811,
-0.009606,
0.046593,
-0.065064,
0.052121,
-0.057851,
-0.086836,
0.028745,
-0.010404,
0.045070,
0.027723,
-0.081754,
-0.075855,
0.041089,
0.030345,
0.015678,
0.038335,
0.000332,
0.000402,
0.000954,
0.039610,
0.035181,
0.025261,
0.019069,
0.032303,
-0.091266,
0.036872,
-0.060215,
0.032625,
0.021165,
-0.066171,
0.025711,
0.025046,
-0.071811,
0.039156,
-0.067916,
0.002599,
0.015426,
-0.075774,
-0.007182,
0.035408,
0.038760,
-0.082825,
-0.057404,
-0.005979,
0.022143,
-0.078870,
0.020917,
0.040026,
0.009472,
0.019206,
0.035443,
0.029405,
-0.009043,
0.035260,
0.030114,
-0.060747,
0.009758,
0.047281,
0.009164,
0.056625,
-0.004978,
0.031204,
0.025464,
0.000042,
-0.057881,
-0.051849,
0.029845,
-0.069586,
0.049027,
0.030588,
-0.067528,
0.031613,
0.038798,
-0.065989,
0.014370,
-0.088788,
-0.012519,
0.006691,
0.025803,
0.028612,
0.029476,
0.028374,
0.034615,
-0.008825,
0.028955,
0.012045,
0.010364,
0.023362,
-0.018620,
0.031406,
0.016216,
0.001430,
0.046443,
-0.053742,
0.010682,
-0.075096,
0.049533,
-0.071492,
0.067993,
0.024699,
0.035479,
0.047462,
0.018482,
0.036791,
0.032047,
0.006187,
-0.060643,
0.017443,
0.035070,
0.011734,
0.038585,
-0.056928,
0.044165,
0.016905,
-0.091719,
0.037465,
0.028788,
0.000432,
0.043340,
0.047912,
0.036859,
0.016481,
-0.007326,
0.049403,
0.007439,
-0.003642,
0.003550,
-0.074124,
0.028760,
0.060935,
0.024574,
-0.069203,
-0.064837,
-0.020689,
0.049495,
0.038273,
-0.102581,
0.005830,
0.035655,
0.023300,
0.044155,
-0.074393,
0.004921,
0.071293,
0.036860,
0.006881,
0.023908,
0.019008,
0.035230,
0.020589,
0.037618,
0.029842,
0.028297,
-0.020817,
0.055824,
0.013623,
-0.081004,
0.029003,
0.042167,
0.030379,
0.035150,
0.037580,
0.043022,
0.055784,
0.025979,
-0.000414,
0.041163,
-0.056951,
-0.076608,
-0.018316,
0.031241,
-0.059809,
0.052751,
-0.054711,
0.016273,
-0.050767,
0.005092,
0.036007,
-0.080036,
0.034350,
0.049061,
-0.069400,
-0.083481,
-0.062745,
0.037292,
0.047902,
0.040896,
-0.063153,
0.041449,
0.049373,
-0.006544,
-0.003670,
0.006621,
0.017216,
-0.055989,
0.031677,
-0.003885,
-0.076899,
0.034238,
0.047793,
0.030634,
-0.059502,
-0.061975,
-0.055939,
0.012937,
-0.080907,
0.085190,
0.031878,
-0.038853,
-0.006286,
0.042904,
-0.032754,
-0.000880,
-0.079201,
0.012916,
0.007372,
0.002395,
0.061537,
0.000538,
-0.062449,
0.012733,
0.030440,
0.033610,
0.007369,
0.032041,
-0.056932,
0.017649,
-0.008493,
-0.002129,
0.048308,
0.007425,
0.064997,
0.044898,
-0.095166,
-0.064032,
0.021597,
0.009176,
0.033548,
0.013464,
0.033886,
-0.077168,
0.030573,
0.036602,
0.011349,
0.033669,
-0.050668,
0.045133,
0.029045,
0.023507,
0.030005,
-0.056260,
0.010459,
0.036465,
0.031523,
-0.041892,
0.032468,
0.009624,
0.029663,
0.005889,
0.017169,
-0.107837,
-0.066510,
-0.083962,
-0.000193,
0.021483,
0.016133,
-0.011260,
0.050079,
-0.068386,
0.015325,
-0.068810,
-0.088919,
-0.025820,
-0.072108,
-0.000098,
0.058977,
0.032032,
-0.065119,
0.025288,
0.032699,
0.045878,
0.054742,
-0.059827,
-0.059678,
-0.009581,
-0.017324,
0.019758,
0.029431,
0.020741,
0.057087,
0.035437,
0.003592,
-0.088371,
-0.000061,
0.044916,
0.034688,
0.002476,
0.022021,
0.041552,
0.028496,
0.009061,
0.023878,
-0.009081,
0.040818,
0.039326,
0.062566,
0.028075,
0.042914,
0.024550,
0.033450,
-0.066751,
-0.087004,
0.026133,
-0.006541,
0.026574,
0.033504,
0.031684,
-0.081280,
0.042178,
0.049007,
-0.016614,
0.034214,
0.019825,
-0.002177,0.051945,0.018902,0.002756,-0.005593,-0.075526,0.010933,-0.005145,0.037925,0.033890,
-0.061983,
-0.055316,
0.020500,
0.044350,
0.022207,
0.069821,
-0.060402,
0.043424,
-0.022445,
-0.013714,
-0.060104,
0.043301,-0.037969,0.028829,0.012303,0.013794,-0.071164,0.034464,-0.051553,-0.051208,
0.005376,0.046834,-0.075016,0.046078,
0.028854,
-0.009876,
0.012000,
0.046037,
0.055601,
0.007218,
0.038421,
-0.070855]).unsqueeze(0)
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

