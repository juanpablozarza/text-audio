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
            speaker_embeddings = torch.tensor([[6.529872894287109, 8.969184875488281, 3.9881420135498047, -18.17928695678711, 16.051790237426758, -30.009857177734375, 34.72180938720703, -11.482611656188965, -21.882543563842773, -36.0731201171875, -7.46410608291626, -32.55157470703125, -19.98310089111328, 12.183157920837402, -27.26100730895996, -0.10619379580020905, 30.776927947998047, 29.839094161987305, -10.855042457580566, -8.108123779296875, -8.590596199035645, -19.30548095703125, -4.689755439758301, 6.099639892578125, 8.945976257324219, 2.4291489124298096, 0.9900116324424744, -35.417579650878906, -27.79094123840332, 25.689258575439453, 8.259321212768555, -6.682685852050781, -7.374863624572754, 9.593156814575195, -6.783708095550537, -33.47209548950195, -22.06951141357422, -12.611262321472168, 4.61800479888916, 37.195556640625, 3.2137792110443115, -9.5857515335083, 16.970844268798828, 16.411731719970703, 1.6876224279403687, -16.720462799072266, 39.88679504394531, -6.353012561798096, 11.36977481842041, -33.988014221191406, -16.732542037963867, -8.5057373046875, 9.27576732635498, -15.762871742248535, 1.0053437948226929, 5.008976459503174, 5.990904808044434, 9.543562889099121, 7.511164665222168, -0.5805424451828003, 25.02376365661621, -14.364334106445312, -5.192376136779785, 8.645689010620117, -8.774984359741211, 19.815032958984375, 9.965303421020508, 3.825721025466919, 11.882308959960938, 2.2651498317718506, 8.539463996887207, -12.11210823059082, -8.17347240447998, -19.436012268066406, 21.275068283081055, -13.746713638305664, -22.103572845458984, -21.0662899017334, 1.958804965019226, 9.284294128417969, 17.059030532836914, -35.64311599731445, 25.321582794189453, -24.557058334350586, -0.6025821566581726, -11.252365112304688, -16.54556655883789, -20.4152774810791, -3.2660770416259766, 30.43497657775879, -16.113903045654297, -20.98823356628418, -15.0850191116333, -2.3913681507110596, -31.185060501098633, 16.070560455322266, -6.345791339874268, 0.5395580530166626, 11.420646667480469, 23.35231590270996, 21.85788917541504, 9.498308181762695, 26.829692840576172, -4.248407363891602, 8.080362319946289, -45.70131301879883, -1.6628291606903076, -7.326174736022949, 4.496189117431641, -14.568927764892578, 5.747090816497803, -13.473017692565918, 4.729023456573486, 1.6750907897949219, -4.191401481628418, -4.988072395324707, -10.614791870117188, -37.4455680847168, -7.570645809173584, -1.1391016244888306, 1.8883416652679443, 13.650063514709473, 15.710457801818848, 22.95478630065918, 6.215841770172119, 14.930456161499023, -3.6906015872955322, 5.962929725646973, 16.268468856811523, 4.343371391296387, 10.316946029663086, -15.786895751953125, -1.0455032587051392, -9.140889167785645, -5.938662528991699, -0.9276109337806702, -8.953985214233398, 17.087203979492188, 30.723743438720703, 0.6596390008926392, -5.299381256103516, 24.060794830322266, -7.688127040863037, -1.4049224853515625, -3.231790542602539, -10.527905464172363, 34.98810577392578, -24.386842727661133, -19.55951690673828, -9.393773078918457, -3.5240211486816406, 17.29764747619629, 18.46356201171875, 8.759380340576172, -3.0701088905334473, 39.08684539794922, 18.5070743560791, -5.827036380767822, 22.739404678344727, -28.229902267456055, 25.611072540283203, -3.600785493850708, -10.506696701049805, -18.417322158813477, 9.223679542541504, -9.114147186279297, -11.771955490112305, -14.73306941986084, -38.323490142822266, 33.44890594482422, 32.07219696044922, 3.318779945373535, 4.65116024017334, -16.474853515625, -1.9262478351593018, 7.636687278747559, 3.7501065731048584, -14.066248893737793, -10.845399856567383, -8.820802688598633, -6.369320392608643, 4.713413238525391, -31.006587982177734, -2.8532750606536865, -39.022342681884766, 17.483776092529297, -0.6185234189033508, 43.984981536865234, 20.34269142150879, -19.874340057373047, 6.0034990310668945, 0.8562405705451965]]).unsqueeze(0)
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

