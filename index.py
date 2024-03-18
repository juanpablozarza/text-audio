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
import pyrubberband as pyrb
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface



mysp= __import__("my-voice-analysis")
# tts = CS_API()
app = Flask(__name__)
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
model_size = "large-v2"
# Run on GPU with FP16
# whisper = WhisperModel(model_size, device=device, compute_type="float16")
# Load processor and model when server starts
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)
models_TTS, cfg, task_TTS = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False, "device": device},
)
model_TTS = models_TTS[0].to(device)
TTSHubInterface.update_cfg_with_data_cfg(cfg, task_TTS.data_cfg)
generator = task_TTS.build_generator([model_TTS], cfg)

# Load the whisper model
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
whisper_model_id = "Juanpablozarza292/whisper-destil-spa-eng"
whisper_tokenizer_id = "distil-whisper/distil-large-v2"

whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype= torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

whisper.to(device)

processor_whisper = AutoProcessor.from_pretrained(whisper_tokenizer_id)
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
    print(file.filename.split('.'))
    ext = os.path.splitext(file.filename)[1]
    print(ext)
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join("./uploads", secure_filename(filename) )
    # Save the original file
    file.save(file_path)
    # Check if the file is in CAF format and convert to WAV if necessary
    if ext.lower() == '.caf':
        print('Converting CAF to WAV', file_path)
        audio = AudioSegment.from_file(file_path, format='caf')
        wav_path = file_path.replace('.caf', '.wav')
        audio.export(wav_path, format='wav')
        # Update file_path to the new WAV file
        file_path = wav_path
    # Process the file with Whisper
    try:
      result = whisper_pipe(file_path)
    except Exception as e:
        print(e)
        return jsonify({"error": "Audio file not supported"})
    
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
    stretch_factor = 1.1
    # Load the audio file
    audio = AudioSegment.from_file(io.BytesIO(audio_array), format="wav")
    # Slow down the audio to half its speed
    processed_audio = pyrb.time_stretch(audio_array, 16000, stretch_factor)
    # Save the processed audio
    # Save the slowed audio
    slowed_audio_bytes = io.BytesIO()
    processed_audio.export(slowed_audio_bytes, format="wav")
    slowed_audio_bytes = slowed_audio_bytes.getvalue()
    return processed_audio
  
  # Function to create VTT file from transcribed text and audio file 

def create_vtt_from_audio_bytes(wav_bytes, sample_rate, transcript, output_vtt_path, lessonRef):
    """
    Create a VTT file from audio bytes and its transcript using Aeneas.

    :param wav_bytes: The audio data as bytes.
    :param sample_rate: The sample rate of the audio data.
    :param transcript: The transcript text.
    :param output_vtt_path: Path where the output VTT file will be saved.
    """
    # Create temporary files for the audio and transcript
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_temp, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as transcript_temp:
        
        # Write the audio bytes to the temporary audio file
        write(audio_temp.name, sample_rate, wav_bytes)
        
        # Write the transcript to the temporary transcript file
        transcript_temp.write(transcript.encode())
        transcript_temp.flush()  # Ensure all data is written to disk

        # Command template for Aeneas
        command_template = [
            'python', '-m', 'aeneas.tools.execute_task',
            audio_temp.name,
            transcript_temp.name,
            "task_language=eng|is_text_type=plain|os_task_file_format=vtt",
            output_vtt_path
        ]

        # Execute the Aeneas command
        try:
            subprocess.run(command_template, check=True)
            print("VTT file created successfully:", output_vtt_path)
            # Upload file to s3 and link to firebase firestore in a transcript property
            s3 = boto3.client('s3')
            s3.upload_file(output_vtt_path, "audios-edgen", output_vtt_path)
            # Generate Pre-signed URL
            presigned_url = s3.generate_presigned_url('get_object',
                                                Params={'Bucket': "audios-edgen",
                                                        'Key': output_vtt_path},
                                                ExpiresIn=17200000) # URL expires in 48 hour
            print(presigned_url)
            if lessonRef:
                doc_ref = db.collection("lessons").document(lessonRef)
                doc_ref.update({
                    "transcript": firestore.ArrayUnion([presigned_url])
                })
        except subprocess.CalledProcessError as e:
            print("An error occurred while creating the VTT file:", str(e))
        finally:
            # Clean up temporary files
            os.remove(audio_temp.name)
            os.remove(transcript_temp.name)

def checkSelectedClass(uid):
    doc_ref = db.collection("users").document(uid) 
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()["selectedClass"]
            
@app.route("/generateAudioFile/<uid>", methods=["POST"])
def generateAudioFile(uid):
    print('Generating audio file...')
    reqData = request.json
    textData = reqData.get("textData")
    selected_class = checkSelectedClass(uid)
    langs = textClassifier(textData)
    print(f"Langs: {langs}")
    combined_audio = [] 
    for chunk in langs: 
        print(f"Chunk: {chunk}")
        spanish_sim_langs = ["it","fr", "pt","ro", "es"] 
        if not langs[chunk] in spanish_sim_langs:
            # Generate audio from text
            sample = TTSHubInterface.get_model_input(task_TTS, chunk)
            sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to("cuda")
            speech, sampRate = TTSHubInterface.get_prediction(task_TTS, model_TTS,  generator, sample)
            # Convert tensor to numpy array
            speech = speech.cpu().numpy()
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
    byte_io = io.BytesIO()
    write(byte_io, SAMPLE_RATE, resampled_speech)
    wav_bytes = byte_io.read()
    byte_io.seek(0)
    # Create a VTT file from the audio and its transcript
    create_vtt_from_audio_bytes(wav_bytes, SAMPLE_RATE, textData, f"results/{uid}.vtt",selected_class)
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
                                              ExpiresIn=1720000) # URL expires in 48 hour
    print(presigned_url)
    return presigned_url

port =8080
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    print("App running on port", port)

