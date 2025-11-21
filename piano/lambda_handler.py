import json
import boto3
import os
import joblib
from pathlib import Path

from audio2midi import midi_to_mp3_bytes, talking_piano

s3_client = boto3.client('s3')
RESULT_BUCKET = os.environ.get('RESULT_BUCKET')

TEMP_DIR = "/tmp"
os.environ["PIANO_MP3_PATH"] = os.path.join(TEMP_DIR, "input.mp3")
os.environ["PIANO_OUTPUT_MIDI_PATH"] = os.path.join(TEMP_DIR, "output.mid")
os.environ["PIANO_OUTPUT_MP3_PATH"] = os.path.join(TEMP_DIR, "output.mp3")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(TEMP_DIR, ".numba_cache")
os.environ.setdefault("PIANO_SF2_PATH", "/opt/soundfont.sf2")

os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

INPUT_MP3_PATH = os.environ["PIANO_MP3_PATH"]
OUTPUT_MIDI_PATH = os.environ["PIANO_OUTPUT_MIDI_PATH"]
OUTPUT_MP3_PATH = os.environ["PIANO_OUTPUT_MP3_PATH"]
SF2_PATH = Path(os.environ["PIANO_SF2_PATH"])

def handler(event, context):
    print("Piano Worker 시작...")
    
    for record in event['Records']:
        try:
            body = json.loads(record['body'])
            task_id = body['task_id']
            s3_key = body['s3_key']
            bucket = body['bucket']
            
            print(f"[{task_id}] 작업 처리 시작. S3 키: {s3_key}")

            input_path = INPUT_MP3_PATH
            s3_client.download_file(bucket, s3_key, input_path)

            print(f"[{task_id}] 오디오 파일 다운로드 완료: {input_path}")

            if not SF2_PATH.exists():
                raise FileNotFoundError(f"SoundFont 파일을 찾을 수 없습니다: {SF2_PATH}")

            with joblib.parallel_backend('threading'):
                midi_file, sample_rate = talking_piano()

            print(f"[{task_id}] MIDI 변환 완료.")

            mp3_bytes = midi_to_mp3_bytes(midi_file, SF2_PATH, sample_rate)
            with open(OUTPUT_MP3_PATH, "wb") as mp3_file:
                mp3_file.write(mp3_bytes)

            print(f"[{task_id}] MP3 변환 및 저장 완료: {OUTPUT_MP3_PATH}")

            midi_result_key = f"results/piano/{task_id}.mid"
            mp3_result_key = f"results/piano/{task_id}.mp3"
            midi_output_path = OUTPUT_MIDI_PATH
            
            s3_client.upload_file(
                midi_output_path,
                RESULT_BUCKET,
                midi_result_key,
                ExtraArgs={'ContentType': 'audio/midi'}
            )
            s3_client.upload_file(
                OUTPUT_MP3_PATH,
                RESULT_BUCKET,
                mp3_result_key,
                ExtraArgs={'ContentType': 'audio/mpeg'}
            )
            print(f"[{task_id}] 결과 S3 업로드 완료: {midi_result_key}, {mp3_result_key}")
            
            os.remove(input_path)
            os.remove(midi_output_path)
            os.remove(OUTPUT_MP3_PATH)

        except Exception as e:
            print(f"[{task_id}] 처리 중 오류 발생: {e}")
            raise e
            
    return {'status': 'success'}