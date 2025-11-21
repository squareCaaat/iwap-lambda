import json
import boto3
import os
import io
from PIL import Image
import gzip
from cnn_processor import processor

s3_client = boto3.client('s3')

RESULT_BUCKET = os.environ.get('RESULT_BUCKET')

def handler(event, context):
    print("Inside Worker 시작...")
    
    for record in event['Records']:
        try:
            body = json.loads(record['body'])
            task_id = body['task_id']
            s3_key = body['s3_key']
            bucket = body['bucket']
            
            print(f"[{task_id}] 작업 처리 시작. S3 키: {s3_key}")

            obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
            image_bytes = obj['Body'].read()
            
            pil_image = Image.open(io.BytesIO(image_bytes))

            result_json = processor.get_normalized_outputs(pil_image)
            
            if result_json is None:
                raise Exception("모델 결과가 None입니다.")

            print(f"[{task_id}] 모델 추론 완료 > GZip 압축")

            json_data = json.dumps(result_json)
            compressed_json = gzip.compress(json_data.encode('utf-8'))

            result_key = f"results/inside/{task_id}.json.gz"
            s3_client.put_object(
                Bucket=RESULT_BUCKET,
                Key=result_key,
                Body=compressed_json,
                ContentType='application/gzip'
            )
            print(f"[{task_id}] 결과 S3 업로드 완료: {result_key}")

        except Exception as e:
            print(f"[{task_id}] 처리 중 오류 발생: {e}")
            raise e

    return {'status': 'success'}