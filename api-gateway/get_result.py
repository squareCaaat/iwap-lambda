import json
import boto3
import os
import base64

s3_client = boto3.client('s3')

RESULT_BUCKET = os.environ.get('RESULT_BUCKET')

def handler(event, context):
    try:
        http_path = event['requestContext']['http']['path']

        if 'pathParameters' not in event or 'task_id' not in event['pathParameters']:
             return {'statusCode': 400, 'body': json.dumps('task_id is missing.')}

        task_id = event['pathParameters']['task_id']
        content_type = ""
        is_json_gz = False

        if http_path.startswith('/api/inside/'):
            result_key = f"results/inside/{task_id}.json.gz"
            content_type = "application/json"
            is_json_gz = True
        elif http_path.startswith('/api/piano/midi/'):
            result_key = f"results/piano/{task_id}.mid"
            content_type = "audio/midi"
        elif http_path.startswith('/api/piano/mp3/'):
            result_key = f"results/piano/{task_id}.mp3"
            content_type = "audio/mpeg"
        elif http_path.startswith('/api/string/array/'):
            result_key = f"results/string/{task_id}.json.gz"
            content_type = "application/json"
            is_json_gz = True
        elif http_path.startswith('/api/string/image/'):
            result_key = f"results/string/{task_id}.png"
            content_type = "image/png"
        else:
            return {'statusCode': 404, 'body': json.dumps('Not Found')}

        try:
            file_obj = s3_client.get_object(Bucket=RESULT_BUCKET, Key=result_key)
            file_content = file_obj['Body'].read()

            headers = {
                'Content-Type': content_type
            }
            if is_json_gz:
                headers['Content-Encoding'] = 'gzip'

            return {
                'statusCode': 200,
                'body': base64.b64encode(file_content).decode('utf-8'),
                'isBase64Encoded': True,
                'headers': headers
            }

        except s3_client.exceptions.NoSuchKey:
            return {
                'statusCode': 202,
                'body': json.dumps({'status': 'PENDING', 'task_id': task_id}),
                'headers': {
                    'Content-Type': 'application/json'
                }
            }

    except Exception as e:
        print(f"Error getting result: {e}")
        return {'statusCode': 500, 'body': json.dumps('Internal Server Error')}