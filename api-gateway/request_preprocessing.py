import json
import boto3
import base64
import os
import uuid
import filetype
from io import BytesIO
from pydub import AudioSegment

os.environ["FFMPEG_PATH"] = "/opt/ffmpeg/ffmpeg"
os.environ["FFPROBE_PATH"] = "/opt/ffmpeg/ffprobe"

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

UPLOAD_BUCKET = os.environ.get('UPLOAD_BUCKET')
INSIDE_QUEUE_URL = os.environ.get('INSIDE_QUEUE_URL')
PIANO_QUEUE_URL = os.environ.get('PIANO_QUEUE_URL')
STRING_QUEUE_URL = os.environ.get('STRING_QUEUE_URL')

def parse_multipart_form_data(body, content_type):
    try:
        boundary = None
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part.split('=', 1)[1].strip('"')
                break
        
        if not boundary:
            return None, {}
        
        boundary_bytes = f'--{boundary}'.encode()
        parts = body.split(boundary_bytes)
        file_bytes = None
        fields = {}
        
        for part in parts:
            if not part or part in (b'--\r\n', b'--'):
                continue
            
            if b'\r\n\r\n' in part:
                header_section, data = part.split(b'\r\n\r\n', 1)
                data = data.rstrip(b'\r\n')
                
                header_lines = header_section.decode('utf-8', errors='ignore').split('\r\n')
                disposition = next((line for line in header_lines if line.lower().startswith('content-disposition')), None)
                if disposition is None:
                    continue
                
                name = None
                filename = None
                for item in disposition.split(';'):
                    item = item.strip()
                    if item.startswith('name='):
                        name = item.split('=', 1)[1].strip('"')
                    elif item.startswith('filename='):
                        filename = item.split('=', 1)[1].strip('"')
                
                if filename:
                    if name == 'file' or file_bytes is None:
                        file_bytes = data
                elif name:
                    fields[name] = data.decode('utf-8', errors='ignore')
        
        return file_bytes, fields
    except Exception as e:
        print(f"Multipart parsing error: {e}")
        return None, {}

def detect_image_format(file_content):
    kind = filetype.guess(file_content)
    if kind is None:
        return None
    
    if kind.mime.startswith('image/'):
        return kind.extension
    return None

def convert_audio_to_mp3(file_content, source_format):
    audio = AudioSegment.from_file(BytesIO(file_content), format=source_format)
    output_buffer = BytesIO()
    audio.export(output_buffer, format='mp3', bitrate='192k')
    output_buffer.seek(0)
    return output_buffer.read()

def detect_audio_format(file_content):
    kind = filetype.guess(file_content)
    if kind is None:
        return None
    
    if kind.mime.startswith('audio/'):
        return kind.extension
    return None

def parse_boolean_text(value):
    lowered = value.strip().lower()
    if lowered in ('true', '1', 'yes', 'y', 'on'):
        return True
    if lowered in ('false', '0', 'no', 'n', 'off'):
        return False
    raise ValueError("Boolean 필드는 true/false 중 하나여야 합니다.")

def handler(event, context):
    try:
        http_path = event['requestContext']['http']['path']
        headers = event.get('headers', {})

        content_type = None
        for key, value in headers.items():
            if key.lower() == 'content-type':
                content_type = value
                break

        file_content = None
        form_fields = {}

        if content_type and 'multipart/form-data' in content_type:
            if event.get('isBase64Encoded', False):
                body = base64.b64decode(event['body'])
            else:
                body = event['body'].encode() if isinstance(event['body'], str) else event['body']

            file_content, form_fields = parse_multipart_form_data(body, content_type)

            if file_content is None:
                return {
                    'statusCode': 400,
                    'body': json.dumps('Multipart Form Data 파싱에 실패했거나 파일을 찾을 수 없습니다.'),
                    'headers': {'Content-Type': 'application/json'}
                }
        elif event.get('isBase64Encoded', False):
            file_content = base64.b64decode(event['body'])
        else:
            return {
                'statusCode': 400,
                'body': json.dumps('파일은 Base64로 인코딩되거나 multipart/form-data 형식이어야 합니다.'),
                'headers': {'Content-Type': 'application/json'}
            }

        if not file_content:
            return {
                'statusCode': 400,
                'body': json.dumps('파일이 없습니다.'),
                'headers': {'Content-Type': 'application/json'}
            }

        task_id = str(uuid.uuid4())
        upload_content = None
        queue_url = None
        s3_key = None
        extra_sqs_payload = {}

        if http_path == '/api/inside':
            img_format = detect_image_format(file_content)
            
            allowed_formats = ['png', 'jpg', 'jpeg']
            if img_format not in allowed_formats:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': f'지원되지 않는 이미지 파일입니다. PNG, JPG, JPEG 파일만 허용됩니다. 보낸 파일의 확장자: {img_format}'
                    }),
                    'headers': {'Content-Type': 'application/json'}
                }
            
            if img_format == 'jpeg':
                img_format = 'jpg'
            
            file_extension = f'.{img_format}'
            s3_key = f"uploads/inside/{task_id}{file_extension}"
            queue_url = INSIDE_QUEUE_URL
            
            if img_format == 'png':
                content_type = 'image/png'
            else:
                content_type = 'image/jpeg'
            
            upload_content = file_content

        elif http_path == '/api/piano':
            audio_format = detect_audio_format(file_content)

            allowed_formats = ['webm', 'wav', 'mp3']            
            if audio_format not in allowed_formats:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': f'지원되지 않는 오디오 파일입니다. WebM, WAV, MP3 파일만 허용됩니다. 보낸 파일의 확장자: {audio_format}'
                    }),
                    'headers': {'Content-Type': 'application/json'}
                }
            
            if audio_format in ['webm', 'wav']:
                try:
                    upload_content = convert_audio_to_mp3(file_content, audio_format)
                except Exception as e:
                    print(f"Error converting audio: {e}")
                    return {
                        'statusCode': 500,
                        'body': json.dumps({
                            'error': f'MP3 파일 변환에 실패했습니다: {str(e)}'
                        }),
                        'headers': {'Content-Type': 'application/json'}
                    }
            else:
                upload_content = file_content
            
            file_extension = '.mp3'
            s3_key = f"uploads/piano/{task_id}{file_extension}"
            queue_url = PIANO_QUEUE_URL
            content_type = 'audio/mpeg'

        elif http_path == '/api/string':
            allowed_formats = ['png', 'jpg', 'jpeg']
            img_format = detect_image_format(file_content)

            if img_format not in allowed_formats:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': f'지원되지 않는 이미지 파일입니다. PNG, JPG, JPEG 파일만 허용됩니다. 보낸 파일의 확장자: {img_format}'
                    }),
                    'headers': {'Content-Type': 'application/json'}
                }

            if img_format == 'jpeg':
                img_format = 'jpg'

            file_extension = f'.{img_format}'
            s3_key = f"uploads/string/{task_id}{file_extension}"
            queue_url = STRING_QUEUE_URL
            content_type = 'image/png' if img_format == 'png' else 'image/jpeg'
            upload_content = file_content

            def get_field_value(field_name):
                value = form_fields.get(field_name)
                if value is None:
                    return None
                value = value.strip()
                if not value:
                    return None
                return value

            def parse_numeric(field_name, cast=float, default=None):
                raw_value = get_field_value(field_name)
                if raw_value is None:
                    return default
                try:
                    return cast(raw_value)
                except ValueError:
                    raise ValueError(f"{field_name} 값이 올바르지 않습니다.")

            def parse_boolean_field(field_name, default=False):
                raw_value = get_field_value(field_name)
                if raw_value is None:
                    return default
                try:
                    return parse_boolean_text(raw_value)
                except ValueError as exc:
                    raise ValueError(f"{field_name}: {str(exc)}")

            try:
                random_nails = parse_numeric('random_nails', int, default=50)
                limit_value = parse_numeric('limit', int, default=5000)
                nail_step = parse_numeric('nail_step', int, default=4)
                strength = parse_numeric('strength', float, default=0.1)
                rgb_value = parse_boolean_field('rgb', default=False)
                wb_value = parse_boolean_field('wb', default=False)
            except ValueError as exc:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': str(exc)}),
                    'headers': {'Content-Type': 'application/json'}
                }

            extra_sqs_payload = {
                "random_nails": random_nails,
                "limit": limit_value,
                "rgb": rgb_value,
                "wb": wb_value,
                "nail_step": nail_step,
                "strength": strength,
            }

        else:
            return {
                'statusCode': 404,
                'body': json.dumps('Not Found'),
                'headers': {'Content-Type': 'application/json'}
            }

        s3_client.put_object(
            Bucket=UPLOAD_BUCKET,
            Key=s3_key,
            Body=upload_content,
            ContentType=content_type
        )

        message_body = {
            "task_id": task_id,
            "s3_key": s3_key,
            "bucket": UPLOAD_BUCKET
        }
        message_body.update(extra_sqs_payload)

        sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message_body)
        )

        return {
            'statusCode': 202,
            'body': json.dumps({'task_id': task_id}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

    except Exception as e:
        print(f"Error processing request: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps('Internal Server Error'),
            'headers': {'Content-Type': 'application/json'}
        }