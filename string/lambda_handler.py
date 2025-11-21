import gzip
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import numpy as np
from PIL import Image

from string_generator import StringArtOptions, StringArtResult, generate_string_art_from_array

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

s3_client = boto3.client('s3')

RESULT_BUCKET = os.environ.get('RESULT_BUCKET')
TEMP_DIR = Path("/tmp")
RESULT_IMAGE_EXTENSION = os.environ.get("STRING_RESULT_IMAGE_EXTENSION", ".png")


def _ensure_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


def _safe_int(value: Optional[Any], default: int = 0) -> Optional[int]:
    if value is None:
        return default
    return int(value)


def _safe_float(value: Optional[Any], default: float) -> float:
    if value is None:
        return default
    return float(value)


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        image = img.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def _save_result_image(image_array: np.ndarray, output_path: Path) -> str:
    clipped = np.clip(image_array, 0.0, 1.0)
    if clipped.ndim == 2:
        pil_image = Image.fromarray((clipped * 255).astype(np.uint8), mode="L")
        content_type = "image/png"
    elif clipped.ndim == 3 and clipped.shape[2] == 3:
        pil_image = Image.fromarray((clipped * 255).astype(np.uint8), mode="RGB")
        content_type = "image/png"
    else:
        raise ValueError("지원되지 않는 이미지 형식입니다.")

    pil_image.save(output_path, format="PNG")
    return content_type


def _build_options(payload: Dict[str, Any]) -> StringArtOptions:
    random_nails = _safe_int(payload.get("random_nails"), 50)
    strength = _safe_float(payload.get("strength"), 0.1)
    nail_step = _safe_int(payload.get("nail_step"), 4)
    nail_step = max(1, nail_step)
    pull_amount = _safe_int(payload.get("limit"), 5000)
    rgb = _ensure_bool(payload.get("rgb"), False)
    wb = _ensure_bool(payload.get("wb"), False)

    return StringArtOptions(
        export_strength=strength,
        pull_amount=pull_amount,
        nail_step=nail_step,
        wb=wb,
        rgb=rgb,
        random_nails=random_nails,
    )


def _build_metadata(
    task_id: str,
    result: StringArtResult
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "mode": result.mode,
        "options": asdict(result.options),
        "nail_count": len(result.nails),
        "scaled_nail_count": len(result.scaled_nails),
        "pull_lengths": [len(order) for order in result.pull_orders],
        "nails": result.nails,
        "scaled_nails": result.scaled_nails,
        "pull_orders": result.pull_orders,
    }


def handler(event, context):
    if not RESULT_BUCKET:
        raise RuntimeError("RESULT_BUCKET 환경 변수가 설정되어 있지 않습니다.")

    print("String Worker 시작...")

    for record in event['Records']:
        task_id = "unknown"
        input_path = None
        output_image_path = None
        try:
            body = json.loads(record['body'])
            task_id = body['task_id']
            s3_key = body['s3_key']
            bucket = body['bucket']

            print(f"[{task_id}] 작업 처리 시작. S3 키: {s3_key}")

            suffix = Path(s3_key).suffix or ".img"
            input_path = TEMP_DIR / f"{task_id}_input{suffix}"
            s3_client.download_file(bucket, s3_key, str(input_path))
            print(f"[{task_id}] 입력 이미지 다운로드 완료: {input_path}")

            source_image = _load_image(input_path)
            options = _build_options(body)
            result = generate_string_art_from_array(source_image, options)
            print(f"[{task_id}] String Art 생성 완료.")

            result_ext = RESULT_IMAGE_EXTENSION if RESULT_IMAGE_EXTENSION.startswith(".") else f".{RESULT_IMAGE_EXTENSION}"
            output_image_path = TEMP_DIR / f"{task_id}_result{result_ext}"
            image_content_type = _save_result_image(result.image, output_image_path)

            image_key = f"results/string/{task_id}{result_ext}"
            json_key = f"results/string/{task_id}.json.gz"

            metadata = _build_metadata(task_id, result)
            metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
            compressed_metadata = gzip.compress(metadata_bytes)

            s3_client.upload_file(
                str(output_image_path),
                RESULT_BUCKET,
                image_key,
                ExtraArgs={'ContentType': image_content_type},
            )
            print(f"[{task_id}] 결과 이미지 업로드 완료: {image_key}")

            s3_client.put_object(
                Bucket=RESULT_BUCKET,
                Key=json_key,
                Body=compressed_metadata,
                ContentType='application/gzip',
                ContentEncoding='gzip',
            )
            print(f"[{task_id}] 메타데이터 업로드 완료: {json_key}")

        except Exception as e:
            print(f"[{task_id}] 처리 중 오류 발생: {e}")
            raise e
        finally:
            for path in (input_path, output_image_path):
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except OSError:
                        pass

    return {'status': 'success'}




