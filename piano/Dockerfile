FROM python:3.13-slim

ENV PIANO_SF2_PATH=/opt/soundfont.sf2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    fluidsynth \
    libfluidsynth-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audio2midi.py .
COPY lambda_handler.py .
COPY soundfont.sf2 ${PIANO_SF2_PATH}

RUN chmod 644 ${PIANO_SF2_PATH}

CMD ["python", "-m", "awslambdaric", "lambda_handler.handler"]