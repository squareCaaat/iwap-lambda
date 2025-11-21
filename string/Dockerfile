FROM python:3.13-slim

ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p ${MPLCONFIGDIR}

WORKDIR /var/task

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY string_generator.py .
COPY lambda_handler.py .

CMD ["python", "-m", "awslambdaric", "lambda_handler.handler"]