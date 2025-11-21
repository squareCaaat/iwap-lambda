FROM python:3.13-slim

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

COPY cnn_processor.py .
COPY resnet18_mnist.pth .
COPY lambda_handler.py .

CMD ["python", "-m", "awslambdaric", "lambda_handler.handler"]