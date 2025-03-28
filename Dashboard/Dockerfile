FROM python:3.10


ADD app.py .

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "./app.py"]