# Base Py image 
FROM python:3.12-slim

# Set working directory
WORKDIR /morse_decoder

# Copy files to the container
COPY . . 

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 5001

# launching FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]