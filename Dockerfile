# 1. Base Image
FROM python:3.10-slim

# 2. Work Directory
WORKDIR /app

# 3. Copy Requirements
COPY requirements.txt .

# 4. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Project Files
COPY . .

# 6. Run Script
CMD ["python", "main.py"]
