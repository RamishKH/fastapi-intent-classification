# Use official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements and application files
COPY requirements.txt ./
COPY main.py ./
COPY intent_classifier/ ./intent_classifier/ 
 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
