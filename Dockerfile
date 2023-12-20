# Start from a CUDA image with Python support
FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y python3-pip python3-dev
 # Upgrade setuptools and wheel
RUN pip3 install --upgrade pip setuptools wheel

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080


# Run index.py when the container launches
CMD ["python3", "index.py"]
