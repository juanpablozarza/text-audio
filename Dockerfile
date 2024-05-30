# Start from a CUDA image with Python support
FROM --platform=linux/amd64 nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 

# Install Python
RUN apt-get update && apt-get install -y python3-pip python3-dev && \
    apt-get install -y python3-pip python3-dev ffmpeg espeak \
    libespeak-dev libespeak1 python3-setuptools


RUN pip3 install --upgrade pip setuptools wheel 

# Set the working directory in the container
WORKDIR /usr/src/app
# Install Python dependencies
RUN  pip3 install numpy
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt --default-timeout=500



# Copy the rest of your application's code into the container at /app
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Create empty uploads folder
RUN mkdir uploads
# Run index.py when the container launches
CMD ["python3", "index.py"]
