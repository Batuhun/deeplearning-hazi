# Use the Graphcore PyTorch Geometric base image
FROM graphcore/pytorch-geometric:latest

# Set the working directory
WORKDIR /app

# Copy the Code.py file into the container
COPY . .

# Install any additional Python packages required
RUN pip install --no-cache-dir pandas gradio

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set the default command to run your script
CMD ["python3", "Gradio.py"]


