# Use the Graphcore PyTorch Geometric base image
FROM graphcore/pytorch-geometric:latest

# Set the working directory
WORKDIR /app

# Copy the Code.py file into the container
COPY Code.py .

# Install any additional Python packages you need (if any)
# Uncomment and add packages below if necessary
# RUN pip install --no-cache-dir <package_name>

# Set the default command to run your script
CMD ["python3", "Code.py"]
