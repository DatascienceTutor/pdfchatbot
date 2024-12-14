# Use the Python 3.10 slim image as the base
FROM python:3.10.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Upgrade pip and install the dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary application files into the container
COPY . /app/

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the entry point for the container to run the Streamlit app
CMD ["streamlit", "run", "app.py","--server.port=8501","--server.address=0.0.0.0"]


