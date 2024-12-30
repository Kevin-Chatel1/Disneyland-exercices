# Base Image
FROM tensorflow/tensorflow:2.15.0

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --ignore-installed blinker && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Copy .env file
COPY .env .env

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose MLflow port
EXPOSE 5000

# Run the entrypoint script
CMD ["/entrypoint.sh"]

