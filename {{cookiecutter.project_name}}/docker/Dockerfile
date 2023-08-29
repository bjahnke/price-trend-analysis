# Use the official Python 3.11 image as the base
FROM python:3.11

# Expose port 8080 for google cloud run compatibility
EXPOSE 8080

# Set the working directory to /app
WORKDIR /app

# Update libpq package
RUN apt-get update && apt-get install -y libpq-dev

# Copy the rest of the application code to the working directory
COPY requirements.txt ./

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app

# Add the application's directory to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set the command to run when the container starts
CMD [ "python3", "src/{{cookiecutter.package_name}}/main.py" ]
