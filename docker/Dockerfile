# Use the official Python 3.11 image as the base
FROM python:3.11

# Expose port 8080 for google cloud run compatibility
EXPOSE 8080

# Set the working directory to /app
WORKDIR /app

# Update libpq package
RUN apt-get update && apt-get install -y libpq-dev

# Copy the rest of the application code to the working directory
COPY requirements-docker.txt ./

# make .ssh/known_hosts
RUN mkdir -p ~/.ssh && touch ~/.ssh/known_hosts
RUN ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install the dependencies
RUN --mount=type=ssh,id=github_ssh_key pip install git+ssh://git@github.com/bjahnke/blue_bird_pms.git
RUN pip install -r requirements-docker.txt
# Copy the rest of the application code to the working directory
COPY . /app

# Add the application's directory to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set the command to run when the container starts
CMD [ "python3", "src/trend_analysis/main.py" ]
