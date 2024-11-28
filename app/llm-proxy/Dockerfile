FROM debian:bookworm-slim as base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./pyproject.toml /app
COPY ./uv.lock /app

# Install any needed packages specified in pyproject.toml
RUN uv venv && \
    uv sync \
    --no-dev \
    --frozen


# Use the official Python image as a parent image
FROM base as llm-proxy-api

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN uv sync \
    --no-dev \
    --frozen

# Make the specified port available to the world outside this container
EXPOSE 18000

# Run app.py when the container launches
CMD ["uv", "run", "python", "src/main.py", "--host", "0.0.0.0", "--port", "18000"]

FROM llm-proxy-api as dev

RUN apt update && apt install -y \
    curl \
    && apt clean
