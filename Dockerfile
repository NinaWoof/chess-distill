# Dockerfile for gen_labels.py on x86
FROM python:3.11-slim

# Install system dependencies including Stockfish and build tools if needed
RUN apt-get update && apt-get install -y \
    stockfish \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .
# Create dummy src to allow pip to see the package structure
RUN mkdir -p src/chess_distill && touch src/chess_distill/__init__.py

# Install only the necessary dependencies for data generation (SKIP torch)
RUN pip install --no-cache-dir chess pandas numpy tqdm pyarrow rich

# Install the package without dependencies (we just handled them)
RUN pip install --no-cache-dir --no-deps .

# Copy project files (but NOT data)
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set environment variables
# Ensure Stockfish (in /usr/games) can be found and PYTHONPATH includes our src
ENV PATH="/usr/games:${PATH}"
ENV STOCKFISH_PATH=/usr/games/stockfish
ENV PYTHONPATH=/app/src

# Default entrypoint to run the label generator
ENTRYPOINT ["python", "scripts/gen_labels.py"]

# Default arguments: expects data to be mounted at /app/data
CMD ["--out", "data/dataset_lichess.parquet", "--max_positions", "1000000", "--pgn_dir", "data/pgns"]
