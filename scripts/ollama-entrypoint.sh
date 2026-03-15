#!/bin/sh
# Start Ollama, pull a small model, then keep serving.
set -e
MODEL="${OLLAMA_MODEL:-tinyllama}"

ollama serve &
serve_pid=$!

# Wait for server to accept requests
sleep 5
for i in 1 2 3 4 5 6 7 8 9 10; do
  if ollama list >/dev/null 2>&1; then break; fi
  sleep 2
done

echo "Pulling model: $MODEL"
ollama pull "$MODEL" || true

wait $serve_pid
