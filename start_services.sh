#!/bin/sh

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
sleep 5

ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1b