import ollama


def generate_response_stream(prompt, model="llama3.2"):
    # stream response from local ollama — no API key needed
    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content

    except Exception as e:
        yield f"\n\n[Ollama error: {str(e)}]\nMake sure Ollama is running and the model is pulled (e.g. `ollama pull llama3.2`)"
