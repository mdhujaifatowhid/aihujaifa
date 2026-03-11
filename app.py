import os
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
from persona import SYSTEM_PROMPT, PERSONA_NAME
import json

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


@app.route("/")
def index():
    return render_template("index.html", persona_name=PERSONA_NAME)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Build the full message history with system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    def generate():
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=full_messages,
                max_tokens=1024,
                temperature=0.85,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
