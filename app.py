from flask import Flask, request, jsonify
import boto3
import json
import os

app = Flask(__name__)

# Create the Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-east-1")  # adjust if needed

# Simple in-memory session tracking (note: won't persist across App Runner restarts)
conversation_history = []

@app.route("/")
def health_check():
    return "Claude Conversational AI Assistant is running."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input", "").strip()
    
    if not user_input:
        return jsonify({"error": "Missing user input"}), 400

    # Append user message to conversation history
    conversation_history.append(f"Human: {user_input}")

    # Build prompt from full history
    prompt = "\n".join(conversation_history) + "\nAssistant:"

    # Construct the body payload
    body = {
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.7,
        "stop_sequences": ["\nHuman:"]
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0", 
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        bot_response = result["completion"].strip()

        # Append assistant reply to conversation history
        conversation_history.append(f"Assistant: {bot_response}")

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5001)
