from flask import Flask, request, jsonify
import boto3
import json
import os

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "temporary-secret-key") 
REGION = os.getenv("AWS_REGION", "us-east-1")
client = boto3.client("bedrock-runtime", region_name=REGION)

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

    # Build prompt from conversation history
    prompt = "\n".join(conversation_history) + "\nAssistant:"

    # Construct Bedrock payload
    body = {
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.5,
        "stop_sequences": ["\nHuman:"]
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-v2:1", 
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        bot_response = result["completion"].strip()

        # Append assistant reply to history
        conversation_history.append(f"Assistant: {bot_response}")

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({
            "error": "Failed to invoke model",
            "detail": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
