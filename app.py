from flask import Flask, request, jsonify, render_template_string
import boto3
import json
import os

app = Flask(__name__)
REGION = os.getenv("AWS_REGION", "us-east-1")
PROMPT = os.getenv(
    "BEDROCK_PROMPT_ARN",
    "arn:aws:bedrock:us-east-1:381492212823:prompt-router/e2nmuu4dw3ai"
)
client = boto3.client("bedrock-runtime", region_name=REGION)

conversation_history = []

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Claude Conversational AI Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        #chat-box { border: 1px solid #ccc; padding: 1em; max-width: 600px; height: 300px; overflow-y: scroll; background: #f9f9f9; }
        #user-input { width: 400px; padding: 0.5em; }
    </style>
</head>
<body>
    <h2>Claude Conversational AI Assistant</h2>
    <div id="chat-box"></div>
    <input type="text" id="user-input" placeholder="Type your message" />
    <button onclick="sendMessage()">Send</button>

    <script>
        async function sendMessage() {
            const input = document.getElementById("user-input");
            const message = input.value;
            if (!message.trim()) return;

            const chatBox = document.getElementById("chat-box");
            chatBox.innerHTML += "<div><b>You:</b> " + message + "</div>";
            input.value = "";

            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_input: message })
            });

            const data = await response.json();
            if (data.response) {
                chatBox.innerHTML += "<div><b>Claude:</b> " + data.response + "</div>";
            } else {
                chatBox.innerHTML += "<div style='color:red;'><b>Error:</b> " + (data.error || "Unknown error") + "</div>";
            }

            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

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
            modelId=PROMPT, 
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
