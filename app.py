import os
import json
import boto3
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set the model ID for the Bedrock model (Claude 3.7 Sonnet)
MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"

# Initialize the boto3 client for Bedrock
bedrock = boto3.client('bedrock', region_name='us-east-1')  # Change region if needed

# Function to get conversation context (mocked for this example)
def get_conversation_context(user_id):
    # For this example, let's assume the context is stored in a file
    try:
        with open(f"conversation{user_id}.json", "r") as file:
            context_data = json.load(file)
        return context_data
    except FileNotFoundError:
        return {"messages": []}

# Function to save conversation context (mocked for this example)
def save_conversation_context(user_id, context_data):
    with open(f"conversation{user_id}.json", "w") as file:
        json.dump(context_data, file)

# Function to call the Bedrock model
def ask_bedrock_model(prompt):
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "top_k": 250,
                "stop_sequences": [],
                "temperature": 1,
                "top_p": 0.999,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            })
        )
        result = json.loads(response['body'].read())
        return result['choices'][0]['message']['text']
    except Exception as e:
        return str(e)

# Route to handle the conversational AI
@app.route('/ask', methods=['POST'])
def ask():
    # Get the input data from the user (via JSON)
    data = request.get_json()

    # Extract user ID and question
    user_id = data.get('user_id')
    question = data.get('question')

    if not user_id or not question:
        return jsonify({"error": "User ID and question are required"}), 400

    # Get the conversation context for the user (if any)
    context = get_conversation_context(user_id)
    context['messages'].append({"role": "user", "content": question})

    # Call Bedrock model with the current prompt (question)
    response_text = ask_bedrock_model(question)

    # Add the model's response to the conversation context
    context['messages'].append({"role": "assistant", "content": response_text})

    # Save the updated context
    save_conversation_context(user_id, context)

    # Return the AI's response to the user
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
