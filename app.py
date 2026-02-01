from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model ONCE (important for performance)
MODEL_NAME = "distilgpt2"  # SAFE DEFAULT

print(f"Loading model {MODEL_NAME}... This may take a moment.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

def generate_response(user_input):
    """
    Generate a response from the model based on the user input.
    """
    prompt = f"""
You are a friendly chatbot that gives short, clear answers.

User: {user_input}
Bot:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = response.split("Bot:")[-1].strip()

    return bot_reply


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"response": "Say something, don’t be shy 🙂"})

    bot_response = generate_response(user_message)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    # Set debug=False if deploying to production
    app.run(debug=True)
