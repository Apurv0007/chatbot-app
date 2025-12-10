from flask import Flask, render_template, request, jsonify
import openai
import os

app = Flask(__name__)

# Set your OpenAI API key here (or use environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")  # safer than hardcoding

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"reply": "Please type a message!"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150
        )

        bot_reply = response.choices[0].message.content.strip()

    except Exception as e:
        bot_reply = "Sorry, something went wrong."

    # ⚠ Make sure the key is "reply"
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
