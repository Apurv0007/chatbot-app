from flask import Flask, request, jsonify, render_template
from chatbot import generate_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    bot_reply = generate_response(user_msg)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
