from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    return jsonify({"reply": f"You said: {user_message}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
