from flask import Flask, render_template, request, jsonify
from chat import CodeSpaceHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


code_space_handler = CodeSpaceHandler()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    query_result = code_space_handler.query_auto_merging(text)
    return query_result


if __name__ == '__main__':
    app.run(host="localhost", port=5000)
