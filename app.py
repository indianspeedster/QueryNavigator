from flask import Flask, render_template, request, jsonify, flash, redirect
from werkzeug.utils import secure_filename
from chat import CodeSpaceHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import subprocess
import traceback


code_space_handler = CodeSpaceHandler()

UPLOAD_FOLDER = 'docs'
if not os.path.exists(UPLOAD_FOLDER): 
    os.makedirs(UPLOAD_FOLDER) 
ALLOWED_EXTENSIONS = set(['pdf', 'docx', 'txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        if 'files[]' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
    
        files = request.files.getlist('files[]')
        errors = {}
        success = False
        success_msg = 'File successfully uploaded !'

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                success = True
            else:
                errors[file.filename] = 'File type is not allowed'
        
        if success and errors:
            errors['message'] = success_msg
            resp = jsonify(errors)
            resp.status_code = 206
            return resp
        if success:
            resp = jsonify({'message' : success_msg})
            resp.status_code = 201
            return resp
        else:
            resp = jsonify(errors)
            resp.status_code = 400
            return resp
    return render_template('upload_multi_files.html')
    # return render_template("upload_file.html", msg = msg, name = file.filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prep_data', methods=['POST'])
def dataPreProcessing():
    print ("I am in data pre processing after successful upload")
    try:
        subprocess.run(['python', 'data.py'], check=True)
        print('Python script executed successfully')
        return jsonify({'message': 'Data preprocessing completed successfully'})
    except subprocess.CalledProcessError as e:
        print(f'Error in dataPreProcessing: {e}')
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during data preprocessing'})


if __name__ == '__main__':
    app.run(host="localhost", port=5000)
