# import os
from flask import Flask, jsonify, request
from gemini_functions import process_query
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)

# List college JSON data paths 
"""
College of Engineering Chengannur ID - 1
College of Engineering Karunagapally - 2
Model Engineering College            - 3
College of Applied Science Adoor     - 4
College of Engineering Poonjar       - 5
College of Engineering Kalloopaara   - 6
"""

college_data_paths = [
    '',
    'college_json_data/cec.json', 
    'college_json_data/cek.json', 
    'college_json_data/mec.json',
    'college_json_data/casa.json',
    'college_json_data/cep.json',
    'college_json_data/cekp.json'
    ]

@app.route("/get_data", methods=["POST"])
def get_response():
    try:
        # Expect JSON payload with 'question' key
        data = request.get_json()
        print(data, "data from the frontend") # for debugging purpose only
        
        user_question = data['question'] 
        college_index = data.get('colleges',None) # list from the frontend of colleges selected
        process_input = { "user_question":user_question, "college_file_path":college_data_paths[college_index]} 
        print(process_input)
        response = process_query(process_input)
        print(response)
        
        return jsonify({
            "output": response.get('output_text', 'No response generated'),
            "image_urls": response.get('image_urls', "No Images"),
            "status": "success"
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/", methods=["GET"])
def  home_page():
    return "go to /get_data to get response"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
