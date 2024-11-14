from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model, predict

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Path to your YOLO model (.pt file) containing the trained weights
MODEL_PATH = "C:\\Users\\frjer\\Videos\\CNN Model\\app\\weights\\bestl.pt" # Update with the correct path to your model
model = load_model(MODEL_PATH)  # Load the model once when the app starts

@app.route('/')
def home():
    """
    Home route to test if the server is up and running.
    """
    return "Mango Quality Classification API - Use /predict to test."

@app.route('/predict', methods=['POST'])
def get_prediction():
    """
    Predict the type and grade of the mango from the uploaded image.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Run the model on the uploaded image
    result = predict(model, file)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app locally (development mode)
