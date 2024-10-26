from flask import Flask, request, render_template, jsonify
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ResNet18 in this case)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Use new weight loading
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes
model.load_state_dict(torch.load('models\ResNet18.pt'))  # Load model weights
model.eval()  # Set model to evaluation mode


# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the state names corresponding to model outputs
state = [
    'safe driving',
    'texting - right',
    'talking on the phone - right',
    'texting - left',
    'talking on the phone - left',
    'operating the radio',
    'drinking',
    'reaching behind',
    'hair and makeup',
    'talking to passenger',
    'UNKNOWN'
]

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML file for uploading images

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains an image
    if 'file' not in request.files:
        error_message = 'No image uploaded. Please upload an image file.'
        return render_template('error.html', error_message=error_message), 400

    file = request.files['file']

    # Check if the file is valid
    if file.filename == '':
        error_message = 'No file selected. Please select an image to upload.'
        return render_template('error.html', error_message=error_message), 

    try:
        # Read the image file and preprocess it
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess(img).unsqueeze(0)  # Add batch dimensiono

        # Make the prediction
        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output, 1)

        # Get the state name
        state_name = state[predicted_class.item()]

        # Render the result page with the prediction
        return render_template('result.html', prediction=predicted_class.item(), state_name=state_name)

    except Exception as e:
        error_message = 'An error occurred during image processing. Please try again.'
        return render_template('error.html', error_message=error_message), 500

if __name__ == '__main__':
    app.run(debug=True)
