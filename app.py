from flask import Flask, request, render_template, redirect, send_from_directory, url_for
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import wikipedia
import requests

app = Flask(__name__)

# Define number of disease classes
num_classes = 5

# Load trained model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('models/skin_disease_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
img_width, img_height = 150, 150
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
class_labels = {
    0: 'Acne',
    1: 'Hairloss',
    2: 'Nail Fungus',
    3: 'Normal',
    4: 'Skin Allergy'
}

# Predict skin disease from image
def predict_skin_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    return class_labels[predicted.item()]

# Fetch Wikipedia summary
def get_wikipedia_summary(topic):
    try:
        return wikipedia.summary(topic, sentences=2)
    except Exception:
        return "Description not available for this condition."

# --------- Routes ----------

# ✅ Landing Page
@app.route('/')
def home():
    return render_template('index.html')

# ✅ About Page
@app.route('/about')
def about():
    return render_template('about.html')

# ✅ Upload Image (Default Route)
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict_skin_disease(file_path)
            summary = get_wikipedia_summary(prediction)
            return render_template('result.html', prediction=prediction, summary=summary, image_file=file.filename)
    return render_template('upload.html')

# ✅ Uploaded Image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# ✅ Contact Page with GitHub API
@app.route('/contact')
def contact():
    usernames = ['rajraushan844','kisankumarbhagat','PranavPrakash28']
    profiles = []
    for username in usernames:
        url = f'https://api.github.com/users/{username}'
        try:
            response = requests.get(url)
            data = response.json()
            profiles.append({
                'name': data.get('name', username),
                'bio': data.get('bio', 'No bio available.'),
                'avatar': data.get('avatar_url', ''),
                'url': data.get('html_url', f'https://github.com/{username}')
            })
        except Exception:
            profiles.append({
                'name': username,
                'bio': 'Unable to load bio.',
                'avatar': '',
                'url': f'https://github.com/{username}'
            })
    return render_template('contact.html', profiles=profiles)

# ✅ Run App
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Get port from environment (Render sets this)
    
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    # Run the app on 0.0.0.0 to allow external access
    app.run(host='0.0.0.0', port=port, debug=True)

