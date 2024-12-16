from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = load_model('plantmodel.h5')

# Class labels
class_labels = ['Aloe_Vera', 'Areca_Palm', 'Boston_Fern', 'Cactus', 'Chinese_Evergreen', 
                'English_Ivy', 'Peace_Lily', 'Rubber_Plant', 'Snake_Plant', 
                'Spider_Plant', 'Tulsi', 'ZZ_Plant']

# Ensure the uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file uploaded')

        file = request.files['file']

        if file:
            img_path = os.path.join('uploads', file.filename)  # Save the file to the uploads directory
            file.save(img_path)

            # Preprocess the image
            img = image.load_img(img_path, target_size=(225, 225))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict the class
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]

            return render_template('index.html', prediction=predicted_label, image_file=file.filename)

    return render_template('index.html', prediction=None)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
