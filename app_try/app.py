from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from PIL import Image  # Pillow library for image processing
import base64
from io import BytesIO
from rembg import remove
from feed_into_classifier import run_pipeline
from custom_cnn import CustomCNN


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# model = ru_pn_pipeline('input_string')

# Function to process the uploaded image
def process_image(image_path):
    try:
        # Open and process the image (e.g., resize, filter, etc.)
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Example: Resize the image to 300x300 pixels

        # Save the processed image
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
        img.save(processed_image_path)

        return processed_image_path
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')


# @app.route("/upload", methods=['GET', 'POST'])
# def upload():
#     base64_img = None
#     metrics = {}

#     if request.method == 'POST' and 'image' in request.files:
#         image = request.files['image']

#         if image.filename == '':
#             return "No selected file"

#         if image:
#             # Convert the image to base64
#             buffered = BytesIO()
#             img = Image.open(image)
#             img = process_image(img)  # Processing the image
#             img.save(buffered, format="PNG")
#             base64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

#             # annotate_b64image, count = model(base64_img) #will modify later to name of model


@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # image = remove(Image.open(file))
            processed_image = remove(file.read())
            # image_data = base64.b64encode(file.read()).decode('utf-8')
            image_data = base64.b64encode(processed_image).decode('utf-8')

    #need to add in the code here to get back whatever the model returns, convert that to base 64 and then display on the front end
            # annotated_image, labels = model(image_data)
            annotated_image = run_pipeline(image_data)
            # percent_value = (labels.count('YES')/len(labels))**100


    # return render_template('index.html', image_data = image_data)
    return render_template('index.html', image_data = annotated_image)

# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filename)

#         # Convert the image to base64
#         buffered = BytesIO()
#         img = Image.open(image)
#         img = Image.Ops.flip(img)  # Processing the image
#         img.save(buffered, format="PNG")
#         base64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')


#         processed_image_path = process_image(filename)
#         if processed_image_path:
#             return redirect(url_for('processed_image', filename='processed_image.jpg'))
#         else:
#             return "Image processing failed."


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('uploaded.html', filename=filename)

@app.route('/processed/<filename>')
def processed_image(filename):
    return render_template('index.html', processed_filename=filename)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5001)
