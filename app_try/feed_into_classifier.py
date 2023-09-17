import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
from tqdm import tqdm
import cv2

import base64
from PIL import Image
from io import BytesIO

#pip install Pillow 

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Flatten(),
                                    nn.Linear(56*56*16, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2))

    def forward(self, X):
        return self.model(X)

def base64_to_image(base64_str, output_path):
    # Decode the base64 string
    decoded_bytes = base64.b64decode(base64_str)
    # Convert bytes to a PIL image
    image = Image.open(BytesIO(decoded_bytes))
    image.save(output_path)

def image_to_base64(image_path):
    # Open the image and convert it to bytes
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        
        # Encode the bytes to base64 string
        img_str = base64.b64encode(buffered.getvalue())
        
    return img_str.decode("utf-8")
    

def remove_background(in_path, out_path):
    image = cv2.imread(in_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    black_bg = np.zeros_like(image)
    image_with_black_bg = cv2.bitwise_and(image, image, mask=mask_inv) + black_bg
    cv2.imwrite(out_path, image_with_black_bg)


def crop_multiple_rectangles(image_path, rectangles, crop_img_dir):
    # Load the image
    image = cv2.imread(image_path)
    for idx, coord in enumerate(rectangles):
        x, y, width, height = coord.astype(int)
        cell_image = image[y:y+height, x:x+width]

        # Convert the image to grayscale
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove small noise by morphology operations
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        ########
        # Step 1: Create a black mask of the same size as the image
        mask = np.zeros_like(cell_image)

        # Step 2: Draw a white-filled ellipse on this mask
        center = (cell_image.shape[1]//2, cell_image.shape[0]//2)  # center of the image
        axes = (cell_image.shape[1]//2, cell_image.shape[0]//2)   # semi-major and semi-minor axes
        angle = 0  # angle of rotation
        start_angle = 0
        end_angle = 360
        color = (255, 255, 255)  # white color
        thickness = -1  # fill the ellipse
        cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, color, thickness)

        # Step 3: Use the mask to get the elliptical region of the image
        result = cv2.bitwise_and(cell_image, mask)
        cv2.imwrite(os.path.join(crop_img_dir, f'cropped_img_{idx}.png'), result)

    


def classify_cropped_images(cropped_images_path, model_path='./classifier.pth'):
    #set up pytorch model
    def gray_world_assumption(img):
        # Calculate the average color of the entire image
        mean_color = img.mean(dim=[1, 2])
        # Calculate the correction factors for each channel
        correction_factors = torch.tensor([0.5, 0.5, 0.5]) / mean_color
        # Apply the correction to the image
        corrected_img = correction_factors.view(3, 1, 1) * img
        #corrected_img = img - mean_color.view(3,1,1) + 0.5
        return corrected_img
    
    trans = transforms.Compose([
        transforms.Resize((230,230)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        gray_world_assumption,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(cropped_images_path, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # for images, labels in dataloader:
    #     for i in range(images.shape[0]):
    #         image = images[i]  # Get a single image
    #         label = labels[i]  # Get the label for that image

    #         # Undo the normalization for display
    #         #image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    #         image = torch.clamp(image, 0, 1)  # Clip to ensure valid pixel values

    #         # Convert the image tensor to a NumPy array
    #         image_np = image.permute(1, 2, 0).numpy()
    #         # Display the image
    #         plt.imshow(image_np)
    #         plt.show()

    model = torch.load(model_path)

    all_preds = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to('mps')
            outputs = model(batch)
            print(outputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def draw_rectangles(in_path, rectangle_coords, out_path):
    # Load the image
    image = cv2.imread(in_path)

    # Draw red rectangles on the image
    for rect in rectangle_coords:
        x, y, width, height = rect.astype(int)
        cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 8)  # Drawing the rectangle in red with thickness 2

    cv2.imwrite(out_path, image)
    cv2.imwrite('./result.png', image) #delete

    # If you want to save the image
    # cv2.imwrite('output_image_with_rectangles.jpg', image)


def run_pipeline(input_str, is_bstring=True):
    try:
        os.mkdir('./tmp')
    except Exception:
        pass
    os.system('rm -rf ./tmp/*')
    
    if is_bstring:
        image_path = base64_to_image(input_str, './tmp/input_img.png')
        image_path = './tmp/input_img.png'
    else:
        image_path = input_str
    #Run matlab script

    os.system(f'./matlab/run_matlab.py {os.path.abspath(image_path)} > ./tmp/matlab_log.txt')
    mat_data = scipy.io.loadmat('./tmp/Segmentation_Output/_Bounding Boxes_.mat')
    rectangle_coords = mat_data['allrect']
    
    try:
        os.mkdir('./tmp/cropped_images')
        os.mkdir('./tmp/cropped_images/imgs')
    except Exception:
        pass
    cropped_img_dir = './tmp/cropped_images/imgs'
    crop_multiple_rectangles(image_path, rectangle_coords, cropped_img_dir)

    classify_cropped_img_dir = './tmp/cropped_images'
    preds = classify_cropped_images(classify_cropped_img_dir, model_path='./classifier2.pth')

    rectangles_to_highlight = rectangle_coords[preds == 0, :]
    result_path = './tmp/result.png'
    draw_rectangles(image_path, rectangles_to_highlight, result_path)
    result = Image.open(result_path)
    return image_to_base64(result_path)


if __name__ == '__main__':
    run_pipeline('/Users/racmukk/Documents/MIT/HackMIT 23/hackMIT2023/NIH-NLM-ThinBloodSmearsPf/Polygon Set/211C70P31_ThinF/Img/IMG_20150813_130510.jpg', is_bstring=False)
    #preds = classify_cropped_images('/Users/racmukk/Documents/MIT/HackMIT 23/hackMIT2023/cell_images/test_infected')
    #print(np.mean(preds))