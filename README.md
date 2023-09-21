# Plasmodium | HackMIT 2023

**1st Place Winner in the CareYaya Elder Tech Innovation Special Challenge**

Created by Rac Mukkamala, Victory Yinka-Banjo, Kosi Ugorji, Ananth Shyamal

## Inspiration
We were inspired by the development of foldscope, a very low-cost, high-resolution foldable microscope (https://en.wikipedia.org/wiki/Foldscope) capable of imaging blood cells. We wanted to create tools that can integrate with this microscope and, more generally, other applications and better improve health in the developing world. Malaria is a leading cause of death in many developing countries, where blood smears are used to identify the presence of parasites in red blood cells (RBCs). To improve the efficiency of detecting parasitic cells in blood smears, ultimately speeding up malaria diagnosis, we aimed to create an online tool that can do this in minutes. 

## What it does
Disease diagnoses is slow. We created a computer vision tool to swiftly diagnose malaria by analyzing blood cell imaging data, allowing for determination of infection rates & expediting clinical diagnoses. The project allows a user to upload a thin blood smear image, and it classifies the RBCs in the blood that are infected with malaria parasites.

## How we built it
We utilized a thin blood smear dataset from 193 patients in a Bangladesh hospital curated by the NIH. It consists of 20,000 labeled cells (exhibiting malaria parasitic infection or not) across 965 blood smear images. Given these images, we performed a multi-step image segmentation procedure to isolate the red blood cells (RBCs): we first used U-Net to segment the blood smears into cell clusters, then used Faster R-CNN to segment the cell clusters into individual RBCs, and then incorporated thresholding techniques to refine the segmentation and smooth the edges. Once each RBC in every blood smear was individually segmented, we trained a CNN to classify whether these segmented images contained the malaria parasite. Mapping these now-labeled segmented images back to their parent blood smears allowed us to output modified blood smear images highlighting the RBCs containing a malaria parasite. 

## Challenges we ran into
We encountered challenges in building an effective RBC segmentation pipeline. Variations in the segmentation procedure greatly affected the classification performance of the CNN, which was somewhat surprising. The various segmentation methodologies we explored yielded segments that looked visually very similar to hand-drawn segmentations provided in the NIH dataset, and these hand-drawn segmentations were classified very well by CNN. We tried integrating various thresholding, grayscale manipulations, filtering, and flood-filling methodologies to integrate with the U-Net + R-CNN for RBC segmentation. In addition, we originally started with pre-trained models like ResNet-18 for classification. However, they tended to overfit the training data, so we opted for a simple, untrained one-layer CNN architecture, which worked the best. 

## Accomplishments that we're proud of
We are proud that we were able to build a comprehensive segmentation and classification pipeline and that we were able to integrate this into a full-stack web app with a front end and a back end. 

## What we learned
We learned many technical skills along the way, such as using Python’s OpenCV framework for image processing/manipulation, various image segmentation methodologies, and using Flask to build out the web app.

## What's next for Plasmodium
In the future, we hope to continue developing our app and refining our segmentation/classification methodologies to increase accuracy. Furthermore, we plan to expand our pipeline to other diseases, such as sickle cell anemia, to create a more comprehensive health diagnostic tool for the developing world. 
