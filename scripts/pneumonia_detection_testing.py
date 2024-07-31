# -*- coding: utf-8 -*-
"""
**Authors:**

*   Danielle Rotem

*   Shahar Loantz

*   Yuval Hajbi


**Description**:

Testing notebook for pneumonia detection using chest x ray project.

**User guide:**

copy into your Google Drive root directory the folder "models_and_weights" from the shared folder : https://drive.google.com/drive/folders/1EIkepDGTZes43p2omPLophCPT3GPlYL_?usp=sharing

Please follow each assignment guide in order to run its test.

# MODEL A1

##Description: get a binary classification (normal or pneumonia) for a new image

## Instructions:
1. make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/A1/
2. make sure **A1** contais these two files: **'model.keras'** and **'weights.model.keras'**
3. run "load before testing"
4. in "test new data" - provide a path to the image(s) you want to test.
  * The images must be saved in your google drive.
  * The images or folder must be extracted and not in a zip.
5. run "test new data"

##load before testing
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from google.colab import drive

def load_model_and_weights():
    # Define the directory path where the model and weights are saved
    directory_path = '/content/drive/My Drive/models_and_weights/A1/'
    # Load the model architecture
    model_path = os.path.join(directory_path, 'model.keras')
    model = load_model(model_path)
    # Load the model weights
    weights_path = os.path.join(directory_path, 'weights.model.keras')
    model.load_weights(weights_path)
    return model  # Return the loaded model

def preprocess_input_data(input_data):
    preprocessed_data = []
    file_paths = []
    target_size = (224, 224)

    def process_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isdir(file_path):
                # Recursively process subfolders
                process_folder(file_path)
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Process image files
                img = cv2.imread(file_path)
                img = cv2.resize(img, target_size)
                preprocessed_data.append(img)
                file_paths.append(file_path)

    if os.path.isdir(input_data):
        process_folder(input_data)
    elif os.path.isfile(input_data) and input_data.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # If input_data is a single image file, preprocess that image
        img = cv2.imread(input_data)
        img = cv2.resize(img, target_size)
        preprocessed_data.append(img)
        file_paths.append(input_data)

    return preprocessed_data, file_paths


def display_images(normal_images, pneumonia_images, normal_file_paths, pneumonia_file_paths):
    for img, file_path in zip(normal_images, normal_file_paths):
        image_name = os.path.basename(file_path)  # Extract the image name from the file path
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        print(f"Prediction: Normal, Image Name: {image_name}")
    for img, file_path in zip(pneumonia_images, pneumonia_file_paths):
        image_name = os.path.basename(file_path)  # Extract the image name from the file path
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        print(f"Prediction: Pneumonia, Image Name: {image_name}")

def predictions_graph(normal, pneumonia):
    normal_count = len(normal)
    pneumonia_count = len(pneumonia)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.bar(['Normal', 'Pneumonia'], [normal_count, pneumonia_count], color=['blue', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Predicted Class Distribution')

def get_prediction_A1(data):
    drive.mount('/content/drive', force_remount=True)
    #load model
    model = load_model_and_weights()
    # Preprocess the new input data
    preprocessed_data, file_paths = preprocess_input_data(data)
    # Perform predictions
    predictions = model.predict(np.array(preprocessed_data))
    # Define class labels
    class_labels = ['Normal', 'Pneumonia']
    # Initialize lists to store image paths for each class
    normal_images = []
    pneumonia_images = []
    # Convert binary predictions to class labels and store corresponding image paths
    for i in range(len(preprocessed_data)):
        predicted_class = class_labels[int(predictions[i][0] > 0.5)]
        if predicted_class == 'Normal':
            normal_images.append(preprocessed_data[i])
        else:
            pneumonia_images.append(preprocessed_data[i])
    # Display images
    predictions_graph(normal_images, pneumonia_images)
    # Display images
    display_images(normal_images, pneumonia_images, file_paths[:len(normal_images)], file_paths[len(normal_images):])

"""##test new data"""

# --------------------------------- test ----------------------------------------------

#new_data = # Replace with your actual directory path  - for example "/content/drive/My Drive/IMAGES/"

new_data = "/content/drive/My Drive/IMG/"

get_prediction_A1(new_data)

"""# MODEL A2

##Description: get a multi class classification (normal, virus or bacteria) for a new image

## Instructions:
1. make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/A2/
2. make sure **A2** contais these two files: **'model.keras'** and **'weights.model.keras'**
3. run "load before testing"
4. in "test new data" - provide a path to the image(s) you want to test.
  * The images must be saved in your google drive.
  * The images or folder must be extracted and not in a zip.
5. run "test new data"

##load before testing
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from google.colab import drive

def load_model_and_weights():
    # Define the directory path where the model and weights are saved
    directory_path = '/content/drive/My Drive/models_and_weights/A2/'
    # Load the model architecture
    model_path = os.path.join(directory_path, 'model.keras')
    model = load_model(model_path)
    # Load the model weights
    weights_path = os.path.join(directory_path, 'weights.model.keras')
    model.load_weights(weights_path)
    return model  # Return the loaded model

def preprocess_input_data(input_data):
    preprocessed_data = []
    file_paths = []
    target_size = (224, 224)

    def process_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isdir(file_path):
                # Recursively process subfolders
                process_folder(file_path)
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Process image files
                img = cv2.imread(file_path)
                img = cv2.resize(img, target_size)
                preprocessed_data.append(img)
                file_paths.append(file_path)

    if os.path.isdir(input_data):
        process_folder(input_data)
    elif os.path.isfile(input_data) and input_data.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # If input_data is a single image file, preprocess that image
        img = cv2.imread(input_data)
        img = cv2.resize(img, target_size)
        preprocessed_data.append(img)
        file_paths.append(input_data)

    return preprocessed_data, file_paths

def display_images(images, labels, file_paths):
    for img, label, file_path in zip(images, labels, file_paths):
        image_name = os.path.basename(file_path)  # Extract the image name from the file path
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        print(f"Prediction: {label}, Image Name: {image_name}")

def predictions_graph(predictions, class_labels):
    class_counts = {label: 0 for label in class_labels}
    for prediction in predictions:
        predicted_class = class_labels[np.argmax(prediction)]
        class_counts[predicted_class] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'green', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Predicted Class Distribution')
    plt.show()

def get_prediction_A2(data):
    drive.mount('/content/drive', force_remount=True)
    #load model
    model = load_model_and_weights()
    # Preprocess the new input data
    preprocessed_data, file_paths = preprocess_input_data(data)
    # Perform predictions
    predictions = model.predict(np.array(preprocessed_data))
    # Define class labels
    class_labels = ['bacteria', 'normal', 'virus']
    # Display images
    predictions_graph(predictions, class_labels)
    # Convert predictions to class labels
    predicted_labels = [class_labels[np.argmax(pred)] for pred in predictions]
    # Display images
    display_images(preprocessed_data, predicted_labels, file_paths)

"""##test new data"""

# --------------------------------- test ----------------------------------------------

new_data = # Replace with your actual directory path  - for example "/content/drive/My Drive/IMAGES/"

get_prediction_A2(new_data)

"""# MODEL B

##Description: get a classification using knn and trained embedding vectors for a new image

## Binary Prediction ('Normal', 'Pneumonia')

### Instructions:
1. make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/A1/
2. make sure **A1** contais these three files: **'model.keras'** and **'weights.model.keras'**
3.make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/B/binary/
4. make sure **binary** contais this file: **knn_classifier.pkl**
5. run "load before testing"
6. in "test new data" - provide a path to the image you want to test.
  * The images must be saved in your google drive.
  * the path must be to a single image
7. run "test new data"

### load before testing
"""

import os
import cv2
import joblib
import numpy as np
from google.colab import drive
from tensorflow.keras import models
from google.colab.patches import cv2_imshow
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier

def load_model_and_weights():
    # Define the directory path where the model and weights are saved
    directory_path = '/content/drive/My Drive/models_and_weights/A1/'
    # Load the model architecture
    model_path = os.path.join(directory_path, 'model.keras')
    model = load_model(model_path)
    # Load the model weights
    weights_path = os.path.join(directory_path, 'weights.model.keras')
    model.load_weights(weights_path)
    return model  # Return the loaded model

def preprocess_input_data(input_data):
    preprocessed_data = []
    file_paths = []
    target_size = (224, 224)
    if os.path.isfile(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # If input_data is a single image file, preprocess that image
        img_path = input_data
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            preprocessed_data.append(img)
            file_paths.append(img_path)
        else:
            print("Warning: Unable to read the image file.")
    else:
        print("Warning: Input data must be a path to a single image file with extension '.png', '.jpg', '.jpeg', or '.gif'.")

    return preprocessed_data, file_paths

def load_knn_model():
    # Load the trained classifier from Google Drive
    knn_classifier_loaded = joblib.load('/content/drive/My Drive/models_and_weights/B/binary/knn_classifier.pkl')
    return knn_classifier_loaded


def get_Binary_prediction(image_path):
    drive.mount('/content/drive', force_remount=True)
    # Preprocess the input image
    preprocessed_data, file_paths = preprocess_input_data(image_path)
    # Ensure that preprocessed_data is not empty
    if not preprocessed_data:
        print("Error: No valid images found.")
        return
    # Load model and embedding vectors
    model = load_model_and_weights()
    embedding_model = models.Model(inputs=model.input, outputs=model.layers[-4].output)
    # Get the embedding vectors for the input images
    img_embeddings = embedding_model.predict(np.array(preprocessed_data))
    knn_classifier = load_knn_model()
    # Predict the class labels of the input image embedding vectors
    predicted_class = knn_classifier.predict(img_embeddings)
    if(predicted_class == 1):
      predicted_class = 'Pneumonia'
    else:
      predicted_class = 'Normal'

    image_name = os.path.basename(file_paths[0])  # Extract the filename from the path
    print(f"Predicted Class for {image_name} is: {predicted_class}")
    # Display the preprocessed image
    cv2_imshow(preprocessed_data[0])

"""### test new image



"""

# --------------------------------- test ----------------------------------------------

new_data = # Replace with your actual directory path  - for example "/content/drive/My Drive/IMAGES/image.jpeg"

get_Binary_prediction(new_data)

"""## Multi Class Prediction ('Bacteria', 'Normal', 'Virus')

### Instructions:
1. make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/A2/
2. make sure **A2** contais these three files: **'model.keras'** and **'weights.model.keras'**
3.make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/B/multiclass/
4. make sure **multiclass** contais this file: **knn_classifier.pkl**
5. run "load before testing"
6. in "test new data" - provide a path to the image you want to test.
  * The images must be saved in your google drive.
  * the path must be to a single image
7. run "test new data"

### load before testing
"""

import os
import cv2
import joblib
import numpy as np
from google.colab import drive
from tensorflow.keras import models
from google.colab.patches import cv2_imshow
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier

def load_model_and_weights():
    # Define the directory path where the model and weights are saved
    directory_path = '/content/drive/My Drive/models_and_weights/A2/'
    # Load the model architecture
    model_path = os.path.join(directory_path, 'model.keras')
    model = load_model(model_path)
    # Load the model weights
    weights_path = os.path.join(directory_path, 'weights.model.keras')
    model.load_weights(weights_path)
    return model  # Return the loaded model

def preprocess_input_data(input_data):
    preprocessed_data = []
    file_paths = []
    target_size = (224, 224)
    if os.path.isfile(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # If input_data is a single image file, preprocess that image
        img_path = input_data
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            preprocessed_data.append(img)
            file_paths.append(img_path)
        else:
            print("Warning: Unable to read the image file.")
    else:
        print("Warning: Input data must be a path to a single image file with extension '.png', '.jpg', '.jpeg', or '.gif'.")

    return preprocessed_data, file_paths

def load_knn_model():
    # Load the trained classifier from Google Drive
    knn_classifier_loaded = joblib.load('/content/drive/My Drive/models_and_weights/B/multiclass/knn_classifier.pkl')
    return knn_classifier_loaded

def get_MultiClass_prediction(image_path):
    drive.mount('/content/drive', force_remount=True)
    # Preprocess the input image
    preprocessed_data, file_paths = preprocess_input_data(image_path)
    # Ensure that preprocessed_data is not empty
    if not preprocessed_data:
        print("Error: No valid images found.")
        return
    # Load model and embedding vectors
    model = load_model_and_weights()
    embedding_model = models.Model(inputs=model.input, outputs=model.layers[-4].output)
    # Get the embedding vectors for the input images
    img_embeddings = embedding_model.predict(np.array(preprocessed_data))
    knn_classifier = load_knn_model()
    # Predict the class labels of the input image embedding vectors
    predicted_classes = knn_classifier.predict(img_embeddings)

    image_name = os.path.basename(file_paths[0])  # Extract the filename from the path
    print(f"Predicted Class for {image_name} is: {predicted_classes[0]}")
     # Display the preprocessed image
    cv2_imshow(preprocessed_data[0])

"""### test new image"""

# --------------------------------- test ----------------------------------------------

new_data = # Replace with your actual directory path  - for example "/content/drive/My Drive/IMAGES/image.jpeg"

get_MultiClass_prediction(new_data)

"""# MODEL D

##Description: detect anomalies -  get a binary classification (normal or anomaly) for a new image

## Instructions:
1. make sure you have this path in your google drive: /content/drive/My Drive/models_and_weights/D/
2. make sure **D** contais these three files: **'model.keras'** , **'weights.model.keras'** and **'threshold.txt'**
3. run "load before testing"
4. in "test new data" - provide a path to the image you want to test.
  * The images must be saved in your google drive.
  * The images or folder must be extracted and not in a zip.
5. run "test new data"

##load before testing
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from google.colab import drive
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import eval
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt


def load_model_and_threshold():
    # Define the directory path where the model and weights are saved
    directory_path = '/content/drive/My Drive/models_and_weights/D/'
    # Load the model architecture
    model_path = os.path.join(directory_path, 'model.keras')
    model = load_model(model_path)
    # Load the model weights
    weights_path = os.path.join(directory_path, 'weights.model.keras')
    model.load_weights(weights_path)
    # Define the filepath for the threshold file
    threshold_file_path = '/content/drive/My Drive/models_and_weights/D/threshold.txt'
    # Load the threshold from the file
    with open(threshold_file_path, 'r') as file:
      threshold = float(file.read())
    return model, threshold

def preprocess_input_data(input_data):
    preprocessed_data = []
    file_paths = []
    target_size = (224, 224)

    def process_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isdir(file_path):
                # Recursively process subfolders
                process_folder(file_path)
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Process image files
                img = cv2.imread(file_path)
                img = cv2.resize(img, target_size)
                img_float = img.astype('float32')
                img_normalized = img_float / 255.0
                preprocessed_data.append(img_normalized)
                file_paths.append(file_path)

    if os.path.isdir(input_data):
        process_folder(input_data)
    elif os.path.isfile(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # If input_data is a single image file, preprocess that image
        img = cv2.imread(input_data)
        img = cv2.resize(img, target_size)
        img_float = img.astype('float32')
        img_normalized = img_float / 255.0
        preprocessed_data.append(img_normalized)
        file_paths.append(input_data)
    else:
        raise ValueError("Input data must be a path to a single image file, a folder containing images, or a folder that contains subfolders with images.")

    return np.array(preprocessed_data), np.array(file_paths)


def display_images(predicted_labels, file_paths, images):
    # Extract image names from file paths
    image_names = [os.path.basename(path) for path in file_paths]
    # Print the predicted label for each image along with the image name
    for i, (predicted_label, image_name, image) in enumerate(zip(predicted_labels, image_names, images)):
        print(f"Predicted label for image {image_name} is: {predicted_label}")
        # Display the image using cv2_imshow or any other method
        cv2_imshow(image * 255.0)

def plot_prediction_counts(predicted_labels):
    # Count occurrences of 'NORMAL' and 'ANOMALY'
    counts = {'NORMAL': predicted_labels.count('NORMAL'), 'ANOMALY': predicted_labels.count('ANOMALY')}
    # Plot the counts as a bar graph
    plt.bar(counts.keys(), counts.values(), color=['blue', 'red'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Prediction Counts')
    plt.show()

def predict_anomaly(new_data):
    drive.mount('/content/drive', force_remount=True)
    autoencoder, threshold = load_model_and_threshold()
    processed_data, file_paths = preprocess_input_data(new_data)
    reconstructed_images_test = autoencoder.predict(processed_data)
    test_loss = tf.keras.losses.binary_crossentropy(processed_data, reconstructed_images_test)
    test_loss = tf.keras.backend.eval(test_loss)
    predicted_labels_test = []
    # Iterate over each reconstruction loss in test_loss
    for loss in test_loss:
        mean_loss = np.mean(loss)
        # Check if the loss is greater than the threshold
        if mean_loss > threshold:
            # If so, append 'anomaly' to the predicted_labels_test list
            predicted_labels_test.append('ANOMALY')
        else:
            # Otherwise, append 'normal'
            predicted_labels_test.append('NORMAL')
    plot_prediction_counts(predicted_labels_test)
    display_images(predicted_labels_test, file_paths, processed_data)

"""##test new data"""

# --------------------------------- test ----------------------------------------------

new_data = # Replace with your actual directory path  - for example "/content/drive/My Drive/IMAGES/"

predict_anomaly(new_data)