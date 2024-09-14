# **Wings of Wisdom - Bird Species Classification**

Welcome to the **Wings of Wisdom** project repository! 🦅 This project focuses on building a **multi-class classifier** to accurately identify various **bird species** based on images. Leveraging **deep learning**, we use **pretrained models** such as ResNet50, EfficientNetB1, and XceptionNet to classify bird species from a dataset containing images of 200 unique species.

<br>

## 📋 **Table of Contents**
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Model Architectures](#-model-architectures)
- [Installation and Setup](#-installation-and-setup)
- [Results](#-results)
- [Challenges and Learning](#-challenges-and-learning)
- [Future Work](#-future-work)
- [Contributing](#-contributing)


<br>

## 🌟 **Project Overview**

Bird species identification plays a crucial role in biodiversity studies and wildlife preservation. The **Wings of Wisdom** classifier takes on this challenge by classifying bird species with **high accuracy** using cutting-edge **convolutional neural networks (CNNs)**. 

Through this project, we aim to develop a scalable and efficient model that can handle real-world image data, where bird species identification is critical.

<br>

## 🐦 **Dataset**

The dataset contains images of **200 bird species**, each labeled with bounding box coordinates, allowing us to isolate and focus on the bird in each image. The dataset is organized as follows:

- **Classes**: 200 bird species
- **Bounding Boxes**: Provided for each image
- **Train/Validation Split**: Data has been split for training and validation to optimize model performance.

<br>

## 🧠 **Approach**

Our approach to solving the classification task involved several steps:

1. **Data Preprocessing**:
   - **Image Resizing**: All images were resized to a consistent dimension for input into the models.
   - **Bounding Box Utilization**: Cropped bird images using bounding boxes to focus the model on the bird rather than the background noise.
   - **Data Augmentation**: Applied techniques like rotation, zoom, and horizontal flips to improve generalization.

2. **Model Training**:
   - Fine-tuned several **pretrained models** (ResNet50, EfficientNetB1, XceptionNet) to adapt them to the bird species classification task.
   - Used **transfer learning** to leverage the feature extraction power of pretrained models.

3. **Ensemble Method**:
   - Combined the predictions of multiple models to create a more robust classification system.

4. **Evaluation**:
   - **Accuracy**, **Precision**, and **F1 Score** were the primary metrics used to evaluate model performance.

<br>

## 🏗️ **Model Architectures**

We experimented with the following pretrained models:

1. **ResNet50**: Known for its residual connections, ResNet50 helps solve the vanishing gradient problem, making it highly effective for deep learning tasks.

2. **EfficientNetB1**: This model uses a compound scaling method that uniformly scales network depth, width, and resolution, providing efficient and powerful performance on image classification tasks.

3. **XceptionNet**: Based on depthwise separable convolutions, XceptionNet excels in capturing fine-grained details, making it ideal for distinguishing between closely related bird species.

Each model was fine-tuned using the dataset, and their performance was evaluated based on accuracy, precision, and F1 score.

<br>

Here’s the updated **Installation and Setup** section tailored for using a Colab notebook:

---

Here’s the updated **Installation and Setup** section with the dataset link included:

---

## ⚙️ **Installation and Setup**

To run the project in **Google Colab**, follow these steps:

1. **Open the Colab notebook**:
   - Click on the following link to open the notebook in Google Colab:
     [Wings of Wisdom - Colab Notebook]([https://colab.research.google.com/github/yourusername/wings-of-wisdom/Wings_of_Wisdom.ipynb](https://colab.research.google.com/drive/1oD6Hyy5OlFZts3gI-qNRI8htISD_kXHp?usp=drive_link))

2. **Download and Upload Dataset**:
   - Download the dataset from the following link:
     [Download Dataset]([https://drive.google.com/file/d/1TisGEJyWvnqMngTdkAbtjvbySBQ2oxYt/view?usp=drive_link](https://drive.google.com/file/d/1TisGEJyWvnqMngTdkAbtjvbySBQ2oxYt/view?usp=drive_link))
   - Once downloaded, upload the dataset to your Colab environment. Click on the **Files** tab in Colab, then **Upload** the dataset files.

   **Alternatively, Mount Google Drive**:
   - Mount your Google Drive to access the dataset directly from there:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Ensure that the dataset is organized in Google Drive as follows:
     ```
     /content/drive/MyDrive/wings-of-wisdom/data/
       ├── train/
       ├── test/
       ├── labels.csv
     ```


4. **Run the code**:
   - Execute the cells in the Colab notebook to **train** the model, **evaluate** performance, and **make predictions**.

---

This setup guide ensures that you can easily run the project on Google Colab by downloading the dataset and installing the necessary dependencies.

<br>

## 📊 **Results**

The best model (EfficientNetB1) achieved the following performance:

- **Accuracy**: 95.99%
  
The ensemble approach slightly boosted these numbers, improving the model’s ability to generalize to new bird species images.

<br>

## 🤯 **Challenges and Learning**

The most exciting and challenging part of this project was **fine-tuning** the models to achieve high accuracy while avoiding overfitting. The nature of the dataset, with birds that look similar, made it difficult to distinguish between species. However, working with **bounding boxes** and experimenting with **data augmentation** helped improve the model's accuracy significantly.

<br>

## 🔭 **Future Work**

In the future, the project can be improved by:
- **Implementing Object Detection**: Use models like YOLO to simultaneously classify and localize bird species.
- **Incorporating Additional Data**: Using external datasets to further enhance the model's ability to classify rare or endangered species.
- **Deploying as a Web App**: Implement the classifier as a web service where users can upload images and get instant predictions.

<br>

## 👨‍💻 **Contributing**

We welcome contributions from the community! Feel free to submit a pull request or open an issue for any bugs or feature requests.

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature-branch`)
3. **Commit your changes** (`git commit -am 'Add some feature'`)
4. **Push to the branch** (`git push origin feature-branch`)
5. **Create a pull request**

<br>


