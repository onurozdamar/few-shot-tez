import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import cv2
import os
from sklearn.metrics import confusion_matrix
import random
import seaborn as sns
from skimage.transform import rotate
from skimage.util import random_noise


class FewShot:
    def __init__(self, train_directory, test_directory):
        self.image_width = 100
        self.image_height = 100
        self.support_files = self.get_image_paths(train_directory)
        self.query_files = self.get_image_paths(test_directory)
        self.support_images = self.create_images(
            self.support_files, self.image_width, self.image_height
        )
        self.query_images = self.create_images(
            self.query_files, self.image_width, self.image_height
        )
        self.support_labels = self.create_labels(self.support_files)
        self.query_labels = self.create_labels(self.query_files)
        self.support_set = np.stack([image.flatten() for image in self.support_images])
        self.query_set = np.stack([image.flatten() for image in self.query_images])
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.predicted_labels = None
        self.confusion_matrix = None

    def get_image_paths(self, directory):
        image_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)
        return image_paths

    def create_images(self, files, width, height):
        images = []
        for file in files:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (width, height))
            images.append(image)
            rotated_image = rotate(image, angle=15, mode="wrap")
            noisy_image = random_noise(image, var=0.01**2)
            images.append(rotated_image)
            images.append(noisy_image)

        return images

    def create_labels(self, files):
        labels = []
        for file in files:
            label = os.path.basename(os.path.dirname(file))
            labels.append(label)
            labels.append(label)
            labels.append(label)

        return labels

    def train_model(self):
        self.model.fit(self.support_set, self.support_labels)

    def predict_labels(self):
        self.predicted_labels = self.model.predict(self.query_set)

    def generate_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(
            self.query_labels, self.predicted_labels
        )

    def show_confusion_matrix(self):
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=self.model.classes_,
            yticklabels=self.model.classes_,
        )
        accuracy = np.trace(self.confusion_matrix) / float(
            np.sum(self.confusion_matrix)
        )
        plt.title("Confusion Matrix\nAccuracy: {:.2f}%".format(accuracy * 100))
        plt.xlabel("Tahmin Edilen Sınıf")
        plt.ylabel("Gerçek Sınıf")
        plt.show()