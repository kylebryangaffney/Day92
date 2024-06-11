import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class ColorExtractor:
    """
    A class used to extract the most common colors from an image using K-Means clustering.

    Attributes
    ----------
    image_path : str
        Path to the input image file.
    num_colors_to_return : int
        Number of most common colors to return (default is 6).
    num_clusters : int
        The number of clusters to form as well as the number of centroids to generate (default is 8).
    image_array : np.ndarray
        Array representation of the loaded image.
    pixels : np.ndarray
        Reshaped array of image pixels.
    most_common_colors : list
        List of tuples representing the most common colors in the image.

    Methods
    -------
    load_image():
        Loads an image from the specified path into an array.
    preprocess_image():
        Preprocesses the image array by ignoring the alpha channel if present and reshaping the pixels array.
    find_most_common_colors():
        Applies K-Means clustering to find the most representative colors in the image.
    get_colors():
        Returns the most common colors found in the image.
    plot_colors():
        Generates a plot of the most common colors and returns it as a base64 string.
    """

    def __init__(self, image_path, num_colors_to_return=6):
        """
        Parameters
        ----------
        image_path : str
            Path to the input image file.
        num_colors_to_return : int
            Number of most common colors to return (default is 6).
        """
        self.image_path = image_path
        self.num_clusters = 8  # Hard limit of 8 clusters
        self.num_colors_to_return = num_colors_to_return
        self.image_array = None
        self.pixels = None
        self.most_common_colors = None

    def load_image(self):
        """
        Loads an image from the specified path into an array.

        This method attempts to open the image file located at `self.image_path` and convert it into a NumPy array.
        If an error occurs during this process, it prints an error message and sets `self.image_array` to None.

        Raises
        ------
        Exception
            If there is an error loading the image file.
        """
        try:
            image = Image.open(self.image_path)
            self.image_array = np.array(image)
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image_array = None

    def preprocess_image(self):
        """
        Preprocesses the image array by ignoring the alpha channel if present and reshaping the pixels array.

        This method ensures that the image array only contains RGB channels (ignoring alpha channel if present).
        It then reshapes the image array into a 2D array where each row is a pixel and each column is a color channel (R, G, B).
        """
        if self.image_array is not None:
            if self.image_array.shape[2] == 4:
                self.image_array = self.image_array[:, :, :3]
            self.pixels = self.image_array.reshape(-1, 3)

    def find_most_common_colors(self):
        """
        Applies K-Means clustering to find the most representative colors in the image.

        This method uses K-Means clustering to cluster the image pixels into `self.num_clusters` clusters.
        It then finds the `self.num_colors_to_return` most common clusters (colors) based on the number of pixels in each cluster.

        Raises
        ------
        ValueError
            If `self.num_colors_to_return` is greater than `self.num_clusters`.
        """
        if self.pixels is not None:
            if self.num_colors_to_return > self.num_clusters:
                raise ValueError("num_colors_to_return must be less than or equal to num_clusters")
            
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(self.pixels)
            cluster_centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            label_counts = Counter(labels)
            most_common_labels = label_counts.most_common(self.num_colors_to_return)
            self.most_common_colors = [tuple(cluster_centers[label]) for label, _ in most_common_labels]
        else:
            print("No pixels to process. Ensure the image is loaded and preprocessed correctly.")
    
    def get_colors(self):
        """
        Returns the most common colors found in the image.

        Returns
        -------
        list of tuple
            A list of tuples representing the most common colors in the image.
        """
        return self.most_common_colors

    def plot_colors(self):
        """
        Generates a plot of the most common colors and returns it as a base64 string.

        This method creates a matplotlib plot of the most common colors extracted from the image.
        It then converts the plot into a PNG image and encodes it as a base64 string.

        Returns
        -------
        str
            A base64 string representing the PNG image of the most common colors plot.

        Raises
        ------
        Exception
            If there is an error generating the plot.
        """
        if self.most_common_colors is not None:
            plt.figure(figsize=(15, 5))
            for i, color in enumerate(self.most_common_colors):
                color_patch = np.ones((100, 100, 3), dtype=np.uint8) * color
                plt.subplot(1, len(self.most_common_colors), i+1)
                plt.imshow(color_patch)
                plt.title(f'#{i+1}\n{color}')
                plt.axis('off')
            plt.suptitle('Most Common Colors')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            return f"data:image/png;base64,{plot_url}"
        else:
            print("No colors to plot. Ensure the colors are extracted correctly.")
            return None
