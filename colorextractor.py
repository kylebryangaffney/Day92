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
    
    def __init__(self, image_path, num_colors_to_return=6):
        self.image_path = image_path
        self.num_clusters = 8  # Hard limit of 8 clusters
        self.num_colors_to_return = num_colors_to_return
        self.image_array = None
        self.pixels = None
        self.most_common_colors = None

    def load_image(self):
        """Load an image from the specified path."""
        try:
            image = Image.open(self.image_path)
            self.image_array = np.array(image)
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image_array = None

    def preprocess_image(self):
        """Preprocess the image array by ignoring the alpha channel if present."""
        if self.image_array is not None:
            if self.image_array.shape[2] == 4:
                self.image_array = self.image_array[:, :, :3]
            self.pixels = self.image_array.reshape(-1, 3)

    def find_most_common_colors(self):
        """
        Apply K-Means clustering to find the most representative colors.
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
        """Return the most common colors."""
        return self.most_common_colors

    def plot_colors(self):
        """Generate a plot of the most common colors and return it as a base64 string."""
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
