
# Cats and Dogs Image Classification

TensorFlow-based Convolutional Neural Network (CNN) for the binary classification of images as either cats or dogs. Utilizing the Keras API, the project demonstrates effective image classification through data augmentation, dropout for regularization, and a sequential CNN architecture. 

Trained Over 20,000 Images of Dogs and Cats

### <a href="https://1drv.ms/u/s!Ak5cN7ry7ksshoRPH2X-IYvDu_HHIQ?e=Z9OFNJ">Download</a> the Model Directly to Use it.

### Model Accuracy -> 90%


Dataset Used -> <a>https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset</a>

# How to Run the Model 

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def prepare_custom_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

# Example usage
custom_image_path = 'INPUT_IMAGE_PATH'
prepared_image = prepare_custom_image(custom_image_path)



from tensorflow.keras.models import load_model

# Path to your saved model
model_path = 'MODEL_PATH'
model = load_model(model_path)

prediction = model.predict(prepared_image)
predicted_class = np.where(prediction[0] > 0.5, "Dog", "Cat")
print(f"The image is a {predicted_class}.")


