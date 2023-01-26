"""Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from pathlib import Path

import tensorflow as tf
from munch import Munch

from viktor import ViktorController
from viktor.core import UserError
from viktor.parametrization import (
    ViktorParametrization,
    FileField,
    OptionListElement,
    OptionField,
    Text,
)
from viktor.views import (
    ImageAndDataView,
    ImageAndDataResult,
    DataGroup,
    DataItem,
    WebView,
    WebResult,
)

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


MODEL_OPTIONS = [
    OptionListElement(label="CNN (2 epochs)", value='cnn_cifar10_model_2_epochs.h5'),
    OptionListElement(label="CNN (5 epochs)", value='cnn_cifar10_model_5_epochs.h5'),
    OptionListElement(label="CNN (10 epochs)", value='cnn_cifar10_model_10_epochs.h5'),
]


def load_model(params: Munch):
    model_path = Path(__file__).parent / 'cnn_model' / params["model"]
    return tf.keras.models.load_model(model_path)


def preprocess_image(unclassified_image_bytes: bytes):
    """Processes images to a tensor. The following is done:
    - The RGB values (ranging from 0-255) are normalized.
    - The image is sized to 32 x 32 pixels.
    - The dimensions of the model and image are aligned.

    :param unclassified_image_bytes: A JPEG/JPG/PNG image in bytes format.
    """
    img = tf.image.decode_jpeg(unclassified_image_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 255.0  # normalize RGB values
    size = (32, 32)  # size depending on size used in model, trained model is 32 x 32 pixels
    img = tf.image.resize(img, size)
    img = tf.expand_dims(img, axis=0)  # expand dimensions of tensor
    return img


class Parametrization(ViktorParametrization):
    text_01 = Text(
        """# Welcome to the Machine Learning Image Classification app!
This application demonstrates how VIKTOR could be used to implement Machine Learning models to help with detecting 
objects within images.

This application is based on the 
[GeeksforGeeks article](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/), 
where a Convolution Neural Network model is trained to be able to classify images into the following classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

**Find some JPEGS/JPG/PNG images online and test it out!**
    """
    )
    image = FileField("Upload a JPEG/JPG or PNG image", file_types=[".jpeg", ".jpg", ".png"])
    model = OptionField("Select CNN model", options=MODEL_OPTIONS, default=MODEL_OPTIONS[0].value)
    text_02 = Text(
        """## How could this be used in practice?
Computer vision and image recognition can be used to help engineers in a variety of ways, such as:
- Quality control: Engineers can use image recognition to inspect products for defects, such as scratches or cracks, 
during the manufacturing process, or to check diagrams for errors or anomalies.
- Inspection: Engineers can use computer vision to inspect infrastructure, such as bridges or pipelines, for signs of 
wear or damage.
- Medical imaging: Engineers can use image recognition to improve the diagnostic accuracy of medical imaging systems 
such as x-ray, CT, and MRI.
- Smart agriculture: Engineers can use computer vision to monitor crop growth, detect pests, and make decisions about 
irrigation, fertilization, and harvesting

**For more information on how this app was made, refer to the "What's next" tab.**
    """
    )


class Controller(ViktorController):
    viktor_enforce_field_constraints = True  # Resolves upgrade instruction https://docs.viktor.ai/sdk/upgrades#U83
    label = "ImageClassifier"
    parametrization = Parametrization

    @ImageAndDataView("Image", duration_guess=3)
    def visualize_image_and_show_classification_results(self, params: Munch, **kwargs) -> ImageAndDataResult:
        """Initiates the process of visualizing the uploaded image and presenting the classification results as
        calculated by the Image Classification Convolution Neural Network model.
        """
        # Load model
        model = load_model(params)

        # Preprocess uploaded image
        if not params.image:
            raise UserError("Upload and select an image first")
        unclassified_image_tensor = preprocess_image(params.image.file.getvalue_binary())

        # Predict
        pred = model.predict(unclassified_image_tensor)
        results = [
            (class_name, prob)
            for class_name, prob in sorted(zip(CLASSES, pred.tolist()[0]), key=lambda x: x[1], reverse=True)
        ]

        # Generate results
        data = [
            DataItem(label=class_name.title(), value=prob * 100, suffix="%", number_of_decimals=2)
            for class_name, prob in results
        ]
        return ImageAndDataResult(image=params.image.file, data=DataGroup(*data))

    @WebView("What's next?", duration_guess=1)
    def whats_next(self, params, **kwargs):
        """Initiates the process of rendering the "What's next" tab."""
        html_path = Path(__file__).parent / "final_step.html"
        with html_path.open(encoding="utf-8") as _file:
            html_string = _file.read()
        return WebResult(html=html_string)
