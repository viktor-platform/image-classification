# Bringing Your Image Classification Model to Life with VIKTOR: A Tutorial

Welcome to this tutorial on setting up an Image Classification Convolution Neural Network within a VIKTOR app. 
The VIKTOR platform is a powerful tool for deploying and managing machine learning models, and in this tutorial, 
we will be showing you how to set up a pre-trained image classification model within a VIKTOR app. Although this 
tutorial will not focus on the training of the model, we will briefly mention the steps required to train the model 
on the CIFAR-10 dataset, an image dataset containing 10 classes of various objects such as airplanes, cars, 
and birds. Using TensorFlow, an open-source machine learning library, to build and train the model.

By the end of this tutorial, you will have a working image classification model deployed within a VIKTOR app, 
and be able to use the model to classify images in real-time. This tutorial is perfect for engineers and developers 
who want to use pre-trained models to add image classification functionality to their projects.

Let's get started!

## Step 1: Install VIKTOR
For one to be able to develop a VIKTOR app, you need a VIKTOR developer account. If you do not have one yet, visit the 
["Start building apps"](https://www.viktor.ai/start-building-apps) page on the VIKTOR website and follow the 
instructions.

In case you are new to VIKTOR, it is recommended to also follow [the tutorial](https://docs.viktor.ai/docs/getting-started/create-first-app) 
provided.

Set your cli to use `venv` as isolation mode.

## Step 2: Train model (Optional)
As the tutorial focuses on implementing a pre-trained image classification model, this step is optional.

In case you are interested in generating a model yourself, follow the instructions in this 
[GeeksforGeeks article](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/), on which this 
tutorial is based.

A local `pip install tensorflow` in your used `venv` should be adequate to start using TensorFlow.

**However, there already are some models available that you can immediately use in your app (generated with version 2.11.0 of tensorflow).
In case you want to use these (which can be found in the folder [`cnn_model`](./cnn_model/)), move on to the next step.**

The result is a Convolution Neural Network that should be able to classify images into the following classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

The result is a .h5 file that can be loaded and used in a VIKTOR application.

**It is strongly advised to run this locally and not in a VIKTOR environment, as it can be time-consuming depending 
on your computer and the amount of epochs you're using (start with 5).**
You can use the example [code provided](./cnn_model/cnn_cifar10_example.py)) to generate your own models.

## Step 3: Create a VIKTOR app
Now, with a machine-learning model available, you can now set up your application.

- Create a new (editor-type) app and add `tensorflow-cpu==2.11.0` to the `requirements.txt`
- Add .h5 model files to your app folder


- Add parametrization that allows the uploading of JPEG/JPG/PNG images
- Add parametrization which allows user to select different models (_optional_: one model will also do)

Here is some example code:

```python
MODEL_OPTIONS = [
    OptionListElement(label="CNN (2 epochs)", value="cnn_cifar10_model_2_epochs.h5"),
    OptionListElement(label="CNN (5 epochs)", value="cnn_cifar10_model_5_epochs.h5"),
    OptionListElement(label="CNN (10 epochs)", value="cnn_cifar10_model_10_epochs.h5"),
]


class Parametrization(ViktorParametrization):
    image = FileField("Upload a JPEG/JPG or PNG image", file_types=[".jpeg", ".jpg", ".png"])
    model = OptionField("Select CNN model", options=MODEL_OPTIONS, default=MODEL_OPTIONS[0].value)
```


- Write a function that preprocesses the uploaded image to tensor: normalize the RGB values and resize to (32, 32) pixels. 
Use `tf.image` functionalities to do this. Finish with `img = tf.expand_dims(img, axis=0)` in your conversion to align the dimensions of your image and the model.
```python
def preprocess_image(unclassified_image_bytes: bytes):
    img = tf.image.decode_jpeg(unclassified_image_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 255.0  # normalize RGB values
    size = (32, 32)  # size depending on size used in model, trained model is 32 x 32 pixels
    img = tf.image.resize(img, size)
    img = tf.expand_dims(img, axis=0)  # expand dimensions of tensor
    return img
```


- Create a view with results (upload image and predicted classes + probabilities for example) within the Controller. 
- Define the logic to load the model and predict the uploaded images using the `model.predict()` with the preprocessed 
image tensor as input to predict the class:
```python
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

...
class Controller(ViktorController):
    ...

    @ImageAndDataView("Image", duration_guess=3)
    def visualize_image_and_show_classification_results(self, params: Munch, **kwargs) -> ImageAndDataResult:
        # Load model
        model = tf.keras.models.load_model(params["model"])
    
        # Preprocess uploaded image
        unclassified_image_tensor = preprocess_image(params.image.file.getvalue_binary())
    
        # Predict
        pred = model.predict(unclassified_image_tensor)
        results = [(class_name, prob) for class_name, prob in sorted(zip(CLASSES, pred.tolist()[0]), key=lambda x: x[1], reverse=True)]
    
        # Generate results
        data = [DataItem(label=class_name.title(), value=prob * 100, suffix="%", number_of_decimals=2) for class_name, prob in results]
        return ImageAndDataResult(image=params.image.file, data=DataGroup(*data))
```
- Find some JPEGS/JPG/PNG images online and test it out!



**Congratulations, you are now able to deploy a locally trained Image Classification model!**

## How could this be used in practice?
Computer vision and image recognition can be used to help engineers in a variety of ways, such as:
- Quality control: Engineers can use image recognition to inspect products for defects, such as scratches or cracks, 
during the manufacturing process, or to check diagrams for errors or anomalies.
- Inspection: Engineers can use computer vision to inspect infrastructure, such as bridges or pipelines, for signs of 
wear or damage.
- Medical imaging: Engineers can use image recognition to improve the diagnostic accuracy of medical imaging systems 
such as x-ray, CT, and MRI.
- Smart agriculture: Engineers can use computer vision to monitor crop growth, detect pests, and make decisions about 
irrigation, fertilization, and harvesting
