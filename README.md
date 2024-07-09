Image Classification Web Application

This project implements a web application for image classification using a convolutional neural network (CNN) trained on the CIFAR-10 dataset. The application offers two main functionalities: a web interface for manual image uploads and an API endpoint for programmatic access. Users can upload images through the web interface or send them to the API endpoint to receive real-time classification results.

Features
CNN Model: Utilizes TensorFlow/Keras to build and train a CNN model on the CIFAR-10 dataset.

Web Interface: Allows users to upload images via a user-friendly web page and view classification results.

API Endpoint: Provides a RESTful API endpoint for uploading images programmatically and receiving classification predictions.

Data Visualization: Uses Matplotlib to visualize the training accuracy and loss, aiding in model evaluation and optimization.

Client-Side Integration: Includes a Python script (use_api.py) for interacting with the API endpoint and retrieving classification results.

Technologies Used
Backend: Python, TensorFlow/Keras, Flask

Frontend: HTML, CSS, JavaScript

Visualization: Matplotlib

HTTP Requests: Requests library
