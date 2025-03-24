# Dicoding Final Project - Image Classification Project

- Created by : Muhammad Hafizh Dzaki
- Last Edited On : March 2025

This project was created to fulfill the completion of the **Machine Learning Development** course in the **Data Scientist** learning path on [Dicoding](https://www.dicoding.com).

This project is the **second** project in **Machine Learning Development** course

**Data Scientist** learning path can be found [here](https://www.dicoding.com/learningpaths/60)

**Machine Learning Development** course can be found [here](https://www.dicoding.com/academies/185-belajar-pengembangan-machine-learning)

## Introduction

Predict an image between **Smoke**, **Cloud**, or **Other**. Dataset that used in this project can be found [here](https://huggingface.co/datasets/sagecontinuum/smokedataset)

## Tools and Environment Used

- Windows Subsystem Linux 2 (WSL2) environment
- GPU CUDA Acceleration Enabled (Not necessary)
- Visual Studio Code that connects to WSL2
- Tested in :
    - Python 3.12.9
    - Nodejs 22.14.0
    - Npm 11.2.0

## How to Use This Website?

1. Upload any image to the server through input files available
2. Wait a little bit for server processing every process. The time takes depend on input-ed image size
3. The result of prediction will appear below input files
4. The server is using **Memory Storage**. Therefore, no image was permanently stored in server side

## Understanding Folder 

1. assets : An asset for website front-end
2. saved_model : A saved model for smoke classification in format .keras
3. saved_model2 : A saved model for smoke classification in format Tensorflow Saved Model protobuf (.pb)
4. tfjs_model : A saved model in format TensorflowJS, compatible for website deployment
5. tflite_model : A saved model in format TensorflowLite, compatible for mobile deployment

## Understanding Files

1. main.ipynb : A main file for doing almost every single thing, included a sequence of data loading, pre-processing, modelling, and evaluating
2. convert.ipynb : A file for doing convertion task from Keras or Protobuf format into TensorflowJS and TensorflowLite
3. custom_function.py : An overall function used when completing Dicoding Machine Learning Development Course
4. custom_server.js : Server-side or back-end to deploy model, receive request, process, and give respond of website application
5. index.html : Client-side or front-end with simple UI for user can interacting with