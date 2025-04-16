# Emotion Classification App

![Emotion Classification](emotion_picture.png)

## Overview
A web application built with Streamlit that classifies text into emotional categories (anger, joy, or fear) using a Multinomial Naive Bayes machine learning model. The app provides real-time predictions with confidence scores

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example Outputs](#example-outputs)
- [DataSet](#dataset)


## Features
- 🎭 Emotion classification for text input
- ⚡ Real-time predictions
- 📊 Displays prediction confidence
- 🖥️ Simple and clean user interface
- 🤖 Powered by scikit-learn's Naive Bayes

## Installation

1. Clone the repository:
   ```python
   git clone https://github.com/ShahdTarek4/AIFinalproject-Emotion_Classification
   cd emotion-classification-app
    ```

2. Install Dependencies:
   ```python
   pip install -r requirements.txt
   ```

3. Run application:
   ```python
   streamlit run emotion_prediction.py
   ```

## Usage:
1.Type or paste your text in the input box

2.Click the "Predict" button

3.View the emotion classification result

## Example Outputs:
1."I'm so excited!" → Joy

2."This makes me furious" → Angry

3."I'm scared of the dark" → Fear

## DataSet:
-Source: Emotion_classify_Data.csv

-Features: Text comments

-Target classes: anger, joy, fear

-Train-Test Split: 75%-25%



   
