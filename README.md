# Doodle Digit Classifier (React + TensorFlow.js)

A fully client-side handwritten digit classifier built with React, Vite, and TensorFlow.js.  
Includes real-time prediction and PCA visualization of CNN embeddings.

Live Demo: https://yourusername.github.io/doodle-classifier/

---

## Features

• Draw digits directly in the browser  
• Real-time prediction using pretrained CNN  
• Logistic regression model support  
• Extracts CNN penultimate-layer embeddings  
• PCA visualization of learned feature space  
• Runs entirely client-side (no backend)  
• GitHub Pages compatible  

---

## Architecture

Pipeline:

Canvas → Preprocess → CNN → Embedding → PCA → Visualization

Models loaded from:

public/model/pretrained-cnn/
public/model/logreg/

---

## Embedding Visualization

The CNN penultimate dense layer produces a 128-dimensional embedding vector representing each digit.

PCA reduces this to 2D for visualization.

Similar digits cluster together in embedding space.

This allows direct visualization of neural network feature representations.

---

## Tech Stack

React  
Vite  
TensorFlow.js  
Canvas API  
Pure client-side ML  

---

## Running locally

```bash
npm install
npm run dev