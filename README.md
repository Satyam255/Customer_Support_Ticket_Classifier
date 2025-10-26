
# Customer Support Ticket Classification Web App

**Full-Stack AI/ML Project**  
Classifies banking customer support tickets into **Billing Question**, **Technical Issue**, or **General Inquiry** using a fine-tuned DistilBERT model and a modern web interface.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Problem Statement / Objective](#problem-statement--objective)  
3. [Project Architecture & Data Flow](#project-architecture--data-flow)  
4. [Libraries & Tools Used](#libraries--tools-used)  
5. [Implementation Details](#implementation-details)  
6. [Setup & Running the Project](#setup--running-the-project)  
7. [API Pipeline](#api-pipeline)  
8. [Evaluation](#evaluation)  
9. [Challenges](#challenges)  
10. [Future Enhancements](#future-enhancements)  

---

## Project Overview
This project implements a **full-stack AI/ML application** for classifying banking customer support tickets. The backend uses a **fine-tuned DistilBERT** model for text classification, and the frontend is a responsive **React (Vite) web app**. A **Node.js server acts as a proxy** between the frontend and Flask AI server for API calls and CORS handling.

---

## Problem Statement / Objective
Automate the classification of banking support tickets into three categories:

- **Billing Question**
- **Technical Issue**
- **General Inquiry**

**Input Example:**  
```

"The ATM did not dispense cash but my account was debited."

```

**Output Example:**  
```

Billing Question

```

The system aims to **streamline ticket routing**, reduce manual intervention, and improve customer support efficiency.

---

## Project Architecture & Data Flow

```

```
        +--------------------+
        |  React Frontend    |
        |  (Port 5137)       |
        +---------+----------+
                  |
    POST /api/classify (ticket text)
                  |
                  v
        +--------------------+
        |  Node.js Server    |
        |  (Proxy, Port 3000)|
        +---------+----------+
                  |
    POST /classify (forward to Flask)
                  |
                  v
        +--------------------+
        |  Flask AI Server   |
        |  (Port 5000)       |
        +---------+----------+
                  |
        Response: JSON { label, score }
                  |
                  v
        +--------------------+
        |  Node.js Server    |
        |  (forwards JSON)   |
        +---------+----------+
                  |
                  v
        +--------------------+
        |  React Frontend    |
        |  Displays result   |
        +--------------------+
```

````

---

## Libraries & Tools Used

| Component           | Libraries / Tools Used |
|--------------------|-----------------------|
| Transformer Model   | Hugging Face Transformers, Accelerate |
| Data Handling       | pandas, Hugging Face Datasets |
| Traditional NLP     | NLTK |
| Deep Learning       | PyTorch |
| Evaluation Metrics  | scikit-learn (Accuracy, F1-score), evaluate |
| Web Frontend        | React.js, Vite, Tailwind CSS |
| Web Backend         | Node.js (Express), Flask (Python) |
| API Communication   | Axios, Flask-CORS |
| Visualization       | Matplotlib |

---

## Implementation Details

### 1. NLP & Preprocessing
- **Sentence Splitting:** Tokenizer handles splitting sentences.  
- **Tokenization:** Sub-word tokenization using `distilbert-base-uncased`.  
- **Lemmatization:** Base form of words using NLTK.  
- **POS Tagging:** Optional syntactic analysis.  
- **Semantic Analysis:** Transformers capture meaning, reducing need for traditional WordNet methods.

### 2. LLM Fine-Tuning
- **Model:** `distilbert-base-uncased`  
- **Dataset:** Banking77 mapped to 3 categories  
- **Training Parameters:**  
  - Epochs: 3  
  - Batch Size: 16  
  - Optimizer: AdamW  
  - Learning Rate: 2e-5  
  - Loss: CrossEntropyLoss  

### 3. API & Web Development
- **Flask AI Server:** Exposes `/classify` POST endpoint.  
- **Node.js Proxy Server:** Handles `/api/classify` POST request from React, forwards to Flask, and returns JSON.  
- **React Frontend:** Responsive UI with loading spinner, result badge, and confidence score.  

---

## Setup & Running the Project

### 1. Backend (Flask AI Server)
```bash
cd flask-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
````

### 2. Node.js Server

```bash
cd node-server
npm install
node server.js
```

### 3. Frontend (React + Vite)

```bash
cd client
npm install
npm run dev
```

Visit: [http://localhost:5137](http://localhost:5137)

---

## API Pipeline

### **Frontend → Node → Flask**

* **POST /api/classify** (React → Node)

```json
{
  "text": "My internet is not working"
}
```

* Node forwards to Flask: **POST /classify**
* Flask returns:

```json
{
  "label": "Technical Issue",
  "score": 0.97
}
```

* Node sends the same JSON back to React for display.

---

## Evaluation Metrics

* **Accuracy:** Percentage of correctly predicted tickets.
* **F1-Score:** Balances precision and recall for robust evaluation.

---

## Challenges

* **Data Cleaning:** Mapping 77 categories to 3 caused NaN values.
* **Computational Constraints:** Training on CPU was slow; GPU recommended.
* **API Pipeline:** Ensuring CORS and proxy routing between Node and Flask.

---

## Future Enhancements

* Deploy using **Docker** for containerized full-stack setup.
* Add **real-time streaming classification**.
* Extend model to handle **multi-label classification**.
* Implement **user authentication** for admin dashboards.
* Add **analytics & reporting** of ticket distribution and trends.



