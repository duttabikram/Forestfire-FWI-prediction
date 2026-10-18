#  Forest Fire FWI Prediction

A Machine Learning project for predicting the **Fire Weather Index (FWI)** using meteorological and environmental conditions.

The project uses a trained machine learning model to estimate the Fire Weather Index and provides a simple **Streamlit web application** for making predictions interactively.

---

## 📌 Overview

Forest fires can cause significant environmental and economic damage. Predicting the likelihood and intensity of fire conditions using weather and environmental parameters can help in early detection and risk assessment.

This project focuses on predicting the **Fire Weather Index (FWI)** from relevant weather conditions using Machine Learning.

The complete workflow includes:

- Data preprocessing
- Exploratory Data Analysis
- Feature selection and engineering
- Machine Learning model training
- Model evaluation
- Model serialization
- Streamlit-based prediction application

---

## 🎯 Objectives

- Analyze weather and environmental factors related to forest fires.
- Build a Machine Learning model for FWI prediction.
- Evaluate the performance of the trained model.
- Deploy the trained model through an interactive Streamlit application.
- Allow users to provide environmental parameters and obtain an FWI prediction.

---

## 🧠 What is FWI?

The **Fire Weather Index (FWI)** is a weather-based index used to estimate the potential for forest fire behavior based on environmental conditions.

It is influenced by factors such as:

- Temperature
- Relative humidity
- Wind speed
- Rainfall
- Fine fuel moisture
- Duff moisture
- Drought conditions

A higher FWI generally indicates more favorable conditions for fire development and spread.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Web Application
- Streamlit

### Development Tools
- Jupyter Notebook
- Git
- GitHub

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Selection
     │
     ▼
Train / Test Split
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Best Model
     │
     ▼
Model Serialization
     │
     ▼
Streamlit Application
     │
     ▼
FWI Prediction
```

---

## 📂 Project Structure

```text
Forestfire-FWI-prediction/
│
├── dataset/
│   └── Dataset files
│
├── models/
│   └── Trained ML model files
│
├── notebook/
│   └── Model development and analysis notebooks
│
├── application.py
│   └── Streamlit prediction application
│
├── requirements.txt
│   └── Project dependencies
│
├── Screenshot 2025-07-24 034722.png
├── Screenshot 2025-07-24 034749.png
│
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/duttabikram/Forestfire-FWI-prediction.git
```

### 2. Navigate to the project directory

```bash
cd Forestfire-FWI-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run application.py
```

The application will open in your browser.

You can then enter the required environmental parameters and generate an **FWI prediction**.

---

## 🖥️ Application

The project includes an interactive Streamlit interface where users can provide the required input parameters.

The application loads the trained Machine Learning model and uses the supplied environmental conditions to generate the predicted FWI value.

### Application Screenshot

![Forest Fire FWI Prediction](Screenshot%202025-07-24%20034722.png)

### Prediction Interface

![FWI Prediction Application](Screenshot%202025-07-24%20034749.png)

---

## 📊 Model Development

The Machine Learning model was developed using the following general process:

1. Load the forest fire dataset.
2. Analyze the available features.
3. Perform data preprocessing.
4. Explore relationships between weather conditions and FWI.
5. Split the dataset into training and testing sets.
6. Train Machine Learning models.
7. Compare model performance.
8. Select the appropriate model.
9. Save the trained model.
10. Integrate the model into the Streamlit application.

The detailed experimentation and analysis can be found in the `notebook` directory.

---

## 🚀 Future Improvements

Some possible improvements for the project include:

- Improve prediction accuracy with advanced ensemble models.
- Perform more extensive hyperparameter tuning.
- Add real-time weather data through an API.
- Add forest fire risk classification such as **Low / Moderate / High / Extreme**.
- Deploy the Streamlit application publicly.
- Add interactive visualizations for weather conditions.
- Build a dashboard for monitoring fire-risk trends.
- Integrate geographical data and map-based fire-risk visualization.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

See the `LICENSE` file for more information.

---

## 👨‍💻 Author

**Bikram Dutta**

GitHub: [duttabikram](https://github.com/duttabikram)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
