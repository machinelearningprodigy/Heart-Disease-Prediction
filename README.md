# ❤️ Heart Disease Prediction App

Welcome to the **Heart Disease Prediction App**, a user-friendly tool built with **Streamlit** that helps predict the likelihood of heart disease based on key health indicators. This app utilizes an **XGBoost** model trained on relevant medical features to provide insights into heart disease risk.

## 🚀 Features

- **Interactive UI:** Easy-to-use sliders, radio buttons, and dropdowns for data input.
- **Machine Learning Model:** Uses a trained **XGBoost** classifier to predict heart disease.
- **Data Visualizations:** Dynamic **Plotly** charts to understand how different factors influence heart disease.
- **Responsive Design:** Works seamlessly on both desktop and mobile devices.

## 📊 How It Works

1. Enter your **health details** such as age, gender, blood pressure, cholesterol levels, chest pain type, etc.
2. Click the **"Predict"** button to get an instant prediction.
3. View the result in a color-coded message box:
   - 🟢 **No Heart Disease** (Green background)
   - 🔴 **Heart Disease Present** (Red background)
4. Explore insights on different **risk factors** through visualizations.

## 🛠 Installation & Usage

To run this app locally, follow these steps:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/machinelearningprodigy/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2️⃣ Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

## 🏗 Technologies Used

- **Python 3.x**
- **Streamlit** (for the web interface)
- **XGBoost** (for machine learning model)
- **Pandas** (for data processing)
- **Plotly** (for visualizations)

## 📦 Requirements

This project requires the following dependencies:

```txt
pandas==2.0.3
plotly==5.9.0
streamlit==1.30.0
xgboost==2.0.3
```

## 🎯 Future Enhancements

- ✅ Add more **health parameters** to improve prediction accuracy.
- ✅ Implement a **database** to store user history.
- ✅ Enhance **UI design** for a better user experience.

## 🤝 Contributing

Want to improve this project? Feel free to fork, create a new branch, and submit a pull request!

## 📜 License

This project is open-source and available under the **MIT License**.

---

📧 **Have any questions?** Feel free to reach out or raise an issue in the repository!

