import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import plotly.express as px

st.set_page_config(page_icon="❤️")

# Load the saved XGBoost model
model = pickle.load(open('Heart.pkl', 'rb'))

# List of features that you used during training
trained_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Chest pain type labels
cp_labels = {
    0: 'Typical angina',
    1: 'Atypical angina',
    2: 'Non-anginal pain',
    3: 'Asymptomatic'
}

# Resting electrocardiographic results labels
restecg_labels = {
    0: 'Normal',
    1: 'ST-T wave abnormality',
    2: 'Left ventricular hypertrophy'
}

# Slope of the peak exercise ST segment labels
slope_labels = {
    0: 'Upsloping',
    1: 'Flat',
    2: 'Downsloping'
}

# Thalassemia labels
thal_labels = {
    0: 'Normal',
    1: 'Fixed defect',
    2: 'Reversible defect',
    3: 'Unknown'  # later
}

def create_bar_chart(feature_values, feature_name):
    fig = px.bar(x=['No Heart Disease', 'Heart Disease'], y=feature_values, labels={'x': '', 'y': feature_name},
                 title=f'{feature_name} and Heart Disease')
    fig.update_layout(template='plotly_white')  # Set light background color
    return fig

def create_pie_chart(feature_values, feature_name):
    fig = px.pie(values=feature_values, names=['No Heart Disease', 'Heart Disease'], 
                 title=f'Distribution of {feature_name} in Heart Disease')
    fig.update_layout(template='plotly_white')  # Set light background color
    return fig

def main():
    st.title("Heart Disease Prediction App")

    # Sidebar navigation
    selected_page = st.sidebar.radio("Navigate to", ["Home", "Age", "Gender", "Chest Pain", "Resting ECG", "Slope", "Thalassemia"])

    if selected_page == "Home":
        st.write("Welcome to the Heart Disease Prediction App!")
    elif selected_page == "Age":
        st.subheader("How Age Affects the Probability of Heart Disease")
        age_values = list(range(1, 101))
        probabilities = [1 / (1 + ((age - 50) / 10)**2) for age in age_values]  # A simple curve for illustration
        fig = px.line(x=age_values, y=probabilities, labels={'x': 'Age', 'y': 'Probability'}, title='Probability of Heart Disease vs Age')
        fig.update_layout(template='plotly_white')  # Set light background color
        st.plotly_chart(fig)
        st.write("Additional information about age and heart disease.")
    elif selected_page == "Gender":
        st.subheader("How Gender Affects the Probability of Heart Disease")
        gender_values = [0.2, 0.8]  # For illustration, replace with actual values
        gender_chart = create_bar_chart(gender_values, 'Gender')
        st.plotly_chart(gender_chart)
        st.write("Additional information about gender and heart disease.")

        # For Chest Pain
    elif selected_page == "Chest Pain":
        st.subheader("Distribution of Chest Pain Types in Heart Disease")
        cp_values = [0.4, 0.3]  # Replace with actual values
        cp_chart = create_pie_chart(cp_values, 'Chest Pain Types')
        st.plotly_chart(cp_chart)
        st.write("Additional information about chest pain types and heart disease.")

# For Slope
    elif selected_page == "Slope":
        st.subheader("Distribution of Slope Types in Heart Disease")
        slope_values = [0.6, 0.4]  # Replace with actual values
        slope_chart = create_pie_chart(slope_values, 'Slope Types')
        st.plotly_chart(slope_chart)
        st.write("Additional information about slope types and heart disease.")

# For Thalassemia
    elif selected_page == "Thalassemia":
        st.subheader("Distribution of Thalassemia Types in Heart Disease")
        thal_values = [0.2, 0.8]  # Replace with actual values
        thal_chart = create_pie_chart(thal_values, 'Thalassemia Types')
        st.plotly_chart(thal_chart)
        st.write("Additional information about thalassemia types and heart disease.")

    elif selected_page == "Resting ECG":
        st.subheader("Distribution of Resting ECG in Heart Disease")
        restecg_values = [0.4, 0.3]  # For illustration, replace with actual values
        restecg_chart = create_pie_chart(restecg_values, 'Resting ECG')
        st.plotly_chart(restecg_chart)
        st.write("Additional information about resting ECG and heart disease.")
    
    # Add similar blocks for other pages

    # User input
    age = st.slider("Enter your age:", min_value=1, max_value=100, value=50)
    sex = st.radio("Select your gender:", options=["Male", "Female"])
    cp = st.selectbox("Select chest pain type:", options=list(cp_labels.values()))
    trestbps = st.slider("Enter resting blood pressure:", min_value=80, max_value=200, value=120)
    chol = st.slider("Enter serum cholesterol:", min_value=100, max_value=600, value=200)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl:", options=[0, 1])
    restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_labels.values()))
    thalach = st.slider("Enter maximum heart rate achieved:", min_value=70, max_value=220, value=150)
    exang = st.radio("Exercise induced angina:", options=[0, 1])
    oldpeak = st.slider("Enter oldpeak depression induced by exercise:", min_value=0.0, max_value=6.2, value=0.0)
    slope = st.selectbox("Slope of the peak exercise ST segment:", options=list(slope_labels.values()))
    ca = st.slider("Number of major vessels colored by fluoroscopy:", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia:", options=list(thal_labels.values()))

    # Convert categorical inputs to numerical
    sex = 1 if sex == "Male" else 0
    cp_numeric = [key for key, value in cp_labels.items() if value == cp][0]
    restecg_numeric = [key for key, value in restecg_labels.items() if value == restecg][0]
    slope_numeric = [key for key, value in slope_labels.items() if value == slope][0]
    thal_numeric = [key for key, value in thal_labels.items() if value == thal][0]

    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp_numeric],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg_numeric],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope_numeric],
        'ca': [ca],
        'thal': [thal_numeric]
    })

    user_data = user_data[trained_features]

    prediction = model.predict(user_data)

    if st.button("Predict"):
        if prediction[0] == 1:
            st.markdown("<div style='background-color: #FFD2D2; padding: 10px; border-radius: 5px;'><p style='font-weight: bold; color: red;'>Prediction: <strong>Heart Disease Present</strong></p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color: #D2FFD2; padding: 10px; border-radius: 5px;'><p style='font-weight: bold; color: green;'>Prediction: <strong>No Heart Disease</strong></p></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
