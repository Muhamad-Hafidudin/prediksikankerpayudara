import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def predict_cancer(new_data):
    # Load dataset
    df = pd.read_csv('data.csv')
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1, inplace=False)
    y = df['diagnosis']

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(new_data)
    return predictions

# Main app
def main():
    # Title and description
    st.title('Web Apps - Prediksi Kanker Payudara')
    st.write('Aplikasi ini memprediksi kanker payudara berdasarkan data masukan yang diberikan.')

    # Load image
    image = Image.open('kankerpayudara.png')

    # Display image
    st.image(image, caption='Aplikasi Web untuk Prediksi Kanker Payudara Ini dibuat oleh Kelompok 6 Manajemen Proyek Universitas Alma Ata: Muhamad Hafidudin, Inayah Khasna Putri Afifah, Salis Nizar Komaruzaman, Afifah Indrawati', use_column_width=True)

    # Sidebar
    st.sidebar.header('New Data')
    new_data = {}
    df = pd.read_csv('data.csv')
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1, inplace=False)
    
    # Iterating through columns of X
    for column in X.columns:
        new_data[column] = st.sidebar.number_input(f'Enter {column}', value=0.0)

   # Prediction
    if st.sidebar.button('Predict'):
        new_data = pd.DataFrame([new_data])
        predictions = predict_cancer(new_data)
        if predictions[0] == 1:
            st.write('Predictions: 1 = Pasien terdiagnosis kanker payudara ganas')
        else:
            st.write('Predictions: 0 = Pasien terdiagnosis kanker payudara jinak')

if __name__ == '__main__':
    main()
