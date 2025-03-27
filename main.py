import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def generate_house_data(n_samples=200):
    """Generate house data with realistic size, price, and location factors."""
    np.random.seed(42)
    
    size = np.random.normal(1400, 200, n_samples)  # House size variation
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    
    # Assign price based on size & location impact
    location_price_factor = {'Urban': 1.3, 'Suburban': 1.0, 'Rural': 0.7}  
    price = (size * 120) * np.vectorize(location_price_factor.get)(location) + np.random.normal(0, 5000, n_samples)
    
    data = pd.DataFrame({'Size': size, 'Price': price, 'Location': location})
    return data

def train_model():
    df = generate_house_data(n_samples=300)
    
    # One-Hot Encode Location
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    location_encoded = encoder.fit_transform(df[['Location']])
    
    # Convert to DataFrame & Combine with Features
    location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['Location']))
    X = pd.concat([df[['Size']], location_df], axis=1)
    y = df['Price']
    
    # Normalize the size feature (helps improve prediction accuracy)
    scaler = StandardScaler()
    X[['Size']] = scaler.fit_transform(X[['Size']])
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, encoder, scaler, df  # Returning necessary objects

def main():
    st.title("House Price Prediction (Size & Location)")
    st.sidebar.header("User Input Features")
    
    model, encoder, scaler, df = train_model()
    
    # User Inputs
    size = st.number_input("Size of the house (sq ft)", min_value=500, max_value=5000, value=1500, step=50)
    location = st.selectbox("Select the location", ['Urban', 'Suburban', 'Rural'])
    
    if st.button("Predict Price"):
        # Encode location
        location_encoded = encoder.transform([[location]])  
        
        # Scale input size
        size_scaled = scaler.transform([[size]])[0][0]
        
        # Create final input array
        input_data = np.concatenate(([size_scaled], location_encoded[0])).reshape(1, -1)
        
        # Predict price
        predicted_price = model.predict(input_data)[0]
        
        st.success(f"Predicted price for a {size} sq ft house in {location}: **${predicted_price:,.2f}**")
        
        # Scatter Plot
        fig = px.scatter(df, x='Size', y='Price', color='Location', title="House Price vs Size")
        fig.add_scatter(x=[size], y=[predicted_price], mode='markers', marker=dict(color='red', size=12), name='Predicted Price')
        
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
