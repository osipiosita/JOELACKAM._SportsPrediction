import streamlit as st
import pickle 
import numpy as np
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error  


with open ('best_model2.pkl', 'rb') as f:
    model = pickle.load(f)

#featuress required for the model
features = ['movement_reactions', 'mentality_composure', 'potential',
            'wage_eur', 'value_eur', 'passing', 'attacking_short_passing',
            'mentality_vision', 'international_reputation', 'skill_long_passing',
            'power_shot_power', 'physic', 'age', 'skill_ball_control', 'dribbling',
            'shooting', 'skill_curve', 'power_long_shots']

st.title('FIFA Player Rating Prediction')

data = []
for feature in features:
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=10.0, key=f'{feature}_input')
    data.append(value)

if st.button('Predict'):
    data = np.array(data).reshape(1,-1)
    prediction = model.predict(data)

    confidence_score = 0.95 # R2 score obtained after training random forest regressor model
    st.write(f'Predicted Player Rating: {prediction[0]:0.2f}')
    st.write(f'Confidence Score (R-squared): {confidence_score:.4f}')
