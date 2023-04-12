import gradio as gr
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from helper_functions import build_model, convert

# Loading in the Label Encoders
car_name_encoder = LabelEncoder()
car_name_encoder.classes_ = np.load('car_name.npy', allow_pickle=True)
car_names_list = car_name_encoder.classes_.tolist()

fuel_type_encoder = LabelEncoder()
fuel_type_encoder.classes_ = np.load('fuel_type.npy', allow_pickle=True)
fuel_type_list = fuel_type_encoder.classes_.tolist()

seller_type_encoder = LabelEncoder()
seller_type_encoder.classes_ = np.load('seller_type.npy', allow_pickle=True)
seller_type_list = seller_type_encoder.classes_.tolist()

transmission_type_encoder = LabelEncoder()
transmission_type_encoder.classes_ = np.load(
    'transmission_type.npy', allow_pickle=True)
transmission_type_list = transmission_type_encoder.classes_.tolist()

# Function to predict
# # Index(['car_name', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type',
#        'transmission_type', 'mileage', 'engine', 'max_power', 'seats',
#        'avg_selling_price'],
#       dtype='object')


def predict(car_name, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats):

    model = build_model()
    model.load_weights("./checkpoints/model_checkpoint")

    car_name = car_name_encoder.transform([car_name])
    seller_type = seller_type_encoder.transform([seller_type])
    fuel_type = fuel_type_encoder.transform([fuel_type])
    transmission_type = transmission_type_encoder.transform(
        [transmission_type])
    features = np.array([car_name, vehicle_age, km_driven, seller_type, fuel_type,
                         transmission_type, mileage, engine, max_power, seats]).astype('float32')
    prediction = model.predict(features.reshape(1, -1))
    return tf.round(prediction)


# Creating input list
car_name_dropdown = gr.inputs.Dropdown(car_names_list, label="Car Name")
vehicle_age_input = gr.Number(label="Vehicle Age")
km_driven_input = gr.Number(label="Kilometers Driven")
seller_type_dropdown = gr.inputs.Dropdown(
    seller_type_list, label="Seller Type")
fuel_type_dropdown = gr.inputs.Dropdown(fuel_type_list, label="Fuel Type")
transmission_type_dropdown = gr.inputs.Dropdown(
    transmission_type_list, label="Transmission Type")
mileage_input = gr.Number(label="Mileage")
engine_input = gr.Number(label="Engine (in cc) (1 L or litre = 1000cc")
max_power_input = gr.Number(label="Max Power (in bhp)")
seats_input = gr.Number(label="Number of Seats")

input_list = [car_name_dropdown, vehicle_age_input, km_driven_input, seller_type_dropdown,
              fuel_type_dropdown, transmission_type_dropdown, mileage_input, engine_input, max_power_input, seats_input]

# Creating output list
output_list = gr.Number(
    label="Selling price of used car in rupees (eg. 1.1 lakh = 110000)")

title = "Car Price Prediction Application🚘🚙🏎️📊📈"
description = """
<img src= "https://thinkingneuron.com/wp-content/uploads/2020/09/Car-price-prediction-case-study.png" align="center">
This application predicts the selling price of a used car based on the input parameters. The model is trained on data from Cardekho.com. The model is a Random Forest Regressor with 1000 estimators. The model is trained on 80% of the data and tested on 20% of the data. The model has an R2 score of 0.95.

"""

# Build the Gradio app
gr.Interface(fn=predict,
             inputs=input_list,
             outputs=output_list,
             title=title,
             description=description,
             allow_flagging="never").launch()
