import json
import requests
import numpy as np
from PIL import Image
import streamlit as st


URL = 'http://localhost:8501/v1/models/house-price:predict'

def image_loader(image_path):
    img = Image.open(image_path)
    img.load()
    img = img.resize((128, 128))
    data = np.asarray(img, dtype='float32')
    return np.reshape(data, (128, 128, 3))

def create_instances(area,
                     bathrooms,
                     bedrooms,
                     zipcode,
                     img_frontal,
                     img_bedroom,
                     img_kitchen,
                     img_bathroom):

    loads = {
        'area': [area],
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'zipcode': [zipcode],
        'img_frontal': img_frontal.tolist(),
        'img_bedroom': img_bedroom.tolist(),
        'img_kitchen': img_kitchen.tolist(),
        'img_bathroom': img_bathroom.tolist()
        }
    return loads

@st.cache
def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": [instances]})
    headers = {"content-type": "application/json"}
    json_response = requests.post(URL, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

def main():
    st.title('🏠 House Price Prediction')
    # # Upload images
    # st.subheader('Upload your images')
    col1, col2 = st.beta_columns(2)
    upload_frontal_img = col1.file_uploader("Frontal Image", 
                                type=["jpg", "png", "jpeg"])
    upload_bedroom_img = col1.file_uploader("Bedroom Image", 
                                type=["jpg", "png", "jpeg"])
    upload_kitchen_img = col2.file_uploader("Kitchen Image", 
                                type=["jpg", "png", "jpeg"])
    upload_bathroom_img = col2.file_uploader("Bathroom Image", 
                                type=["jpg", "png", "jpeg"])

    zipcode_choices = [62234, 81524, 85255, 85262, 85266, 91901, 92021, 92276, 92677,
       92802, 92880, 93111, 93446, 93510, 94501, 94531, 95220, 96019]
    # Fill data
    st.sidebar.title('Fill data')
    n_area = st.sidebar.number_input('Area(s)', value=999,
                                min_value=0, max_value=9999)
    n_bedrooms = st.sidebar.number_input('Bedroom(s)', value=1,
                                min_value=0, max_value=10)
    n_bathrooms = st.sidebar.number_input('Bathroom(s)', value=1,
                                min_value=0, max_value=10)
    n_zipcode = st.sidebar.selectbox('Zipcode',
                                options=zipcode_choices)

    if upload_frontal_img is not None:
        img_frontal = image_loader(upload_frontal_img)
    if upload_bedroom_img is not None:
        img_bedroom = image_loader(upload_bedroom_img)
    if upload_kitchen_img is not None:
        img_kitchen = image_loader(upload_kitchen_img)
    if upload_bathroom_img is not None:
        img_bathroom = image_loader(upload_bathroom_img)
    
    uploaded_images = [upload_frontal_img, upload_bedroom_img,
                       upload_kitchen_img, upload_bathroom_img]
    
    if all(x is not None for x in uploaded_images):
        if st.button('Predict'):
            instance = create_instances(area=n_area,
                                        bathrooms=n_bedrooms,
                                        bedrooms=n_bathrooms,
                                        zipcode=n_zipcode, 
                                        img_frontal=img_frontal,
                                        img_bedroom=img_bedroom,
                                        img_kitchen=img_kitchen, 
                                        img_bathroom=img_bathroom)
            with st.spinner('Predicting...'):
                pred = make_prediction(instance)
                st.subheader('💵 Predicted price: ${}'.format(int(pred[0][0] * 2500000)))

    st.sidebar.text('TensorFlow with multiple input.')

if __name__ == "__main__":
    main()