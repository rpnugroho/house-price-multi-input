# House Price Prediction

Predict house price with multiple input: structured data and images.\
![Image of demo](/images/demo.gif)

## Dataset
All dataset can be obtain from this [repository](https://github.com/emanhamed/Houses-dataset).\
The dataset contains 535 instances, 4 images for each house and a text file. Each row in text file represents number of bedrooms, bathrooms, area of the house, zipcode and the price.\
We map all images and create a CSV file. We also remove rare zipcode and outliers from the dataset. You can see this process in data preparation notebook.

## Model
We use Keras Functional API and some experimental feature in preprocessing layers.\
There are 5 extractor model, 1 MLP to handle structured data and 4 CNN to handle image data. We concat all outputs from this extractor and create predictions.
![Image of architecture](/images/model.png)
\
After training for 150 epochs we got **MAPE 27%** on validation data. You can see this process in modeling notebook.

## Deployment
We use TensorFlow Serving to deploy our model.
```cmd
docker pull tensorflow/serving:latest

docker run -p 8501:8501 \
	--name tfserving_house_price \
	--mount type=bind, \
	source=PATH/house-price-multi-input/models/, \
	target=/models/house-price/ \
	-e MODEL_NAME=house-price \
	-t tensorflow/serving
```
and create prediction using this script.
```cmd
INSTANCES = {
        'area': [area],
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'zipcode': [zipcode],
        'img_frontal': [img_frontal],
        'img_bedroom': [img_bedroom],
        'img_kitchen': [img_kitchen],
        'img_bathroom': [img_bathroom]
}

curl -X POST \
	http://localhost:8501/v1/models/house-price:predict \
	-d "{"signature_name": "serving_default", "instances": [INSTANCES]}"
```

## Demo
We also create app demo using streamlit:
```cmd
streamlit run app.py
```

## Reference
- H. Ahmed E. and Moustafa M. (2016). House Price Estimation from Visual and Textual Features. In Proceedings of the 8th International Joint Conference on Computational Intelligence (IJCCI 2016)ISBN 978-989-758-201-1, pages 62-68. DOI: 10.5220/0006040700620068
- https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/