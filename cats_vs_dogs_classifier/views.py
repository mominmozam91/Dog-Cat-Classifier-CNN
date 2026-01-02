from django.shortcuts import render
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO


model = load_model('cats_vs_dogs_classifier/dog_vs_cat_cnn_model.h5')

def index(request):
    prediction = None
    image_url = None
    error_message = None 

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        
        try:
            img = Image.open(img_file)
            img = img.convert('RGB')  

            img.save('media/' + img_file.name)  
            image_url = 'media/' + img_file.name 

            img = img.resize((128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0


            result = model.predict(img_array)[0][0]
            prediction = "Dog" if result > 0.5 else "Cat"
        except Exception as e:
            error_message = f"Error opening or processing image: {e}" 
            prediction = "Error processing image"

    return render(request, 'index.html', {'prediction': prediction, 'image_url': image_url, 'error_message': error_message})