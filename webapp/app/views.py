from django.shortcuts import render
import numpy as np 
import tensorflow as tf 
from PIL import Image, ImageOps
from .froms import ImageUploadForm

def handle_upload_file(f):

    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def index(request):
    return render(request,'index.html')
def result(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_upload_file(request.FILES['image'])
    
        model = tf.keras.models.load_model(r'C:\Users\tmachine\Desktop\food\webapp\app\best_model.hdf5')
        file = r'C:\Users\tmachine\Desktop\food\webapp\img.jpg'


        if file is None:
            print('no image')
        else:
            image = Image.open(file)            
            prediction = import_and_predict(image, model)

            class_labels = {
                0: "Apple Pie",
                1: "Bibimbap",
                2: "Cannoli",
                3: "Edamame",
                4: "Falafel",
                5: "French Toast",
                6: "Ice Cream",
                7: "Ramen",
                8: "Sushi",
            }
            predicted_class_index = np.argmax(prediction)
            output=class_labels.get(predicted_class_index, "Unknown Class")
            return render(request, 'result.html', {'data': output})

    return render(request,'result.html')

def import_and_predict(image_data, model):
    size=(75,75)
    image=ImageOps.fit(image_data, size, method=0, bleed=0.0, centering=(0.5, 0.5))
    image=image.convert('RGB')
    image = np.array(image)
    image=(image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction









