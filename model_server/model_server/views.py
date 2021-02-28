from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from PIL import Image
import os
from keras.preprocessing import image


def demo_ui(request):
    return render(request, "demo.html")


def predict_api(request):
    model = settings.MODEL

    # temporarily store the uploaded file into the current dir
    file_upload = request.FILES['file_upload']
    with open(file_upload.name, 'wb') as f:
        for chunk in file_upload.chunks():
            f.write(chunk)

    # read the stored uploaded file in the current dir and predict
    img = Image.open(file_upload.name)
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    y = model.predict(x)
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    predict = labels[np.argmax(y)]

    # remove the file after the prediction is done
    os.remove(file_upload.name)

    return JsonResponse({'predict': predict})

