import os
import shutil
import glob
from keras import backend as K
K.set_image_data_format('channels_first')
import matplotlib
matplotlib.use('TkAgg')
from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

from flask import Flask, render_template, request

poll_data = {
    'Result' : 1
}

# Creates an instance of the Flask class
app = Flask(__name__)

# MODEL_PATH = 'model/pneumonia_model.h5'
# MODEL_PATH = 'model/20epoches_model.h5'
MODEL_PATH = 'model/vgg16.h5'
# MODEL_PATH = 'model/vgg19.h5'

# app.config["ALL_UPLOADS"] = "static/images/Testing_Image_Folder/Test"
app.config["IMAGE_UPLOADS"] = "Testing_Image_Folder/Test/"
app.config["PAGE_IMAGES"] = "static/images"

# tells Flask what URL should trigger our function
@app.route("/")
def home():
    return render_template('/homePage.html')

@app.route("/about")
def aboutPage():
    return render_template('/aboutPage.html')

@app.route("/test")
def testPage():
    return render_template('/testPage.html')

@app.route("/contact")
def contactUsPage():
    return render_template('/contact.html')

@app.route("/aboutUs")
def aboutUsPage():
    return render_template('/aboutUsPage.html')

@app.route("/ourServices")
def ourServicesPage():
    return render_template('/ourServicePage.html')

@app.route("/privacyPolicy")
def privacyPolicyPage():
    return render_template('/privacyPolicyPage.html')

@app.route("/FaQ")
def faqPage():
    return render_template('/faqPage.html')

@app.route('/footer')
def footer():
    return render_template('footer.html')

@app.route("/step2")
def step2():
    return render_template('pneumoniaTestPage2.html')

@app.route("/step3", methods=["GET", "POST"])
def step3():

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            files = glob.glob('Testing_Image_Folder/Test/*')
            for f in files:
                os.remove(f)

            files = glob.glob('static/Testing_Image_Folder/Test/*')
            for f in files:
                os.remove(f)

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            print("saved")

            original = r'Testing_Image_Folder/Test/'+image.filename
            target = r'static/Testing_Image_Folder/Test/test.jpeg'

            shutil.copyfile(original, target)

    x= predictXray()
    print(x)
    return render_template('pneumoniaTestPage3.html', test_result = x, imgName=image.filename  )

def predictXray():

    # Loading created model
    model = load_model(MODEL_PATH)
    img = image.load_img('static/Testing_Image_Folder/Test/test.jpeg',
                         target_size=(224, 224))

    # Converting the X-Ray into numpy array. returns a A 3D Numpy array.
    imagee = image.img_to_array(img)

    # expands the array by inserting a new axis at the specified position
    imagee = np.expand_dims(imagee, axis=0)

    #making the image into required range
    # img_data = preprocess_input(imagee)
    prediction = model.predict(imagee)


    if prediction[0][0] < prediction[0][1]:  # Printing the prediction of model.
        print(f'Person is affected with Pneumonia. {prediction}')
        return 1
    else:
        print(f'Person is safe.{prediction}')
        return 0

    # print(f'Predictions: {prediction}')

    return none



if __name__ == "__main__":
    app.run(debug=True)