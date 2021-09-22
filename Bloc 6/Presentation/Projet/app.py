import os
from flask import Flask 
from flask import request
from flask import render_template
import requests

import tensorflow as tf
import pandas as pd



# render_template est utilisé pour générer la sortie à partir d'un fichier html qui se trouve dans le dossier templates de l'application

app = Flask(__name__)


UPLOAD_FOLDER = "C:/Users/marti/Projet_LDS/Projet/static" #l'endroit où seront enregistré les images


model = tf.keras.models.load_model("InceptionV3_25classes.h5")


def resize(image_path):
    """
    IN : localisation de l'image
    OUT: l'image redimensionné
    obj: redimensionner l'igame avant prediction
    """
    image_resize = tf.io.read_file(image_path) # read the file as byte type
    image_resize = tf.image.decode_image(image_resize, channels=3) # convert bytes to a tensor
    image_resize = tf.image.convert_image_dtype(image_resize, tf.float64) # convert the image tensor from int 0-255 to float 0-1
    image_resize = tf.image.resize(image_resize, [150, 150]) # resize the image so the model can run inference
    image_resize = tf.expand_dims(image_resize, axis=0) # let's expand the dimension so the input data has shape (1,299,299,3)
    return image_resize

def prediction(model, image):
    """
    IN: model est le model chargé avec les poids; image est l'image redimensionné par def resize
    OUT: la valeur de la prediction, c'est un integer
    obj: appliquer le model à une image pour sortir une prediction
    """
    pred = model.predict(image)
    pred_label = tf.argmax(pred, axis=-1).numpy()
    return pred_label[0] # On obtien un array avec la prediction, je viens chercher la valeur de cette prediction

def categorie(pred):
    """
    IN: integer de def prediction()
    OUT: Le nom de la catégorie de la prediction
    obj: Rechercher dans le  fichier csv qui contient les labels et les catégories. Sortir la catégorie à laquelle l'image appartient
    """
    labels_df = pd.read_csv("labels.csv", sep=";")
    label = labels_df.iloc[pred,:]['categorie']
    return label

def dictionnaire():
    """
    Cette définition retourne un dictionnaire contenant 12 clefs qui représente 12 lettres. Elles ne contiennent pas de valeurs car elle seront remplacés par les lettres de la prédiction.
    Cette def permet de remettre à 0 le dictionnaire à chaque nouvelle prédiction.
    """
    return {"l0" : "","l1" : "","l2" : "","l3" : "","l4" : "","l5" : "","l6" : "","l7" : "","l8" : "","l9" : "","l10" : "","l11" : ""}



@app.route('/')# methods =['GET', 'POST']
def man():
    return render_template('upload.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/projet')
def projet():
    return render_template('projet.html')

@app.route('/predict', methods =['GET', 'POST'])
def upload_predict():
    if request.method == 'POST': # Cela existera que lorsque que la methode de demande est POST
        image_file = request.files['image'] #'image' c'est le nom mis dans 'name' dans le fichier upload.html
        if image_file:
            
            image_location = os.path.join(
            UPLOAD_FOLDER,
            image_file.filename)
            image_file.save(image_location) # Dans cette partie on sauvegarde l'image téléchargée
            
            
            image = resize(image_location) # On redimensionne l'image avec la def resize
            pred = prediction(model, image) # On fait une prédiction gr^ce à notre model pré-entraîné, il donne un chiffre corespond à une catégorie
            cat = categorie(pred) # On va chercher la catégorie de la prediction dans dans fichier csv
            
            dic = dictionnaire()
            for i in range(0,len(cat)):
                cat = cat.lower()
                dic["l{}".format(i)] = cat[i]
            
            return render_template("prediction.html",
                                   prediction = cat,
                                   href = 'static/{}'.format(image_file.filename),
                                   href_video = "static/videoLDS/{}.mp4".format(cat),
                                   lettre_0 = dic["l0"],
                                   lettre_1 = dic["l1"],
                                   lettre_2 = dic["l2"],
                                   lettre_3 = dic["l3"],
                                   lettre_4 = dic["l4"],
                                   lettre_5 = dic["l5"],
                                   lettre_6 = dic["l6"],
                                   lettre_7 = dic["l7"],
                                   lettre_8 = dic["l8"],
                                   lettre_9 = dic["l9"],
                                   lettre_10 = dic["l10"],
                                   lettre_11 = dic["l11"]
                                   ) 
        
    return render_template("projet.html", erreur = "Veuillez choisir une image ")

if __name__ == '__main__':
    app.run(debug = True)
    
    