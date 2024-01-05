###########################################################################################################################
# Fichier de fonctions du Projet 6 - fct_image
###########################################################################################################################

###########################################################################################################################
## 1. Importation des libraires :
 
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm

## PILLOW 
from PIL import Image, ImageOps, ImageFilter

## OpenCV
import cv2

## Tensoflow 
import tensorflow as tf

## Scitkit-learn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

## Keras 
from keras.applications import vgg16, VGG16, resnet, ResNet152, ResNet152V2
from keras.applications import inception_v3, InceptionV3, inception_resnet_v2, InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

###########################################################################################################################   
## PRE-TRAITEMENT DES IMAGES


def dimensions_image(serie):
    """
    Calcul et renvoie la taille, la largeur et la hauteur en pixels de chaque image 

    Args:
        serie (pd.Series): Série pandas contenant les chemin d'accès aux images

    Returns:
        pd.Series: Retroune un tupple de 3 Séries pandas contenant la taille, la largeur et la hauteur des images.
    """
    # 1. Création des listes de stockage :
    
    tailles = []
    largeurs = []
    hauteurs = []
    
    # 2. Boucle sur chaque chemin : 
    for chemin in serie:
        image = Image.open(chemin)
        
        largeur, hauteur = image.size
        taille = largeur * hauteur
        
        tailles.append(taille)
        largeurs.append(largeur)
        hauteurs.append(hauteur)
        
    return pd.Series(tailles), pd.Series(largeurs), pd.Series(hauteurs)


def histogram_pixels(img, titre):
    """
    Tace 3 graphiques : grpahique de l'image, histogramme de répartition des pixels et histogramme cumulés des pixels 

    Args:
        img (PIL.Image): Image à traiter
        titre (str): Titre pour le premier graphique
    """
    
    # 1. Transformation de l'image en tableau numpy : 
    img = np.array(img)
    
    # 2. Création des graphique : 
    plt.figure(figsize=(40,15))

    # 2.1 Affichage de l'image originale 
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(titre, fontsize=40)

    # 2.2 Affichage de la répartition des pixels : 
    plt.subplot(1, 3, 2)
    plt.hist(img.flatten(), bins=range(256))
    plt.xlim([0,255])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('Nombre de pixels', fontsize=30)
    plt.xlabel('Niveau de gris', fontsize=30)
    plt.title('Histogramme de répartition des pixels', fontsize=40)

    # 2.3 Affichage de l'histogramme cumulés des pixels : 
    plt.subplot(1, 3, 3)
    plt.hist(img.flatten(), bins=range(256), cumulative=True)
    plt.xlim([0,255])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('Nombre de pixels', fontsize=30)
    plt.xlabel('Niveau de gris', fontsize=30)
    plt.title('Histogramme cumulé des pixels', fontsize=40)

    plt.tight_layout()
    plt.show()
    
    
def process_image(chemin_image):
    """
    Récupère l'image d'après le chemin donné. 
    
    Traite l'image avec PIL comme suit : 
    - Réduction des taille (224x224) --> taille nécessaire pour l'utilisation des réseaux de neuronnes
    - Conversion de l'image en niveau de gris --> nécessaire pour l'utilisation de SIFT 
    - Suppression du bruit 
    - Egalisation de l'histogramme

    Enregistre la nouvelle image dans un fichier de sauvegarde.

    Args:
    chemin_image (str): chemin de l'image à traiter.
    
    Returns:
        str: Chemin de l'image traitée
     """
    
    # 1. Chargement de l'image :
    image = Image.open(chemin_image)
        
    # 2. Réduction de la taille à 224x224 :
    image = image.resize((224, 224), Image.LANCZOS)
        
    # 3. Conversion en niveaux de gris : 
    image = image.convert('L')
        
    # 4. Suppression du bruit (filtre median) : 
    image = image.filter(ImageFilter.MedianFilter(size=3))
        
    # 5. Égalisation de l'histogramme : 
    image = ImageOps.equalize(image)
        
    # 6. Sauvegarde des images traitées :
    path  = '2. Sauvegardes/2. IMAGE/Images traitées'
    
    # 6.1 Vérification de l'existance du dossier : 
    if not os.path.exists(path):
            os.makedirs(path)
        
    # 6.2 Générer le chemin de sauvegarde pour l'image traitée : 
    nom_image_traitée = os.path.basename(chemin_image)
    chemin_sauvegarde = os.path.join(path, nom_image_traitée)
        
    # 6.3 Enregistrement de l'image traitée dans le dossier de sauvegarde : 
    image.save(chemin_sauvegarde)
        
    # 7. Renvoyer le chemin de l'image traitée : 
    return chemin_sauvegarde
    
    
###########################################################################################################################
## EXTRACTION FEATURES IMAGE 


def extraction_features_sift(dico):
    """
    Création d'une liste des features présents dans l'ensemble des images. 
    Création d'un dictionnaire (clé: nom image, valeur: tableau numpy des features de l'image) 

    Args:
        dico (dict): dictionnaire des talbeaux numpy des pixels de chaque image 

    Returns:
        list, dict: liste de l'ensemble des features, dictionnaire des bags of visual words par image
    """
    
    # 1. Création des données de stockage : 
    bags_of_visual_words = {}
    all_features = []
    
    # 2. Insrtanciation de SIFT : 
    sift = cv2.SIFT_create() # type: ignore
    
    # 3. Parcours du dictionnaire d'image : 
    for name, value in dico.items():
        
        # 3.1 Liste des features : 
        #image_features = []
       
       # 3.2 Récupération des descirpteurs de l'image : 
        _, descripteurs = sift.detectAndCompute(value, None)
    
        # 3.3 Extension de la listes des features : 
        all_features.extend(descripteurs)
    
        # 3.4 Gestion des descripteurs nuls : 
        if descripteurs is None: 
            descripteurs = np.zeros((1,128))
        
        # 3.5 Création des features finales de l'image : 
        #image_features.append(descripteurs)
        
        # 3.6 Ajout des features de l'image aux dictionnaire bags of visual words :
        bags_of_visual_words[name] = descripteurs
    
    # 4. Transformation de all_features en tableau numpy 
    all_features = np.array(all_features)
        
    return all_features, bags_of_visual_words


def creation_vecteur_img_SIFT(descriptors_dict, model_kmeans):
    """
    Calcul des vecteurs Bag of Words (BoW) pour chaque image à partir de descripteurs SIFT.

    Args:
        descriptors_dict (dict): Un dictionnaire où chaque clé est l'identifiant de l'image
                                 et la valeur est une liste de descripteurs SIFT.
        model_kmeans (model): Modèle KMeans entrainé sur l'ensemble des features. 

    Returns:
        img_vectors (dict): Un dictionnaire où chaque clé est l'identifiant de l'image
              et la valeur est le vecteur BoW correspondant.
    """
    img_vectors = {}

    for image_id, descriptors in descriptors_dict.items():
        # 1. Prédiction du cluster pour chaque descripteurs : 
        prediction = model_kmeans.predict(descriptors)
        
        # 2. Comptage des occurrences de chaque cluster : 
        bow_vector, _ = np.histogram(prediction, bins=np.arange(len(model_kmeans.cluster_centers_) + 1))
        
        # 3. Ajout du vecteur BoW au dictionnaire : 
        img_vectors [image_id] = bow_vector
        
    return img_vectors


def extraction_features_vgg16(chemin_image):
    """
    Extraction des features pour chaque image en entrée à l'aide de VGG16 

    Args:
        chemin_image (pd.Series): Serie pandas contenant les chemin d'accès à chaque image

    Returns:
        np.array : tableau numpy contenant les features extraites de chaque image.
    """
    # 1. Instanciation du modèle : 
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 2. Pre-processing des images : 
    images = []

    for path in tqdm(chemin_image, desc="Exctration features"):
    
        # 2.1 Lecture et redimensionnement des images : 
        img = tf.keras.preprocessing.image.load_img(path=path,
                                                    target_size=(224, 224),
                                                    )
        
        # 2.2 Convertion des images en tableau numpy : 
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # 2.3 Normalisation spécifique pour VGG16 : 
        img = vgg16.preprocess_input(img)
        
        # 2.4 Stockage : 
        images.append(img)
        
    # 2.5 Convertion de la liste en tableau numpy : 
    images = np.array(images)

    # 3. Extraction des features : 
    features = model.predict(images)
    
    # 4. Redimenssionnement en 2D de features : 
    features = features.reshape(features.shape[0], -1)
    
    return features


def extraction_features_resnet152(chemin_image, version='v1'):
    """
    Extraction des features pour chaque image en entrée à l'aide de ResNet-152.

    Args:
        chemin_image (pd.Series): Série pandas contenant les chemins d'accès à chaque image.
        version (str): Version de ResNet-152 à utiliser ('v1' ou 'v2').

    Returns:
        np.array : Tableau numpy contenant les caractéristiques extraites de chaque image.
    """
    
    # 1. Instanciation du modèle : 
    if version == 'v1':
        model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif version == 'v2':
        model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("La version doit être 'v1' ou 'v2'.")
    
    # 2. Pré-traitement des images :
    images = []

    for path in tqdm(chemin_image, desc="Extraction des features"):
    
        # 2.1 Lecture et redimensionnement des images :
        img = tf.keras.preprocessing.image.load_img(path=path,
                                                    target_size=(224, 224),
                                                    )
        
        # 2.2 Conversion des images en tableau numpy :
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # 2.3 Pré-traitement spécifique à ResNet-152 :
        img = resnet.preprocess_input(img)
        
        # 2.4 Stockage :
        images.append(img)
        
    # 2.5 Conversion de la liste en tableau numpy :
    images = np.array(images)

    # 3. Extraction des caractéristiques :
    features = model.predict(images)
    
    # 4. Rédimensionnement en 2D des caractéristiques :
    features = features.reshape(features.shape[0], -1)
    
    return features


def extraction_features_inceptionnet(chemin_image, version='inceptionv3'):
    """
    Extraction des features pour chaque image en entrée à l'aide de modèles InceptionV4 ou Inception-ResNet.

    Args:
        chemin_image (pd.Series): Série pandas contenant les chemins d'accès à chaque image.
        version (str): Nom du modèle à utiliser ('inceptionv4' ou 'inceptionresnet').

    Returns:
        np.array : Tableau numpy contenant les caractéristiques extraites de chaque image.
    """
        
    # 1. Instanciation du modèle : 
    if version == 'inceptionv3':
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    elif version == 'inceptionresnet':
        model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    else:
        raise ValueError("Le modèle doit être 'inceptionv3' ou 'inceptionresnet'.")
    
    # 2. Pré-traitement des images :
    images = []

    for path in tqdm(chemin_image, desc="Extraction des features"):
    
        # 2.1 Lecture et redimensionnement des images :
        img = tf.keras.preprocessing.image.load_img(path=path,
                                                    target_size=(299, 299),
                                                    )
        
        # 2.2 Conversion des images en tableau numpy :
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # 2.3 Pré-traitement spécifique au modèle :
        if version == 'inceptionv4':
            img = inception_v3.preprocess_input(img)
        elif version == 'inceptionresnet':
            img = inception_resnet_v2.preprocess_input(img)
                    
        # 2.4 Stockage :
        images.append(img)
        
    # 2.5 Conversion de la liste en tableau numpy :
    images = np.array(images)

    # 3. Extraction des caractéristiques :
    features = model.predict(images)
    
    # 4. Rédimensionnement en 2D des caractéristiques :
    features = features.reshape(features.shape[0], -1)
    
    return features


###########################################################################################################################
## CLASSIFICATION SUPERVISEE


def preparation_img_ResNet(chemin_image):
    """
    Prépare les images pour être utilisées avec le modèle ResNet-152_v1.
    
    Args:
        chemin_image (list): Une liste de chemins vers les fichiers d'images à préparer.

    Returns:
        np.array: Un tableau numpy contenant les images préparées prêtes à être utilisées avec ResNet-152_v1.
    """
    
    images = []

    for path in chemin_image:
        
        # 1 Lecture et redimensionnement des images :
        img = tf.keras.preprocessing.image.load_img(
            path=path,                                                       
            target_size=(224, 224),
            )
            
        # 2 Conversion des images en tableau numpy :
        img = tf.keras.preprocessing.image.img_to_array(img)
            
        # 3 Pré-traitement spécifique à ResNet-152 :
        img = resnet.preprocess_input(img)
            
        # 4 Stockage :
        images.append(img)
               
    # 5 Conversion de la liste en tableau numpy :
    images = np.array(images)
    
    return images


def creation_model_class_ResNet():
    """
    Crée un modèle de classification basé sur ResNet-152V1 pré-entraîné.

    Returns:
        keras.Model: Le modèle de classification complet prêt à être entraîné.
    """
    
    # 1. Chargement du modèle ResNet-152V1 pré-entraîné
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 2. Ajout des couches personnalisées au modèle : 
    x = base_model.output # Utilisation des premières couches pré-entrainées. 
    x = GlobalAveragePooling2D()(x) # Réduction de la sortie en un vecteur de dimension 1D
    x = Dense(256, activation='relu')(x) # Formatage de la sortie à des vecteurs de dimensions 256. 
    x = Dropout(0.5)(x) # Désactivation aléatoire de 50% des neuronnes de sorties à chaque apparentissage. 
    predictions = Dense(7, activation='softmax')(x)  # Sortie du modèle (classification en 7 catégories).

    # 3. Création d'un modèle complet : 
    model = Model(inputs=base_model.input, outputs=predictions)

    # 4. Désactivation de l'entrainement des couches de base du modèle : 
    for layer in base_model.layers:
        layer.trainable = False

    # 5. Compilation du modèle : 
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy"])

    return model

    
def train_random_forest(X_train, y_train, X_test, y_test, params):
    """
     Entraîne un classifieur RandomForestClassifier avec une grille de validation croisée.

    Args:
        X_train (array-like): Les caractéristiques d'entraînement.
        y_train (array-like): Les étiquettes d'entraînement.
        X_test (array-like): Les caractéristiques de test.
        y_test (array-like): Les étiquettes de test.
        params (dict): Dictionnaire contenant les hyperparamètres à rechercher.

    Returns:
        float: valeur accuracy train 
        float: valeur accuracy validation
        float: valeur accuracy test
        np.array: prédictions sur l'ensemble de test
        dict: meilleures hyperparamètres de RandomFOrestClassifier
    """
    
    # 1. Création du modèle : 
    rf = RandomForestClassifier(random_state=42)
    
    # 2. Création de grid : 
    grid = GridSearchCV(estimator=rf, param_grid=params, cv=5, scoring='accuracy', verbose=1)
    
    # 3. Entrainement par validation croisée : 
    grid.fit(X_train, y_train)
        
    # 4. Récupération des meilleures hyperparamètres :
    best_params = grid.best_params_
    
    # 5. Récupération du meilleur modèle : 
    best_model = grid.best_estimator_
    
    # 6. Calcul accuracy sur l'entrainement : 
    y_pred_train = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    # 7. Récupération de l'accuracy de validation : 
    accuracy_val = grid.best_score_
    
    # 8. Caclcul l'accuracy test :
    y_pred_test = best_model.predict(X_test) 
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    # 9. Transformation de y_pred_test : 
    #y_pred_test = np.argmax(y_pred_test, axis=1)
    
    # Affichage : 
    print('--'*50)
    print(f"Meilleure hyperparamètres pour RandomForestClassifier : {best_params}\n")
    print(f"Trainning Accuracy : {accuracy_train *100:.2f}%")
    print(f"Validation Accuracy : {accuracy_val *100:.2f}%")
    print(f"Test Accuracy : {accuracy_test *100:.2f}%")

    
    return accuracy_train, accuracy_val, accuracy_test, y_pred_test, best_params


def train_knn(X_train, y_train, X_test, y_test, params):
    """
    Entraîne un classifieur K-Nearest Neighbors (KNN) avec une grille de validation croisée.

    Args:
        X_train (array-like): Les caractéristiques d'entraînement.
        y_train (array-like): Les étiquettes d'entraînement.
        X_test (array-like): Les caractéristiques de test.
        y_test (array-like): Les étiquettes de test.
        params (dict): Dictionnaire contenant les hyperparamètres à rechercher.

    Returns:
        float: valeur accuracy train 
        float: valeur accuracy validation
        float: valeur accuracy test
        np.array: prédictions sur l'ensemble de test
        dict: meilleures hyperparamètres de KNNClassififer
    """
    
    # 1. Création du modèle : 
    knn = KNeighborsClassifier()
    
    # 2. Création de la grille de recherche : 
    grid = GridSearchCV(estimator=knn, param_grid=params, cv=5, scoring='accuracy', verbose=1)
    
    # 3. Entraînement par validation croisée : 
    grid.fit(X_train, y_train)
        
    # 4. Récupération des meilleures hyperparamètres :
    best_params = grid.best_params_
    
    # 5. Récupération du meilleur modèle : 
    best_model = grid.best_estimator_
    
    # 6. Calcul de l'accuracy sur l'entraînement : 
    y_pred_train = best_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    # 7. Récupération de l'accuracy de validation : 
    accuracy_val = grid.best_score_
    
    # 7. Calcul de l'accuracy sur le test : 
    y_pred_test = best_model.predict(X_test) 
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    # Affichage : 
        
    print('--'*50)
    print(f"Meilleure hyperparamètres pour KNNClassififer : {best_params}\n")
    print(f"Trainning Accuracy : {accuracy_train *100:.2f}%")
    print(f"Validation Accuracy : {accuracy_val *100:.2f}%")
    print(f"Test Accuracy : {accuracy_test *100:.2f}%")

    
    return accuracy_train, accuracy_val, accuracy_test, y_pred_test, best_params    
    
    
def create_data_generator(dataframe, subset): 
    """
    
    Crée et retourne un générateur de données d'images à partir d'un DataFrame.

    Args:
        dataframe (pd.DataFrame): Un DataFrame contenant les données de l'image et les étiquettes.
        subset (str): L'ensemble de données à générer. Les options valides sont 'training', 'validation' ou 'test'.

    Raises:
        ValueError: Si le paramètre 'subset' n'est pas l'une des options valides.

    Returns:
        generator: Un générateur de données d'images prêt à être utilisé pour l'entraînement, la validation ou les tests.
    """
    
    # 1. Défintion des data_generator : 
    
    datagen_train = ImageDataGenerator(
        rotation_range=10,                  # Rotation aléatoire jusqu'à 10 degrés
        width_shift_range=0.1,              # Déplacement horizontal aléatoire jusqu'à 20% de la largeur de l'image
        height_shift_range=0.1,             # Déplacement vertical aléatoire jusqu'à 20% de la hauteur de l'image
        zoom_range=0.2,                     # Zoom aléatoire jusqu'à 20%
        horizontal_flip=True,               # Retournement horizontal aléatoire
        validation_split=0.2,
        preprocessing_function=resnet.preprocess_input,
    )

    datagen_test = ImageDataGenerator(
        validation_split=0,
        preprocessing_function=resnet.preprocess_input,
        )
    
    if subset == 'training' or subset == 'validation': 
        
        generator = datagen_train.flow_from_dataframe(
            dataframe=dataframe,
            x_col='chemin_image',
            y_col='category_cleaned',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            seed=42,
            subset=subset,
        )
        
        return generator
    
    elif subset == 'test':
    
        generator = datagen_test.flow_from_dataframe(
            dataframe=dataframe,
            x_col='chemin_image',
            y_col='category_cleaned',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            seed=42,
        )
        
        return generator

    elif subset == 'train_Knn' or subset == 'train_RandomForest':
         
        generator = datagen_train.flow_from_dataframe(
            dataframe=dataframe,
            x_col='chemin_image',
            y_col='category_cleaned',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            seed=42,
            subset=None,
        )
        
        return generator
    
    else: 
        raise ValueError("Le paramètre 'subset' doit être 'training', 'validation', 'test', 'train_Knn' 'train_RandomForest'.")


def df_result_class_supervisee(dico):
    
    # 1. Listes de stockage : 
    index = []
    accuracy_train = []
    accuracy_validation = []
    accuracy_test = []
    
    # 2. Récupération des données : 
    for key in dico.keys():
        
        index.append(key)
        accuracy_train.append(round(dico[key]['Accuracy_train']*100, 2))
        accuracy_validation.append(round(dico[key]['Accuracy_val']*100, 2))
        accuracy_test.append(round(dico[key]['Accuracy_test']*100, 2))
    
    # 3. Création d'un DataFrame : 
    
    df = {
        'Index': index,
        'Accuracy_train': accuracy_train,
        'Accuracy_validation': accuracy_validation, 
        'Accuracy_test': accuracy_test,
    }
 
    df = pd.DataFrame(df)
    df = df.set_index('Index')
    df = df.sort_values('Accuracy_test', ascending=False)
    
    return df


###########################################################################################################################
## Fonctions en attentes : 

""
def plot_training_history(history):
    """
    Visualisation de l'historique d'entraînement d'un modèle Keras.

    Args:
        history (keras.callbacks.History): L'objet d'historique retourné par model.fit.
    """
    # Visualisation de la perte
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Perte d'entraînement")
    plt.plot(history.history['val_loss'], label="Perte de validation")
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    # Visualisation de la précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Précision d'entraînement")
    plt.plot(history.history['val_accuracy'], label="Précision de validation")
    plt.xlabel('Epochs')
    plt.ylabel('Précision')
    plt.legend()

    plt.tight_layout()
    plt.show()  
""

""
def create_data_generators_keras(data_frame, validation_split, data_augmentation=False,):
    """
    Crée des générateurs de données d'entraînement et de validation à partir d'un DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame contenant les chemins d'accès aux images et les catégories.
        validation_split (float): Fraction des données à utiliser pour la validation.
        data_augmentation (bool): Indique si une augmentation de données doit être appliquée.

    Returns:
        tuple: Un tuple contenant les générateurs de données d'entraînement et de validation.
    """
    
    # 1. Création d'un générateur de données pour le prétraitement des images : 
    if data_augmentation:
        
        datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,                # Mise à l'échelle des valeurs de pixel entre 0 et 1
        rotation_range=10,                  # Rotation aléatoire jusqu'à 40 degrés
        width_shift_range=0.1,              # Déplacement horizontal aléatoire jusqu'à 20% de la largeur de l'image
        height_shift_range=0.1,             # Déplacement vertical aléatoire jusqu'à 20% de la hauteur de l'image
        shear_range=0.1,                    # Déformation en cisaillement aléatoire
        zoom_range=0.2,                     # Zoom aléatoire jusqu'à 20%
        horizontal_flip=True,               # Retournement horizontal aléatoire
        fill_mode='nearest',                # Mode de remplissage des pixels lorsque des transformations sont appliquées
    )
        
    else:
        datagen = ImageDataGenerator(rescale=1./255)    # Mise à l'échelle des valeurs de pixel entre 0 et 1


    # 2. Création des générateurs de données : 

    if validation_split !=0:
        
        data_train, data_validation, _, _ = train_test_split(
            data_frame,
            data_frame['category_cleaned'],
            test_size=validation_split,
            random_state=42,
            stratify=data_frame['category_cleaned'],
            )
    
        # Création du générateur de données de train : 
        train_generator = datagen.flow_from_dataframe(
            dataframe=data_train,
            x_col='chemin_image',
            y_col='category_cleaned',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            preprocessing_function=resnet.preprocess_input, 
        )

        # Création du générateur de données de validation
        validation_generator = datagen.flow_from_dataframe(
            dataframe=data_validation,
            x_col='chemin_image',
            y_col='category_cleaned',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            preprocessing_function=resnet.preprocess_input, 
        )

        return train_generator, validation_generator
    
    else: 
        
        generator = datagen.flow_from_dataframe(
            dataframe=data_frame,
            x_col='chemin_image',
            y_col='category_cleaned',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            preprocessing_function=resnet.preprocess_input, 
        )
        
        return generator
""