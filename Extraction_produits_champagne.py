################################################################################################################
# Projet 6 - Extraction des produits à base de champagne.
################################################################################################################
"""
Ce fichier extrait depuis l'API Edamam Food and Grocery Database les 10 premiers produits à base de champagne. 
Récupération pour chaque produit de : 
    - foodId
    - label
    - category
    - foodContentsLabel
    - image

Enregistrement du tableau dans un fichier csv.

"""
################################################################################################################
# Importation des libraires : 
import pandas as pd 
import requests
import os
from io import BytesIO
from PIL import Image

################################################################################################################
# Etape 1 : Test de connexion

# 1. Paramètres de connexion : 
url = "https://edamam-food-and-grocery-database.p.rapidapi.com/api/food-database/v2/parser"

headers = {
    "X-RapidAPI-Key" : 'c6e4082ebbmsh75b8ea62d440ce4p138b2cjsn6ca375e42487',
    "X-RapidAPI-Host" : 'edamam-food-and-grocery-database.p.rapidapi.com',
    } 
# 2. Test : 
response = requests.get(url, headers=headers)

if response.status_code == 200: 
    print("Connexion à  l'API réussie.")
else: 
    print(f"Problème de connexion erreur {response.status_code}")

################################################################################################################
# Etape 2 :Recherche des produtis à base de champagne

# 1. Paramètres de requêtes  
querystring = {"ingr": "champagne"}

# 2. Création des données : 
all_results = []  
page_count = 0

# 3. Récupération des résulats : 

while True:
    response = requests.get(url, headers=headers, params=querystring).json()
    
    # 3.1 Ajout des résultats à la liste s'ils existent : 
    all_results.extend(response.get('hints', []))
    
    # 3.2 Vérification de la présence d'un lien vers un autre page :
    next_link = response.get('_links', {}).get('next', None)
    
    # 3.3 Mise à jour des paramètres vers le lien suivant s'il existe :
    if next_link:
        url = next_link['href']
        page_count += 1 
    else:
        print(f"Collecte de données terminée.\n\tNombre de pages parcourues : {page_count}\n\tNombre total d'éléments récoltés : {len(all_results)}")
        break 
    
################################################################################################################
# Etape 3 : Récupération des éléments nécéssaires

# 1. Création d'un dictionnaire de stockage de données : 
dico = {}

# 2. Récupération des données : 

for i in range(len(all_results)):
    # 2.1 Récupération du dicctionnaire 'food' : 
    food_info = all_results[i].get('food', {})
    
    # 2.2 Récupération des éléments : 
    dico[i] = {
        'foodid': food_info.get('foodId', None),  
        'label': food_info.get('label', None),   
        'category': food_info.get('category', None),
        'foodContentsLabel' : food_info.get('foodContentsLabel', None),
        'image': food_info.get('image', None),      
    }

# 3. Création d'un Dataframe : 
df = pd.DataFrame(dico).T

################################################################################################################
# Etape 4 : Suppression des produits en doublon

df = df.drop_duplicates(subset='foodid')

################################################################################################################
# Etape 5 ; Téléchargement des images à l'aide des url associées et récupération des noms de fichiers : 

# 1. Création d'un dossier de sauvegarde des images :
dossier_enregistrement = '2. Sauvegardes/4. API'
os.makedirs(dossier_enregistrement, exist_ok=True)

# 2. Création d'une fonction pour télécharger et sauvegarder les images à partir des URL : 
def telechargement_et_enregistrement_image(url, dossier):
    
    # Cas particulier des URL None : 
    if url is None:
        return None

    # Téléchargement du fichier : 
    response = requests.get(url)
    
    # Récupération uniquement des fichiers images : 
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image_name = url.split('/')[-1]
        
        if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.png'):
            image_path = os.path.join(dossier, image_name)
            image.save(image_path)
            
            return image_name
        
        else:
            return None
        
    else:
        return None

# 3. Application de la fonction à la colonne 'image' et remplzcement de celle-ci par uniquement les noms de fichier :
df['image'] = df['image'].apply(
    telechargement_et_enregistrement_image,
    dossier=dossier_enregistrement
)

print ("Téléchargement des photos de produits dans le dossier '2. Sauvegardes/4. API' terminé")

################################################################################################################
# Etape 6 : Enregistrement des 10 premiers produits au format csv

df = df.head(10)
df.to_csv('2. Sauvegardes/4. API/Extraction_10_prduits_champagne.csv')    

print ("Création du fichier csv contenant les 10 premiers produits dans le dossier '2. Sauvegardes/4. API' terminé")

