###########################################################################################################################
# Fichier de fonctions du Projet 6 - fct_data
###########################################################################################################################

###########################################################################################################################
## 1. Importation des libraires :
 
from sys import displayhook
import pandas as pd 
import pickle

###########################################################################################################################


def dataframe_info(df):
    """
    Donne une descrition complète de des informations du DataFrame. 

    Args:
        df (pd.DataFrame): DataFrame pour à décrire
    """
    

    # 1. Création d'une liste de dictionnaires pour stocker les informations : 
    data_info = []
    
    # 2. Récupération des éléments pour chaque colonnes du DataFrame :
    for col in df.columns:
        col_name = col
        col_type = str(df[col].dtype)
        col_missing_percentage = (df[col].isna().mean()) * 100
        
        # Ajout des éléments à la liste : 
        data_info.append({'Nom_de_colonne': col_name,
                          'Data_type': col_type,
                          'Valeurs_manquantes_pourcentage': col_missing_percentage})
    
    # 3. Création du DataFrame à partir de la liste de dictionnaires : 
    column_info = pd.DataFrame(data_info)
    
    # 3. Définition de l'index de column_info :
    column_info = column_info.set_index('Nom_de_colonne')
    
    # 4. Affichage des résultats :
    ########################################################
   
    # 4.1 Dimmension :
    print(f"Dimmension : {df.shape}\n")
    print('--' * 50)
    
    # 4.2 Infos colonnes :
    print(f"Info sur les colonnes :\n")
    displayhook(column_info)
    print('--' * 50)
    
    # 4.3 Vérification si des colonnes numériques existent avant d'appeler describe : 
    numeric_columns = df.select_dtypes(include='number')
    if not numeric_columns.empty:
        # Info sur la répartition des variables numériques :
        print(f"Statistiques descriptives des colonnes numériques : \n")
        displayhook(numeric_columns.describe())
    
    # 4.4 Vérification si des colonnes catégorielles existent avant d'appeler describe : 
    categorical_columns = df.select_dtypes(include='object')
    if not categorical_columns.empty:
        # Info sur la répartition des variables catégorielles :
        print(f"Statistiques descriptives des colonnes catégorielles : \n")
        displayhook(categorical_columns.describe())
 
 ########################################################


def enregistrement_pickle(name, chemin, fichier):
    
    path = chemin + '/' + name + '.pickle'

    with open(path, 'wb') as f:
        pickle.dump(fichier, f)


def chargement_pickle(name, chemin): 
    
    path = chemin + '/' + name + '.pickle'
    
    with open(path, 'rb') as f:
        fichier = pickle.load(f)

    return fichier



