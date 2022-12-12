import numpy as np
from random import *

def lecture_donnees(nom_fichier, delimiteur=',',prop_test=0.4):
    """ Lit le fichier contenant les données et renvoiee les matrices correspondant

    Parametres
    ----------
    nom_fichier : nom du fichier contenant les données
    delimiteur : caratère délimitant les colonne dans le fichier ("," par défaut)

    Retour
    -------
    X : matrice des données de dimension [N, nb_var]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    """

    # Lecture du fichier
    data = np.loadtxt(nom_fichier, delimiter=delimiteur)
    #shuffle permet de mélanger les données
    shuffle(data)
    [nb_ligne, nb_colonne] = data.shape
    N=nb_ligne
    nb_var=nb_colonne-1
    Y=np.zeros((N,1))
    # Extraction des données
    
    X = data[:,:-1]
    Y = data[:,-1].reshape(N,1)


    return X, Y, N, nb_var