import numpy as np
from sigmoide import *


def prediction(X,theta):
    """ Predit la classe de chaque élement de X
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives


    Retour
    -------
    p : matrices de dimension [N,1] indiquant la classe de chaque élement de X (soit 0, soit 1)

    """
    p = np.zeros((X.shape[0],1))
    p = sigmoide(np.dot(X, theta.T)) >= 0.5
    

    return p


