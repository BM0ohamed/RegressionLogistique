import matplotlib.pyplot as plt
import numpy as np


def affichage(X, Y):
    """ Affichage en 2 dimensions des données (2 dimensions de X) et représentation de la 
        classe (indiquée par Y) par une couleur
    

    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    None

    """
    # Extraction des données
    X1 = X[:, 1]
    X2 = X[:, 2]
    Y = Y.reshape(X.shape[0], 1)
    # Affichage des données
    plt.scatter(X1, X2, c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.show()
    
    return None