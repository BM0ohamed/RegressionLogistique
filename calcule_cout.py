import numpy as np
from math import *
from sigmoide import *


def calcule_cout(X, Y, theta):
    """ Calcule la valeur de la fonction cout (moyenne des différences au carré)
    
    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var+1]
    Y : matrice contenant les valeurs de la variable cible de dimension [N, 1]
    theta : matrices contenant les paramètres theta du modèle linéaire de dimension [1, nb_var+1]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Return
    -------
    cout : nombre correspondant à la valeur de la fonction cout (moyenne des différences au carré)

    """
    cout = - np.sum( Y*(np.log(sigmoide(np.dot(X, theta.T)))) + (1-Y)*(np.log(1-sigmoide(np.dot(X, theta.T)))))

    return cout


# if __name__ == '__main__':
#     # Test de la fonction
#     X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     Y = np.array([[1], [0], [1]])
#     theta = np.array([[1, 1, 1]])
#     print(calcule_cout(X, Y, theta))