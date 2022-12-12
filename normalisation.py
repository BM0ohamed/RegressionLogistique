import numpy as np


def normalisation(X):
    """
    

    Parametres
    ----------
    X : matrice des données de dimension [N, nb_var]
    
    avec N : nombre d'éléments et nb_var : nombre de variables prédictives

    Retour
    -------
    X_norm : matrice des données centrées-réduites de dimension [N, nb_var]
    mu : moyenne des variables de dimension [1,nb_var]
    sigma : écart-type des variables de dimension [1,nb_var]

    """
    # Initialisation des variables de sortie
    X_norm = np.zeros(X.shape)
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))

    # Calcul de la moyenne et de l'écart-type de chaque variable
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # moy=mat.mean(0) #on obtient la moyenne par colonne
    # ectype = mat.std(0) #on obtient l'écart type par colonne
    # mat_norm = (mat - moy) / ectype

    X_norm = (X-mu)/sigma 

    return X_norm, mu, sigma
