import numpy as np

def decoupage_donnees(x,d,prop_test=0.4):
    """ Découpe les données initiales en deux sous-ensembles distincts d'apprentissage, et de test
    
    Parametres
    ----------
    x : matrice des données de dimension [N, nb_var]
    y : matrice des valeurs cibles [N, nb_cible]
    prop_test : proportion des données de test sur l'ensemble des données (entre 0 et 1)
    
    avec N : nombre d'éléments, nb_var : nombre de variables prédictives, nb_cible : nombre de variables cibles

    Retour
    -------
    x_app : matrice des données d'apprentissage
    d_app : matrice des valeurs cibles d'apprentissage
    x_test : matrice des données de test
    d_test : matrice des valeurs cibles de test

    """

    N,nb_var = np.shape(x)
    N_app = int(N*(1-prop_test))

    x_app = x[:N_app]
    y_app = d[:N_app]
    x_test = x[N_app:]
    y_test = d[N_app:]

    return x_app, y_app, x_test, y_test,N_app
    

