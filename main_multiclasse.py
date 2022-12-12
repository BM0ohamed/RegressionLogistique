import matplotlib.pyplot as plt
import numpy as np
from lecture_donnees import *
from normalisation import *
from descente_gradient import *
from affichage import *
from prediction import *
from taux_classification import *
from decoupage import *
from predMultiClasse import *
from random import *
#Ce code correspond à la regression linéaire multi classe en utilisant le principe de 1 contre tous
# ===================== Partie 1: Lecture et normalisation des données=====================
print("Lecture des données ...")

X, Y, N, nb_var = lecture_donnees("TD5-donnees/iris.txt")

# Affichage des 10 premiers exemples du dataset
print("Affichage des 10 premiers exemples du dataset : ")
for i in range(0, 10):
    print(f"x = {X[i,:]}, y = {Y[i]}")
    
# Normalisation des variables (centrage-réduction)
print("Normalisation des variables ...")

X, mu, sigma = normalisation(X)
X=np.hstack((np.ones((N,1)), X))

#On découpe les données en données d'apprentissage et donnéees de test
prop_test=0.2
x_app, y_app, x_test, y_test,N_app=decoupage_donnees(X,Y,prop_test)

# Affichage des points en 2D et représentation de leur classe réelle par une couleur



# ===================== Partie 2: Descente du gradient =====================
print("")
print("=====================================================")
print("Apprentissage par descente du gradient ...")

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.01
nb_iters = 10000

#On détermine le nombre de classes :
nb_classes=int(np.max(y_app)+1)
print("il y'a {} classes dans ce jeu de données.".format(nb_classes))
print("il y'a {} variables dans ce jeu de données.".format(nb_var))
# Initialisation de theta et réalisation de la descente du gradient
theta_multi = np.zeros((nb_classes,nb_var+1))


for i in range(nb_classes):
    y_app_i=np.zeros(N_app)
    for j in range(N_app):
        if y_app[j]==i:
            y_app_i[j]=1
        else:
            y_app_i[j]=0
    y_app_i=y_app_i.reshape(N_app,1)
    theta_multi[i,:] = theta_multi[i,:].reshape(1,nb_var+1)
    # print("shape x_app : ",x_app.shape)
    # print("shape y_app_i : ",y_app_i.shape)
    # print("shape theta_multi : ",theta_multi.shape)
    # print("shape theta_multi[i,:] : ",theta_multi[i,:].shape)
    theta_multi[i,:], J_history = descente_gradient(x_app, y_app_i, theta_multi[i,:].reshape(nb_var+1,1).T, alpha, nb_iters)
    print("theta_multi[{}] : {}".format(i,theta_multi[i,:]))
    # Affichage de la courbe de convergence
    plt.figure(1)
    plt.title("Evolution de le fonction de cout lors de la descente du gradient")
    plt.plot(np.arange(J_history.size), J_history)
    plt.xlabel("Nombre d'iterations")
    plt.ylabel("Cout J")

# ===================== Partie 3: Prédiction et taux de classification =====================
print("")
print("=====================================================")
print("Prédiction et taux de classification ...")

# Prédiction sur les données d'apprentissage
p_app = predMultiC(x_app,theta_multi)
print("Taux de classification sur les données d'apprentissage : {}%".format(taux_classification(p_app, y_app)*100))

# Prédiction sur les données de test
p_test = predMultiC(x_test,theta_multi)
print("Taux de classification sur les données de test : {}%".format(taux_classification(p_test, y_test)*100))






