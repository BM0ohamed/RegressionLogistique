import matplotlib.pyplot as plt
import numpy as np
from lecture_donnees import *
from normalisation import *
from descente_gradient import *
from affichage import *
from prediction import *
from taux_classification import *
from decoupage import *

# ===================== Partie 1: Lecture et normalisation des données=====================
print("Lecture des données ...")

X, Y, N, nb_var = lecture_donnees("TD5-donnees/notes.txt")

# Affichage des 10 premiers exemples du dataset
print("Affichage des 10 premiers exemples du dataset : ")
for i in range(0, 10):
    print(f"x = {X[i,:]}, y = {Y[i]}")
    
# Normalisation des variables (centrage-réduction)
print("Normalisation des variables ...")

X, mu, sigma = normalisation(X)
#Ajout d'une colonne de 1 pour theta0

X=np.hstack((np.ones((N,1)), X))
#On découpe les données en données d'apprentissage et donnéees de test
x_app, y_app, x_test, y_test,N_app=decoupage_donnees(X,Y,0.4)

# Affichage des points en 2D et représentation de leur classe réelle par une couleur
if nb_var == 2 :
    plt.figure(0)
    plt.title("Disposition des points en 2D")
    affichage(x_app,y_app)

# ===================== Partie 2: Descente du gradient =====================
print("")
print("=====================================================")
print("Apprentissage par descente du gradient ...")

# Choix du taux d'apprentissage et du nombre d'itérations
alpha = 0.01
nb_iters = 10000

# Initialisation de theta et réalisation de la descente du gradient
theta = np.zeros((1,nb_var+1))
theta, J_history = descente_gradient(x_app, y_app, theta, alpha, nb_iters)

# Affichage de l'évolution de la fonction de cout lors de la descente du gradient
plt.figure(1)
plt.title("Evolution de le fonction de cout lors de la descente du gradient")
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel("Nombre d'iterations")
plt.ylabel("Cout J")

# Affichage de la valeur de theta
print(f"Theta calculé par la descente du gradient : {theta}")

# Evaluation du modèle avec les données d'apprentissage
print("Affichage des points en 2D et représentation de leur classe prédite par une couleur")
Ypred = prediction(x_app,theta)

print("Taux de classification : ", taux_classification(Ypred,y_app))

# Affichage des points en 2D et représentation de leur classe prédite par une couleur
if nb_var == 2 :
    plt.figure(2)
    plt.title("Disposition des points en 2D données d'apprentissage")
    affichage(x_app,Ypred)
    
plt.show()

print("Shape theta : ", theta.shape)
print("Shape x_app : ", x_app.shape)
print("Shape y_app : ", y_app.shape)
print("Shape x_test : ", x_test.shape)
print("Shape y_test : ", y_test.shape)



# ===================== Partie 3: Evaluation du modèle =====================
print("")
print("=====================================================")
print("Evaluation du modèle avec les donnéees de test")

# Evaluation du modèle avec les données de test
print("Affichage des points en 2D et représentation de leur classe prédite par une couleur")
Ypred_test = prediction(x_test,theta)

print("Taux de classification : ", taux_classification(Ypred_test,y_test))

# Affichage des points en 2D et représentation de leur classe prédite par une couleur
if nb_var == 2 :
    plt.figure(3)
    plt.title("Disposition des points en 2D données de test")
    affichage(x_test,Ypred_test)

plt.show()


print("Regression logistique Terminée.")


# ===================== Partie 4: Classification à plusieurs classes =====================
print("")
print("=====================================================")
print("Classification à plusieurs classes ...")


