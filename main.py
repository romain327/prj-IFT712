from Window import Window
from Controller import Controller

##
# @file main.py
# @brief Point d'entrée principal de l'application de classification.
# @details Ce script orchestre le lancement de l'application en suivant le pattern MVC :
#          1. Instanciation de la vue (Window).
#          2. Instanciation du contrôleur (Controller) qui lie la logique à la vue.
#          3. Démarrage de la boucle d'événements principale (mainloop).
# @author Romain Brouard et Paul Henry

window = Window()
Controller = Controller(window)
window.run()
