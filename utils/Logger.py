from tkinter import *
import time

from utils.SingletonMeta import SingletonMeta


##
# @brief Retourne la couleur Tkinter associée à un tag de log donné.
# @param tag (str) Le niveau ou type de log (ex: "WARNING", "ERROR", "SUCCESS", "RESULT").
# @return (str) Le nom de la couleur correspondante (ex: "orange", "red", "green").
def get_log_color(tag):
    match tag:
        case "WARNING":
            return "orange"
        case "ERROR":
            return "red"
        case "SUCCESS":
            return "green"
        case "RESULT":
            return "blue"
        case "TQDM":
            return "purple"
        case _:
            return "black"


##
# @class Logger
# @brief Classe gérant l'affichage des logs dans la console de l'interface graphique.
# @details Utilise le pattern Singleton pour être accessible globalement depuis n'importe quelle partie du code.
#          Gère l'écriture thread-safe (via update Tkinter) dans un widget Text.
# @extends SingletonMeta
class Logger(metaclass=SingletonMeta):

    ##
    # @brief Constructeur de la classe Logger.
    # @param root (tk.Tk) L'objet racine de la fenêtre (nécessaire pour forcer la mise à jour graphique).
    # @param console (tk.Text) Le widget Text de l'interface où les logs seront insérés.
    def __init__(self, root, console):
        self.console = console
        self.root = root

    ##
    # @brief Ajoute un message dans la console de logs.
    # @details Active temporairement l'édition du widget, insère le message avec un timestamp et la couleur appropriée,
    #          scrolle automatiquement vers le bas, puis désactive l'édition.
    # @param message (str) Le texte du message à logger.
    # @param tag (str) Le type de message déterminant la couleur (par défaut "INFO").
    # @param buffered (bool) Si True, supprime les lignes précédentes avant d'écrire (utile pour les barres de progression dynamiques).
    def log(self, message, tag="INFO", buffered=False):
        self.console.config(state=NORMAL)
        if buffered:
            self.console.delete("end-2l", "end-1c")
        self.console.tag_configure(tag, foreground=get_log_color(tag))
        m = time.strftime("[%H:%M:%S]", time.localtime()) + " " + message + "\n"
        self.console.insert(END, m, tag)
        self.console.see(END)
        self.console.config(state=DISABLED)
        self.root.update()
