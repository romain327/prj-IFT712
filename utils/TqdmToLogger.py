##
# @class TqdmToLogger
# @brief Redirige la sortie standard (utilisée par tqdm) vers le logger de l'application.
# @details Permet d'afficher les barres de progression textuelles générées par la bibliothèque tqdm
#          directement dans le widget Text de l'interface Tkinter, en gérant le rafraîchissement des lignes.
class TqdmToLogger:
    ##
    # @brief Constructeur de la classe TqdmToLogger.
    # @param logger (Logger) L'instance du logger principal de l'application.
    # @param tag (str) Le tag de couleur à utiliser pour l'affichage (par défaut "INFO").
    def __init__(self, logger, tag="INFO"):
        self.__logger = logger
        self.__tag = tag
        self.__first_line = True

    ##
    # @brief Écrit le contenu du buffer dans le logger.
    # @details Cette méthode est appelée par tqdm. Elle détecte si le buffer contient du texte,
    #          et demande au logger de l'afficher. Gère le mode `buffered` pour éviter
    #          l'empilement des lignes de progression.
    # @param buf (str) La chaîne de caractères envoyée par tqdm.
    def write(self, buf):
        if buf.strip():
            self.__logger.log(buf.strip(), self.__tag, buffered=not self.__first_line)
            self.__first_line = False

    ##
    # @brief Méthode de vidage du buffer (requise par l'interface file-like).
    # @details Ne fait rien dans cette implémentation car l'affichage est géré directement dans write().
    def flush(self):
        pass
