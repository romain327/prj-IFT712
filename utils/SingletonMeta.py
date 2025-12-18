##
# @class SingletonMeta
# @brief Métaclasse implémentant le design pattern Singleton.
# @details Cette classe assure qu'une classe qui l'utilise comme métaclasse n'aura qu'une seule instance
#          partagée tout au long du cycle de vie de l'application.
# @extends type
class SingletonMeta(type):
    _instances = {}

    ##
    # @brief Méthode spéciale appelée lors de l'instanciation de la classe.
    # @details Vérifie si une instance de la classe existe déjà dans `_instances`.
    #          Si oui, retourne l'instance existante. Sinon, appelle le constructeur parent,
    #          stocke la nouvelle instance et la retourne.
    # @param cls La classe en cours d'instanciation.
    # @param args Arguments positionnels passés au constructeur.
    # @param kwargs Arguments nommés passés au constructeur.
    # @return L'instance unique de la classe.
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
