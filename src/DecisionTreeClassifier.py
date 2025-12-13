from sklearn.tree import DecisionTreeClassifier as SKLearnDT
from src.Model import Model


##
# @class DecisionTreeClassifier
# @brief Implémentation d'un classificateur par Arbre de Décision.
# @extends Model
# @details Utilise `sklearn.tree.DecisionTreeClassifier` avec une profondeur maximale de 30
#          et le critère de Gini.
class DecisionTreeClassifier(Model):
    """Classificateur Decision Tree adapté."""

    def __init__(self, n_train, n_val):
        super().__init__(n_train, n_val, model_name="DecisionTree")

    ##
    # @brief Instancie l'arbre de décision avec des hyperparamètres pré-définis.
    # @param kwargs Arguments supplémentaires pour le constructeur DecisionTreeClassifier.
    # @return sklearn.tree.DecisionTreeClassifier Le modèle configuré.
    def _create_model(self, **kwargs):
        return SKLearnDT(
            criterion="gini",
            max_depth=30,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            **kwargs
        )
