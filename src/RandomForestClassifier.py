from sklearn.ensemble import RandomForestClassifier as SKLearnRF
from src.Model import Model


##
# @class RandomForestClassifier
# @brief Implémentation d'un classificateur Random Forest.
# @extends Model
# @details Ensemble de 100 arbres de décision, utilisant tous les cœurs CPU disponibles (n_jobs=-1).
class RandomForestClassifier(Model):
    def __init__(self, n_train, n_val):
        super().__init__(n_train, n_val, model_name="Random Forest")

    ##
    # @brief Configure et retourne le classificateur Random Forest.
    # @param kwargs Arguments supplémentaires pour `sklearn.ensemble.RandomForestClassifier`.
    # @return sklearn.ensemble.RandomForestClassifier Le modèle configuré.
    def _create_model(self, **kwargs):
        return SKLearnRF(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
            **kwargs
        )
