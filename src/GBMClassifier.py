from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.Model import Model


##
# @class GBMClassifier
# @brief Implémentation d'un classificateur Gradient Boosting Machine (Histogram-based).
# @extends Model
# @details Optimisé pour les grands jeux de données, utilise `HistGradientBoostingClassifier`
#          précédé d'une standardisation des données.
class GBMClassifier(Model):

    def __init__(self, logger, n_train, n_val):
        super().__init__(logger, n_train, n_val, model_name="Gradient Boosting")

    ##
    # @brief Crée le pipeline GBM.
    # @param kwargs Arguments supplémentaires pour `HistGradientBoostingClassifier`.
    # @return sklearn.pipeline.Pipeline Pipeline configuré (StandardScaler -> HistGradientBoostingClassifier).
    def _create_model(self, **kwargs):
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=100,
                max_leaf_nodes=31,
                random_state=42,
                verbose=0,
                **kwargs
            ),
        )
