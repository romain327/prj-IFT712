from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.Model import Model


##
# @class LogisticRegressionClassifier
# @brief Implémentation d'un classificateur par Régression Logistique.
# @extends Model
# @details Utilise un pipeline avec Standardisation et PCA (95% variance) avant la régression logistique
#          (solver lbfgs).
class LogisticRegressionClassifier(Model):

    def __init__(self, logger, n_train, n_val):
        super().__init__(logger, n_train, n_val, model_name="Logistic Regression")

    ##
    # @brief Crée le pipeline de Régression Logistique.
    # @param kwargs Arguments supplémentaires pour `LogisticRegression`.
    # @return sklearn.pipeline.Pipeline Pipeline configuré (StandardScaler -> PCA -> LogisticRegression).
    def _create_model(self, **kwargs):
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=42,
                n_jobs=-1,
                **kwargs
            ),
        )
