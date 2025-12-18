from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.Model import Model


##
# @class SVMClassifier
# @brief Implémentation d'un classificateur Support Vector Machine (SVM) à noyau gaussien.
# @extends Model
# @details Utilise un pipeline Scikit-learn incluant une standardisation,
#          une réduction de dimension (PCA) conservant 95% de variance, et un SVC à noyau RBF.
class SVMClassifier(Model):

    def __init__(self, logger, n_train, n_val):
        super().__init__(logger, n_train, n_val, model_name="SVM")

    ##
    # @brief Crée le pipeline SVM.
    # @param kwargs Arguments supplémentaires passés au constructeur SVC.
    # @return sklearn.pipeline.Pipeline Pipeline configuré (StandardScaler -> PCA -> SVC).
    def _create_model(self, **kwargs):
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=42,
                probability=True,
                verbose=0,
                **kwargs
            ),
        )
