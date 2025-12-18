import numpy as np
from PIL import Image


##
# @class ImageProcessor
# @brief Classe utilitaire pour le chargement et le prétraitement des images d'inférence.
class ImageProcessor:

    ##
    # @brief Charge une image depuis un fichier, la redimensionne et la normalise.
    # @details Convertit l'image en RGB, redimensionne en 32x32 (format CIFAR-10),
    #          et normalise les pixels (float entre 0.0 et 1.0).
    # @param file_path (str) Le chemin complet vers le fichier image.
    # @return np.ndarray Un tableau numpy de forme (32, 32, 3).
    # @throws ValueError Si l'image ne peut pas être chargée ou traitée.
    @staticmethod
    def load_and_preprocess(file_path: str) -> np.ndarray:
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            img_array = img_array.astype(np.float32) / 255.0

            return img_array

        except Exception as e:
            raise ValueError(f"Impossible de traiter l'image {file_path}: {str(e)}")
