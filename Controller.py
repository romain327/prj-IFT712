import os
import threading

from utils.Logger import Logger
from utils.SingletonMeta import SingletonMeta
from utils.DataHandler import DataHandler
from utils.ImageProcessor import ImageProcessor

from src.DecisionTreeClassifier import DecisionTreeClassifier
from src.RandomForestClassifier import RandomForestClassifier
from src.LogisticRegressionClassifier import LogisticRegressionClassifier
from src.SVMClassifier import SVMClassifier
from src.GBMClassifier import GBMClassifier
from src.CNNClassifier import CNNClassifier


##
# @class Controller
# @brief Contrôleur principal de l'application (Pattern MVC).
# @details Cette classe fait le lien entre l'interface graphique (Window) et la logique métier (Modèles, DataHandler).
#          Elle gère les événements utilisateur, valide les entrées, et exécute les tâches lourdes (entraînement, test, inférence)
#          dans des threads séparés pour maintenir la réactivité de l'interface.
# @extends SingletonMeta
class Controller(metaclass=SingletonMeta):
    ##
    # @brief Constructeur de la classe Controller.
    # @details Initialise le logger, configure les callbacks des boutons de l'interface graphique
    #          et initialise le dictionnaire des classes de modèles disponibles.
    # @param window (Window) L'instance de la fenêtre principale de l'application.
    def __init__(self, window):
        self.window = window
        self.logger = Logger(self.window.get_root(), self.window.get_console())

        self.window.get_train_button().config(command=self.start_train)
        self.window.get_test_button().config(command=self.start_test)
        self.window.get_inference_button().config(command=self.start_classify)

        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.data_handler = None
        self.model = None
        self.training_counter = 0
        self.inference_path = None

        self.model_classes = {
            "CNN": CNNClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Logistic Regression": LogisticRegressionClassifier,
            "SVM": SVMClassifier,
            "Gradient Boosting": GBMClassifier,
        }

    ##
    # @brief Déclenche le processus d'entraînement.
    # @details Vérifie les entrées utilisateur via check_inputs(), désactive le bouton d'entraînement
    #          et lance l'exécution de `_train_model_thread` dans un thread séparé.
    def start_train(self):
        self.logger.log("Start training...", "INFO")
        if not self.check_inputs("train"):
            self.logger.log("Input error, aborting...", "ERROR")
        else:
            selected_model = self.window.get_selected_model()

            if selected_model not in self.model_classes:
                self.logger.log(
                    f"Model '{selected_model}' is not implemented yet!", "ERROR"
                )
                return

            self.training_counter += 1

            self.window.get_train_button().config(
                state="disabled", text="Training in progress..."
            )

            thread = threading.Thread(
                target=self._train_model_thread, args=(selected_model,)
            )
            thread.daemon = True
            thread.start()

    ##
    # @brief Logique métier de l'entraînement (exécutée dans un thread).
    # @details Charge les données via DataHandler, instancie le modèle sélectionné, lance l'entraînement
    #          et demande l'affichage des résultats une fois terminé.
    # @param model_name (str) Le nom du modèle à entraîner.
    def _train_model_thread(self, model_name):
        try:
            self.logger.log("Loading dataset...", "INFO")
            self.data_handler = DataHandler(self.logger, self.dataset)
            self.data_handler.load_data(normalize=True, flatten=True)

            self.logger.log(f"Initializing {model_name} model...", "INFO")
            model_class = self.model_classes[model_name]

            self.model = model_class(
                self.logger, int(self.train_data), int(self.val_data)
            )

            self.logger.log(f"Training {model_name} model...", "INFO")
            model_data, metrics_data = self.model.train(
                self.data_handler.get_data(), self.data_handler.get_class_names()
            )

            self.logger.log("Training completed successfully!", "SUCCESS")

            self.window.get_root().after(
                0, self._display_training_results, model_name, metrics_data
            )

        except Exception as e:
            self.logger.log(f"Training error: {str(e)}", "ERROR")
            import traceback

            self.logger.log(traceback.format_exc(), "ERROR")
        finally:
            self.window.get_root().after(0, self._reset_train_button)

    ##
    # @brief Met à jour l'interface graphique avec les résultats de l'entraînement.
    # @details Crée un nouvel onglet dans la fenêtre via `window.create_training_results_tab`.
    # @param model_name (str) Le nom du modèle entraîné.
    # @param metrics_data (dict) Les métriques et graphiques résultants de l'entraînement.
    def _display_training_results(self, model_name, metrics_data):
        try:
            training_history_fig = None
            if model_name == "CNN":
                training_history_fig = metrics_data.get("history_fig", None)

            confusion_matrix_fig = metrics_data.get("confusion_matrix_fig", None)

            tab_name = f"{model_name} #{self.training_counter}"

            self.window.create_training_results_tab(
                model_name=tab_name,
                metrics_data=metrics_data,
                confusion_matrix_fig=confusion_matrix_fig,
                history=training_history_fig,
            )

            self.logger.log(f"Results tab created: {tab_name}", "INFO")

        except Exception as e:
            self.logger.log(f"Error creating results tab: {str(e)}", "ERROR")
            import traceback

            self.logger.log(traceback.format_exc(), "ERROR")

    ##
    # @brief Réactive le bouton d'entraînement à la fin du thread.
    def _reset_train_button(self):
        self.window.get_train_button().config(state="normal", text="Train Model")

    ##
    # @brief Déclenche le processus de test.
    # @details Vérifie les entrées, désactive le bouton de test et lance le thread de test.
    def start_test(self):
        self.logger.log("Start Testing...", "INFO")
        if not self.check_inputs("test"):
            self.logger.log("Input error, aborting...", "ERROR")
        else:
            selected_model = self.window.get_selected_model()
            self.window.get_test_button().config(
                state="disabled", text="Testing in progress..."
            )

            thread = threading.Thread(
                target=self._test_model_thread, args=(selected_model,)
            )
            thread.daemon = True
            thread.start()

    ##
    # @brief Logique métier du test (exécutée dans un thread).
    # @details Charge le modèle sélectionné (sans l'entraîner) et lance la méthode `test` du modèle.
    # @param model_name (str) Le nom du modèle à tester.
    def _test_model_thread(self, model_name):
        try:
            self.logger.log("Loading dataset for testing...", "INFO")
            if self.data_handler is None:
                self.data_handler = DataHandler(self.logger, self.dataset)
                self.data_handler.load_data(normalize=True, flatten=True)

            self.logger.log(f"Initializing {model_name} wrapper...", "INFO")
            model_class = self.model_classes[model_name]
            self.model = model_class(
                self.logger, 0, 0
            )  # Pas besoin de n_train/n_val pour tester

            self.logger.log(f"Testing {model_name} model...", "INFO")
            metrics_data = self.model.test(
                self.data_handler.get_data_dict(), self.data_handler.get_class_names()
            )

            self.logger.log("Testing completed successfully!", "SUCCESS")
            self.window.get_root().after(
                0, self._display_test_results, model_name, metrics_data
            )

        except Exception as e:
            self.logger.log(f"Testing error: {str(e)}", "ERROR")
            import traceback

            self.logger.log(traceback.format_exc(), "ERROR")
        finally:
            self.window.get_root().after(0, self._reset_test_button)

    ##
    # @brief Affiche les résultats du test dans l'interface graphique.
    # @param model_name (str) Nom du modèle testé.
    # @param metrics_data (dict) Dictionnaire des métriques de test.
    def _display_test_results(self, model_name, metrics_data):
        try:
            confusion_matrix_fig = metrics_data.get("confusion_matrix_fig", None)
            tab_name = f"TEST: {model_name}"
            self.window.create_training_results_tab(
                model_name=tab_name,
                metrics_data=metrics_data,
                confusion_matrix_fig=confusion_matrix_fig,
                history=None,
            )
            self.logger.log(f"Test results tab created: {tab_name}", "INFO")
        except Exception as e:
            self.logger.log(f"Error creating results tab: {str(e)}", "ERROR")

    ##
    # @brief Réactive le bouton de test à la fin du thread.
    def _reset_test_button(self):
        self.window.get_test_button().config(state="normal", text="Test Model")

    ##
    # @brief Déclenche le processus d'inférence (classification).
    # @details Vérifie les entrées, désactive le bouton d'inférence et lance le thread associé.
    def start_classify(self):
        self.logger.log("Start Classify...", "INFO")
        if not self.check_inputs("classify"):
            self.logger.log("Input error, aborting...", "ERROR")
        else:
            selected_model = self.window.get_selected_model()
            self.window.get_inference_button().config(
                state="disabled", text="Classifying..."
            )

            thread = threading.Thread(
                target=self._classify_model_thread, args=(selected_model,)
            )
            thread.daemon = True
            thread.start()

    ##
    # @brief Logique métier de l'inférence (exécutée dans un thread).
    # @details Charge le modèle, prépare les images (fichier unique ou dossier) et exécute la prédiction.
    # @param model_name (str) Le nom du modèle à utiliser.
    def _classify_model_thread(self, model_name):
        try:
            self.logger.log(f"Loading {model_name} for inference...", "INFO")
            model_class = self.model_classes[model_name]
            self.model = model_class(self.logger, 0, 0)
            self.model.load_model()

            files_to_process = []
            if os.path.isfile(self.inference_path):
                files_to_process.append(self.inference_path)
            elif os.path.isdir(self.inference_path):
                valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
                files_to_process = [
                    os.path.join(self.inference_path, f)
                    for f in os.listdir(self.inference_path)
                    if f.lower().endswith(valid_extensions)
                ]

            if not files_to_process:
                self.logger.log("No valid images found!", "ERROR")
                return

            self.logger.log(f"Processing {len(files_to_process)} image(s)...", "INFO")

            if self.data_handler is not None:
                class_names = self.data_handler.get_class_names()
            else:
                class_names = [
                    "Airplane",
                    "Automobile",
                    "Bird",
                    "Cat",
                    "Deer",
                    "Dog",
                    "Frog",
                    "Horse",
                    "Ship",
                    "Truck",
                ]

            results_data = []

            for file_path in files_to_process:
                try:
                    img_array = ImageProcessor.load_and_preprocess(file_path)

                    prediction_index = self.model.classify(img_array)

                    if 0 <= prediction_index < len(class_names):
                        predicted_label = class_names[prediction_index]
                    else:
                        predicted_label = f"Unknown ({prediction_index})"

                    results_data.append(
                        (os.path.basename(file_path), predicted_label, img_array)
                    )

                    self.logger.log(
                        f"{os.path.basename(file_path)} -> {predicted_label}", "RESULT"
                    )

                except Exception as img_err:
                    self.logger.log(
                        f"Error processing {os.path.basename(file_path)}: {str(img_err)}",
                        "ERROR",
                    )

            self.window.get_root().after(
                0, self._display_classification_results, results_data
            )

        except Exception as e:
            self.logger.log(f"Inference error: {str(e)}", "ERROR")
            import traceback

            self.logger.log(traceback.format_exc(), "ERROR")
        finally:
            self.window.get_root().after(0, self._reset_classify_button)

    ##
    # @brief Affiche les résultats de classification dans l'interface.
    # @param results_data (list) Liste de tuples contenant les noms de fichiers, prédictions et images.
    def _display_classification_results(self, results_data):
        try:
            self.window.create_inference_results_tab(results_data)
            self.logger.log("Inference results tab displayed.", "INFO")
        except Exception as e:
            self.logger.log(f"UI Error: {str(e)}", "ERROR")

    ##
    # @brief Réactive le bouton d'inférence à la fin du thread.
    def _reset_classify_button(self):
        self.window.get_inference_button().config(
            state="normal", text="Classify Image(s)"
        )

    ##
    # @brief Valide les entrées du formulaire de l'interface graphique.
    # @param action (str) L'action demandée : "train", "test" ou "classify".
    # @return (bool) True si toutes les entrées requises sont valides, False sinon.
    def check_inputs(self, action):
        match action:
            case "train":
                self.dataset = self.window.get_dataset_folder().get()
                self.train_data = self.window.get_train_data().get()
                self.val_data = self.window.get_val_data().get()

                if (not os.path.exists(self.dataset)) or self.dataset == "":
                    self.logger.log("Dataset folder not found", "ERROR")
                    return False

                if self.train_data == "" or not self.train_data.isdigit():
                    self.logger.log("Missing or invalid train data number", "ERROR")
                    return False

                if self.val_data == "" or not self.val_data.isdigit():
                    self.logger.log(
                        "Missing or invalid validation data number", "ERROR"
                    )
                    return False

            case "test":
                self.dataset = self.window.get_dataset_folder().get()
                if (not os.path.exists(self.dataset)) or self.dataset == "":
                    self.logger.log(
                        "Dataset folder not found (needed for test data)", "ERROR"
                    )
                    return False

            case "classify":
                self.inference_path = self.window.get_inference_data().get()

                if not os.path.exists(self.inference_path) or self.inference_path == "":
                    self.logger.log("Inference file/folder not found", "ERROR")
                    return False

        return True
