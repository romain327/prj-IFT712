import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import numpy as np
import csv


##
# @class Window
# @brief Classe gérant l'interface graphique (GUI) de l'application.
# @details Cette classe initialise la fenêtre principale (Tkinter), construit les onglets,
#          les formulaires de configuration (entraînement, test, inférence) et gère l'affichage
#          des résultats (graphiques, logs, images).
class Window:
    ##
    # @brief Constructeur de la classe Window.
    # @details Initialise l'environnement graphique, crée le dossier `_models` si nécessaire,
    #          et construit tous les widgets de l'interface (onglets, boutons, champs de saisie, console).
    def __init__(self):
        os.makedirs("_models", exist_ok=True)

        self.__root = Tk()
        self.__root.title("Classifier interface")
        self.__root.geometry("1920x1080")
        self.__root.resizable(width=True, height=True)
        self.__root.grid_columnconfigure(0, weight=1)
        self.__root.grid_columnconfigure(1, weight=6)
        self.__root.grid_rowconfigure(0, weight=1)

        style = ttk.Style()

        # Forçage du thème "clam" pour forcer le mode clair sur macOS
        style.theme_use("clam")

        self.__root.option_add("*foreground", "black")
        self.__root.option_add("*background", "#f0f0f0")

        self.__root.option_add("*Entry.background", "white")
        self.__root.option_add("*Entry.foreground", "black")
        self.__root.option_add("*Text.background", "white")
        self.__root.option_add("*Text.foreground", "black")
        self.__root.option_add("*Button.foreground", "black")

        style.configure(".", foreground="black", background="#f0f0f0")
        style.configure("TLabel", foreground="black", background="#f0f0f0")
        style.configure("TNotebook", background="#f0f0f0")
        style.configure("TNotebook.Tab", foreground="black")

        self.__models = "_models"

        self.__form = Frame(self.__root)
        self.__form.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.__tab_system = ttk.Notebook(self.__root)
        self.__tab_system.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.__logs_tab = Frame(self.__tab_system)
        self.__logs_tab.grid_columnconfigure(1, weight=1)
        self.__tab_system.add(self.__logs_tab, text="logs")

        self.__console = tk.Text(self.__logs_tab, wrap=WORD)
        self.__scrollbar = tk.Scrollbar(self.__logs_tab, command=self.__console.yview)
        self.__console.config(yscrollcommand=self.__scrollbar.set)
        self.__console.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.__scrollbar.grid(row=0, column=1, sticky="nse", padx=5, pady=5)

        self.__select_model_area = LabelFrame(
            self.__form,
            text="Select model",
            padx=15,
            pady=15,
            font=("Arial", 10, "bold"),
        )
        self.__select_model_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.__select_model_area.grid_columnconfigure(1, weight=1)

        self.__model_label = Label(
            self.__select_model_area,
            text="Select model:",
            font=("Arial", 9),
            width=18,
            anchor="w",
        )
        self.__model_label.grid(row=0, column=0, padx=5, pady=8, sticky="w")
        self.__model_information = Label(
            self.__select_model_area,
            text="Model selected: CNN",
            font=("Arial", 10),
            bg="#e3f2fd",
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=5,
        )
        self.__model_information.grid(
            row=4, column=0, columnspan=2, padx=5, pady=(0, 10), sticky="ew"
        )

        self.__model_var = tk.StringVar()
        self.__model_dropdown = ttk.Combobox(
            self.__select_model_area, textvariable=self.__model_var, font=("Arial", 9)
        )
        self.__model_dropdown["values"] = [
            "CNN",
            "Decision Tree",
            "Random Forest",
            "Logistic Regression",
            "SVM",
            "Gradient Boosting",
        ]
        self.__model_dropdown.current(0)
        self.__model_dropdown.grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        self.__model_dropdown.bind("<<ComboboxSelected>>", self.update_model_info)

        # ===== TRAINING AREA =====
        self.__train_area = LabelFrame(
            self.__form, text="Train Data", padx=15, pady=15, font=("Arial", 10, "bold")
        )
        self.__train_area.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.__train_area.grid_columnconfigure(1, weight=1)

        # Dataset folder
        self.__dataset_folder_label = Label(
            self.__train_area,
            text="Dataset folder:",
            font=("Arial", 9),
            width=18,
            anchor="w",
        )
        self.__dataset_folder = tk.StringVar()
        self.__dataset_folder_entry = Entry(
            self.__train_area, textvariable=self.__dataset_folder, font=("Arial", 9)
        )
        self.__dataset_folder_button = Button(
            self.__train_area, text="Browse", width=10, command=self.select_dataset
        )
        self.__dataset_folder_label.grid(row=0, column=0, padx=5, pady=8, sticky="w")
        self.__dataset_folder_entry.grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        self.__dataset_folder_button.grid(row=0, column=2, padx=5, pady=8)

        # Training data size
        self.__train_data_label = Label(
            self.__train_area,
            text="Training data size:",
            font=("Arial", 9),
            width=18,
            anchor="w",
        )
        self.__train_data = tk.StringVar()
        self.__train_data_entry = Entry(
            self.__train_area,
            textvariable=self.__train_data,
            font=("Arial", 9),
            width=15,
        )
        self.__train_data_label.grid(row=1, column=0, padx=5, pady=8, sticky="w")
        self.__train_data_entry.grid(row=1, column=1, padx=5, pady=8, sticky="w")

        # Validation data size
        self.__val_data_label = Label(
            self.__train_area,
            text="Validation data size:",
            font=("Arial", 9),
            width=18,
            anchor="w",
        )
        self.__val_data = tk.StringVar()
        self.__val_data_entry = Entry(
            self.__train_area, textvariable=self.__val_data, font=("Arial", 9), width=15
        )
        self.__val_data_label.grid(row=2, column=0, padx=5, pady=8, sticky="w")
        self.__val_data_entry.grid(row=2, column=1, padx=5, pady=8, sticky="w")

        # Buttons frame for training
        self.__train_buttons_frame = Frame(self.__train_area)
        self.__train_buttons_frame.grid(
            row=3, column=0, columnspan=2, pady=(15, 5), sticky="ew"
        )
        self.__train_buttons_frame.grid_columnconfigure(0, weight=1)
        self.__train_buttons_frame.grid_columnconfigure(1, weight=1)
        self.__train_button = Button(
            self.__train_buttons_frame,
            text="Train Model",
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            height=2,
        )
        self.__train_button.grid(row=0, column=0, padx=5, sticky="ew")

        # ===== TEST AREA =====
        self.__test_area = LabelFrame(
            self.__form, text="Test Data", padx=15, pady=15, font=("Arial", 10, "bold")
        )
        self.__test_area.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.__test_area.grid_columnconfigure(1, weight=1)

        self.__test_buttons_frame = Frame(self.__test_area)
        self.__test_buttons_frame.grid(
            row=0, column=0, columnspan=2, pady=(15, 5), sticky="ew"
        )
        self.__test_buttons_frame.grid_columnconfigure(0, weight=1)
        self.__test_buttons_frame.grid_columnconfigure(1, weight=1)
        self.__test_button = Button(
            self.__test_buttons_frame,
            text="Test Model",
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            height=2,
        )
        self.__test_button.grid(row=0, column=0, padx=5, sticky="ew")

        # ===== INFERENCE AREA =====
        self.__inference_area = LabelFrame(
            self.__form, text="Inference", padx=15, pady=15, font=("Arial", 10, "bold")
        )
        self.__inference_area.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        self.__inference_area.grid_columnconfigure(1, weight=1)

        # Inference data type
        self.__inference_data_type_label = Label(
            self.__inference_area,
            text="Inference data type:",
            font=("Arial", 9),
            width=18,
            anchor="w",
        )
        self.__is_file = tk.BooleanVar()
        self.__inference_type_frame = Frame(self.__inference_area)

        self.__inference_data_type_file = ttk.Radiobutton(
            self.__inference_type_frame,
            text="File",
            variable=self.__is_file,
            value=True,
        )
        self.__inference_data_type_folder = ttk.Radiobutton(
            self.__inference_type_frame,
            text="Folder",
            variable=self.__is_file,
            value=False,
        )

        self.__inference_data_type_file.pack(side="left", padx=5)
        self.__inference_data_type_folder.pack(side="left", padx=5)

        self.__inference_data_type_label.grid(
            row=0, column=0, padx=5, pady=8, sticky="w"
        )
        self.__inference_type_frame.grid(row=0, column=1, padx=5, pady=8, sticky="w")

        # Inference data path
        self.__inference_data_label = Label(
            self.__inference_area,
            text="Inference data:",
            font=("Arial", 9),
            width=18,
            anchor="w",
        )
        self.__inference_data = tk.StringVar()
        self.__inference_data_entry = Entry(
            self.__inference_area, textvariable=self.__inference_data, font=("Arial", 9)
        )
        self.__inference_data_button = Button(
            self.__inference_area,
            text="Browse",
            width=10,
            command=self.select_inference_data,
        )

        self.__inference_data_label.grid(row=1, column=0, padx=5, pady=8, sticky="w")
        self.__inference_data_entry.grid(row=1, column=1, padx=5, pady=8, sticky="ew")
        self.__inference_data_button.grid(row=1, column=2, padx=5, pady=8)

        self.__inference_buttons_frame = Frame(self.__inference_area)
        self.__inference_buttons_frame.grid(
            row=2, column=0, columnspan=2, pady=(15, 5), sticky="ew"
        )
        self.__inference_buttons_frame.grid_columnconfigure(0, weight=1)
        self.__inference_buttons_frame.grid_columnconfigure(1, weight=1)
        self.__inference_button = Button(
            self.__inference_buttons_frame,
            text="Classify Image(s)",
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            height=2,
        )
        self.__inference_button.grid(row=0, column=0, padx=5, sticky="ew")

        self.__error_label = Label(self.__form, text="", font=("Arial", 10, "bold"))
        self.__error_label.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        if not os.path.exists(os.path.join(self.__models, "CNN.pth")):
            self.__error_label.config(
                text="Selected model (CNN) does not exist, please train it first !",
                fg="red",
            )

    ##
    # @brief Retourne l'objet racine Tkinter.
    # @return (tk.Tk) L'objet fenêtre principale.
    def get_root(self):
        return self.__root

    ##
    # @brief Retourne le widget de texte utilisé pour la console de logs.
    # @return (tk.Text) Le widget de texte des logs.
    def get_console(self):
        return self.__console

    ##
    # @brief Retourne le bouton d'entraînement.
    # @return (tk.Button) Le bouton "Train Model".
    def get_train_button(self):
        return self.__train_button

    ##
    # @brief Retourne la variable Tkinter contenant le chemin du dossier dataset.
    # @return (tk.StringVar) La variable liée au champ dataset.
    def get_dataset_folder(self):
        return self.__dataset_folder

    ##
    # @brief Retourne la variable Tkinter contenant la taille des données d'entraînement.
    # @return (tk.StringVar) La variable de taille d'entraînement.
    def get_train_data(self):
        return self.__train_data

    ##
    # @brief Retourne la variable Tkinter contenant la taille des données de validation.
    # @return (tk.StringVar) La variable de taille de validation.
    def get_val_data(self):
        return self.__val_data

    ##
    # @brief Retourne le bouton de test.
    # @return (tk.Button) Le bouton "Test Model".
    def get_test_button(self):
        return self.__test_button

    ##
    # @brief Retourne le bouton d'inférence.
    # @return (tk.Button) Le bouton "Classify Image(s)".
    def get_inference_button(self):
        return self.__inference_button

    ##
    # @brief Met à jour les informations affichées lors de la sélection d'un modèle.
    # @details Vérifie l'existence du modèle sur le disque et affiche un message d'erreur si nécessaire.
    # @param event L'événement Tkinter déclencheur (changement de sélection).
    def update_model_info(self, event):
        selected_model = self.__model_var.get()
        self.__model_information.config(text=f"Model selected: {selected_model}")
        self.__error_label.config(text="", fg="red")
        if selected_model == "CNN":
            if not os.path.exists(os.path.join(self.__models, selected_model + ".pth")):
                self.__error_label.config(
                    text="Selected model ("
                    + selected_model
                    + ") does not exist, please train it first !",
                    fg="red",
                )
        else:
            if not os.path.exists(os.path.join(self.__models, selected_model + ".pkl")):
                self.__error_label.config(
                    text="Selected model ("
                    + selected_model
                    + ") does not exist, please train it first !",
                    fg="red",
                )

    ##
    # @brief Crée et ajoute un onglet affichant les résultats d'un entraînement.
    # @details Construit l'interface pour afficher les métriques (précision, rappel, F1),
    #          le rapport de classification, la matrice de confusion et l'historique (si applicable).
    # @param model_name (str) Le nom du modèle entraîné.
    # @param metrics_data (dict) Dictionnaire contenant les métriques calculées.
    # @param confusion_matrix_fig (matplotlib.figure.Figure, optional) Figure de la matrice de confusion.
    # @param history (matplotlib.figure.Figure, optional) Figure de l'historique d'entraînement (pour CNN).
    def create_training_results_tab(
        self, model_name, metrics_data, confusion_matrix_fig=None, history=None
    ):
        results_tab = Frame(self.__tab_system)
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(1, weight=1)

        title_label = Label(
            results_tab,
            text=f"Résultats d'entraînement - {model_name}",
            font=("Arial", 14, "bold"),
            bg="#e8f5e9",
            pady=10,
        )
        title_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        canvas = tk.Canvas(results_tab, bg="white")
        scrollbar = tk.Scrollbar(results_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        metrics_frame = LabelFrame(
            scrollable_frame,
            text="Métriques principales",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=15,
            bg="white",
        )
        metrics_frame.pack(fill="x", padx=20, pady=10)

        metrics_to_display = [
            ("Accuracy", metrics_data.get("accuracy", 0)),
            ("Precision", metrics_data.get("precision_macro", 0)),
            ("Recall", metrics_data.get("recall_macro", 0)),
            ("F1-Score", metrics_data.get("f1_macro", 0)),
        ]

        for i, (metric_name, value) in enumerate(metrics_to_display):
            metric_container = Frame(
                metrics_frame,
                bg="#e3f2fd",
                relief="solid",
                borderwidth=1,
                padx=15,
                pady=10,
            )
            metric_container.grid(
                row=i // 2, column=i % 2, padx=10, pady=10, sticky="ew"
            )

            Label(
                metric_container,
                text=metric_name,
                font=("Arial", 10),
                bg="#e3f2fd",
                anchor="w",
            ).pack(anchor="w")
            Label(
                metric_container,
                text=f"{value:.4f}",
                font=("Arial", 12, "bold"),
                bg="#e3f2fd",
                fg="#1976d2",
            ).pack(anchor="w")

        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)

        if "classification_report" in metrics_data:
            report_frame = LabelFrame(
                scrollable_frame,
                text="Rapport de classification détaillé",
                font=("Arial", 11, "bold"),
                padx=20,
                pady=15,
                bg="white",
            )
            report_frame.pack(fill="both", expand=True, padx=20, pady=10)

            report_text = tk.Text(
                report_frame, wrap=WORD, height=15, font=("Courier", 9)
            )
            report_scrollbar = tk.Scrollbar(report_frame, command=report_text.yview)
            report_text.config(yscrollcommand=report_scrollbar.set)

            report_text.insert("1.0", metrics_data["classification_report"])
            report_text.config(state="disabled")

            report_text.pack(side="left", fill="both", expand=True)
            report_scrollbar.pack(side="right", fill="y")

        if confusion_matrix_fig:
            cm_frame = LabelFrame(
                scrollable_frame,
                text="Matrice de confusion",
                font=("Arial", 11, "bold"),
                padx=20,
                pady=15,
                bg="white",
            )
            cm_frame.pack(fill="both", expand=True, padx=20, pady=10)

            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

                canvas_plot = FigureCanvasTkAgg(confusion_matrix_fig, master=cm_frame)
                canvas_plot.draw()
                canvas_plot.get_tk_widget().pack(fill="both", expand=True)
            except ImportError:
                Label(
                    cm_frame,
                    text="Matplotlib backend non disponible",
                    font=("Arial", 10),
                    fg="red",
                ).pack()

        if history:
            hist_frame = LabelFrame(
                scrollable_frame,
                text="CNN history",
                font=("Arial", 11, "bold"),
                padx=20,
                pady=15,
                bg="white",
            )
            hist_frame.pack(fill="both", expand=True, padx=20, pady=10)

            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

                canvas_plot = FigureCanvasTkAgg(history, master=hist_frame)
                canvas_plot.draw()
                canvas_plot.get_tk_widget().pack(fill="both", expand=True)
            except ImportError:
                Label(
                    hist_frame,
                    text="Matplotlib backend non disponible",
                    font=("Arial", 10),
                    fg="red",
                ).pack()

        button_frame = Frame(scrollable_frame, bg="white")
        button_frame.pack(fill="x", padx=20, pady=20)

        close_button = Button(
            button_frame,
            text="Fermer cet onglet",
            font=("Arial", 10, "bold"),
            bg="#f44336",
            fg="white",
            command=lambda: self.close_tab(results_tab),
        )
        close_button.pack(side="right")

        canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 10))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 10))

        tab_name = f"Training: {model_name}"
        self.__tab_system.add(results_tab, text=tab_name)
        self.__tab_system.select(results_tab)

        self.get_console().insert(
            tk.END, f"\n[INFO] Onglet de résultats créé: {tab_name}"
        )
        self.get_console().see(tk.END)

    ##
    # @brief Crée et ajoute un onglet affichant les résultats d'inférence.
    # @details Affiche une liste déroulante d'images avec leur prédiction associée.
    # @param results (list) Liste de tuples (nom_fichier, prédiction, tableau_image).
    def create_inference_results_tab(self, results):
        results_tab = Frame(self.__tab_system)
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(1, weight=1)

        title_label = Label(
            results_tab,
            text=f"Résultats d'inférence ({len(results)} images)",
            font=("Arial", 14, "bold"),
            bg="#e1f5fe",
            pady=10,
        )
        title_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        canvas = tk.Canvas(results_tab, bg="white")
        scrollbar = tk.Scrollbar(results_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="white")

        frame_window_id = canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw"
        )

        def on_canvas_configure(event):
            canvas.itemconfig(frame_window_id, width=event.width)

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", on_canvas_configure)
        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame.image_refs = []

        headers_frame = Frame(scrollable_frame, bg="#eeeeee", pady=5)
        headers_frame.pack(fill="x", padx=0, pady=0)

        headers_frame.grid_columnconfigure(1, weight=1)

        Label(
            headers_frame,
            text="Image",
            width=10,
            font=("Arial", 10, "bold"),
            bg="#eeeeee",
            anchor="center",
        ).grid(row=0, column=0, padx=10)
        Label(
            headers_frame,
            text="Fichier",
            font=("Arial", 10, "bold"),
            bg="#eeeeee",
            anchor="w",
        ).grid(row=0, column=1, padx=10, sticky="w")
        Label(
            headers_frame,
            text="Prédiction",
            width=15,
            font=("Arial", 10, "bold"),
            bg="#eeeeee",
            anchor="center",
        ).grid(row=0, column=2, padx=10)

        for filename, prediction, img_array in results:
            row_frame = Frame(
                scrollable_frame, bg="white", pady=10, borderwidth=1, relief="solid"
            )
            row_frame.pack(fill="x", padx=10, pady=5)

            row_frame.grid_columnconfigure(1, weight=1)

            img_data = (img_array * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_data)
            pil_img = pil_img.resize((64, 64), Image.Resampling.NEAREST)
            tk_img = ImageTk.PhotoImage(pil_img)

            scrollable_frame.image_refs.append(tk_img)

            img_label = Label(row_frame, image=tk_img, bg="white")
            img_label.grid(row=0, column=0, padx=10)

            name_label = Label(
                row_frame, text=filename, anchor="w", font=("Arial", 10), bg="white"
            )
            name_label.grid(row=0, column=1, padx=10, sticky="ew")

            pred_label = Label(
                row_frame,
                text=prediction,
                width=15,
                font=("Arial", 11, "bold"),
                fg="#2e7d32",
                bg="white",
            )
            pred_label.grid(row=0, column=2, padx=10)

        button_frame = Frame(results_tab)
        button_frame.grid(row=2, column=0, pady=10, sticky="e", padx=20)

        export_button = Button(
            button_frame,
            text="Exporter en CSV",
            font=("Arial", 10, "bold"),
            bg="#2196F3",
            fg="white",
            command=lambda: self.export_results_to_csv(results),
        )
        export_button.pack(side="left", padx=10)

        close_button = Button(
            button_frame, text="Fermer", command=lambda: self.close_tab(results_tab)
        )
        close_button.pack(side="left")

        canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 10))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 10))

        self.__tab_system.add(results_tab, text="Inférence")
        self.__tab_system.select(results_tab)

    ##
    # @brief Ouvre une boîte de dialogue pour enregistrer les résultats d'inférence en CSV.
    # @param results (list) Liste de tuples (nom_fichier, prédiction, image_array).
    def export_results_to_csv(self, results):
        file_path = fd.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Fichiers CSV", "*.csv")],
            title="Enregistrer les résultats de classification",
        )

        if file_path:
            try:
                with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(["image", "classe"])

                    for fname, pred, _ in results:
                        writer.writerow([fname, pred])

                print(f"[INFO] Résultats exportés avec succès dans : {file_path}")
            except Exception as e:
                print(f"[ERROR] Impossible d'exporter le fichier CSV : {str(e)}")

    ##
    # @brief Ouvre un sélecteur de dossier pour choisir le dataset.
    def select_dataset(self):
        self.__dataset_folder.set(fd.askdirectory(title="Select dataset folder"))

    ##
    # @brief Ferme un onglet spécifique.
    # @param tab (tk.Widget) Le widget représentant l'onglet à fermer.
    def close_tab(self, tab):
        self.__tab_system.forget(tab)

    ##
    # @brief Retourne le gestionnaire d'onglets (Notebook).
    # @return (ttk.Notebook) Le gestionnaire d'onglets principal.
    def get_tab_system(self):
        return self.__tab_system

    ##
    # @brief Retourne le nom du modèle actuellement sélectionné.
    # @return (str) Le nom du modèle.
    def get_selected_model(self):
        return self.__model_var.get()

    ##
    # @brief Ouvre un sélecteur de fichier ou de dossier pour l'inférence selon le mode choisi.
    def select_inference_data(self):
        if self.__is_file:
            self.__inference_data.set(
                fd.askopenfilename(
                    filetypes=[("png files", "*.png"), ("jpg files", "*.jpg")]
                )
            )
        else:
            self.__inference_data.set(fd.askdirectory(title="Select dataset folder"))

    ##
    # @brief Retourne la variable Tkinter contenant le chemin des données d'inférence.
    # @return (tk.StringVar) La variable liée au champ d'inférence.
    def get_inference_data(self):
        return self.__inference_data

    ##
    # @brief Lance la boucle principale de l'interface graphique.
    # @details Cette méthode bloque l'exécution jusqu'à la fermeture de la fenêtre.
    def run(self):
        self.__root.mainloop()
