import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
# Add clustering metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from mpl_toolkits.mplot3d import Axes3D
import umap  # Add UMAP import


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None

        # Neural network configuration
        self.layer_config = []

        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()

        # Top row for dataset selection and loading
        top_row = QHBoxLayout()

        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "California Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        top_row.addWidget(QLabel("Dataset:"))
        top_row.addWidget(self.dataset_combo)
        top_row.addWidget(self.load_btn)
        data_layout.addLayout(top_row)

        # Middle row for data split controls
        split_group = QGroupBox("Data Split Configuration")
        split_layout = QGridLayout()

        # Train split
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
        self.train_split.setValue(0.7)
        self.train_split.setSingleStep(0.05)
        self.train_split.valueChanged.connect(self.update_splits)
        split_layout.addWidget(QLabel("Train Split:"), 0, 0)
        split_layout.addWidget(self.train_split, 0, 1)

        # Validation split
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.0, 0.3)
        self.val_split.setValue(0.15)
        self.val_split.setSingleStep(0.05)
        self.val_split.valueChanged.connect(self.update_splits)
        split_layout.addWidget(QLabel("Validation Split:"), 0, 2)
        split_layout.addWidget(self.val_split, 0, 3)

        # Test split
        self.test_split = QDoubleSpinBox()
        self.test_split.setRange(0.0, 0.3)
        self.test_split.setValue(0.15)
        self.test_split.setSingleStep(0.05)
        self.test_split.setReadOnly(True)  # Automatically calculated
        split_layout.addWidget(QLabel("Test Split:"), 0, 4)
        split_layout.addWidget(self.test_split, 0, 5)

        # Random state for reproducibility
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 999)
        self.random_state.setValue(42)
        split_layout.addWidget(QLabel("Random Seed:"), 1, 0)
        split_layout.addWidget(self.random_state, 1, 1)

        split_group.setLayout(split_layout)
        data_layout.addWidget(split_group)

        # Bottom rows for preprocessing
        # Scaling options
        scaling_group = QGroupBox("Preprocessing")
        scaling_layout = QGridLayout()

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        scaling_layout.addWidget(QLabel("Scaling:"), 0, 0)
        scaling_layout.addWidget(self.scaling_combo, 0, 1)

        # Missing value handling options
        self.missing_value_combo = QComboBox()
        self.missing_value_combo.addItems([
            "No Handling",
            "Mean Imputation",
            "Median Imputation",
            "Forward Fill",
            "Backward Fill",
            "Linear Interpolation"
        ])
        scaling_layout.addWidget(QLabel("Missing Value Handling:"), 1, 0)
        scaling_layout.addWidget(self.missing_value_combo, 1, 1)

        # Apply preprocessing button
        self.apply_preprocessing_btn = QPushButton("Apply Preprocessing")
        self.apply_preprocessing_btn.clicked.connect(self.apply_preprocessing)
        scaling_layout.addWidget(self.apply_preprocessing_btn, 2, 0, 1, 2)

        scaling_group.setLayout(scaling_layout)
        data_layout.addWidget(scaling_group)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def update_splits(self):
        """Update split values to ensure they sum to 1.0"""
        train = self.train_split.value()
        val = self.val_split.value()

        # Ensure minimum values
        if train < 0.1:
            train = 0.1
            self.train_split.setValue(train)
        if val < 0:
            val = 0
            self.val_split.setValue(val)

        # Calculate test split
        test = 1.0 - train - val

        # Adjust if total exceeds 1.0
        if test < 0:
            val = 1.0 - train
            self.val_split.setValue(val)
            test = 0

        self.test_split.setValue(test)

    def apply_preprocessing(self):
        """Apply all selected preprocessing steps"""
        try:
            # Apply scaling if selected
            self.apply_scaling()

            # Apply missing value handling if selected
            self.handle_missing_values()

            self.status_bar.showMessage("Preprocessing applied successfully")
        except Exception as e:
            self.show_error(f"Error in preprocessing: {str(e)}")

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()

            if dataset_name == "Load Custom Dataset":
                return

            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "California Housing Dataset":
                data = datasets.fetch_california_housing()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test,
                                     y_test) = tf.keras.datasets.mnist.load_data()
                # For MNIST, we'll split the training data to create a
                # validation set
                train_size = self.train_split.value() / (self.train_split.value() +
                                                         self.val_split.value())
                X_train, X_val, y_train, y_val = model_selection.train_test_split(
                    X_train, y_train,
                    test_size=1 - train_size,
                    random_state=self.random_state.value()
                )
                self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
                self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return

            # Get split ratios
            train_size = self.train_split.value()
            val_size = self.val_split.value()
            test_size = self.test_split.value()

            # First split into train+val and test
            train_val_size = train_size + val_size
            relative_val_size = val_size / train_val_size  # Relative size for second split

            # First split: separate test set
            X_train_val, self.X_test, y_train_val, self.y_test = model_selection.train_test_split(
                data.data, data.target,
                test_size=test_size,
                random_state=self.random_state.value()
            )

            # Second split: separate train and validation
            self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                X_train_val, y_train_val,
                test_size=relative_val_size,
                random_state=self.random_state.value()
            )

            # Apply preprocessing if selected
            self.apply_preprocessing()

            self.status_bar.showMessage(
                f"Loaded {dataset_name} with splits: {train_size: .1%}/{val_size: .1%}/{test_size: .1%}")

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )

            if file_name:
                # Load data
                data = pd.read_csv(file_name)

                # Ask user to select target column
                target_col = self.select_target_column(data.columns)

                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]

                    # Get split ratios
                    train_size = self.train_split.value()
                    val_size = self.val_split.value()
                    test_size = self.test_split.value()

                    # First split into train+val and test
                    train_val_size = train_size + val_size
                    relative_val_size = val_size / train_val_size

                    # First split: separate test set
                    X_train_val, self.X_test, y_train_val, self.y_test = model_selection.train_test_split(
                        X, y, test_size=test_size, random_state=self.random_state.value())

                    # Second split: separate train and validation
                    self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                        X_train_val, y_train_val, test_size=relative_val_size, random_state=self.random_state.value())

                    # Apply preprocessing if selected
                    self.apply_preprocessing()

                    self.status_bar.showMessage(
                        f"Loaded custom dataset: {file_name} with splits: {
                            train_size: .1%}/{
                            val_size: .1%}/{
                            test_size: .1%}"
                    )

        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)

        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()

        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()

                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)

            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def handle_missing_values(self):
        """Handle missing values in the dataset using selected method"""
        if self.X_train is None or self.X_test is None:
            self.show_error("Please load a dataset first")
            return

        try:
            method = self.missing_value_combo.currentText()

            if method == "No Handling":
                return

            # Convert to pandas DataFrame for easier handling
            X_train_df = pd.DataFrame(self.X_train)
            X_test_df = pd.DataFrame(self.X_test)

            if method == "Mean Imputation":
                X_train_df = X_train_df.fillna(X_train_df.mean())
                # Use training mean for test set
                X_test_df = X_test_df.fillna(X_train_df.mean())

            elif method == "Median Imputation":
                X_train_df = X_train_df.fillna(X_train_df.median())
                # Use training median for test set
                X_test_df = X_test_df.fillna(X_train_df.median())

            elif method == "Forward Fill":
                X_train_df = X_train_df.fillna(method='ffill')
                X_test_df = X_test_df.fillna(method='ffill')

            elif method == "Backward Fill":
                X_train_df = X_train_df.fillna(method='bfill')
                X_test_df = X_test_df.fillna(method='bfill')

            elif method == "Linear Interpolation":
                X_train_df = X_train_df.interpolate(method='linear')
                X_test_df = X_test_df.interpolate(method='linear')

            # Convert back to numpy arrays
            self.X_train = X_train_df.to_numpy()
            self.X_test = X_test_df.to_numpy()

            self.status_bar.showMessage(
                f"Applied {method} for missing value handling")

        except Exception as e:
            self.show_error(f"Error handling missing values: {str(e)}")

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()

        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Cross Validation", self.create_cross_validation_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        regression_layout.addWidget(lr_group)

        # Support Vector Regression
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {"kernel": ["linear", "rbf", "poly"],
             "C": "double",
             "epsilon": "double",
             "degree": "int",
             "gamma": ["scale", "auto", "double"]}
        )
        regression_layout.addWidget(svr_group)

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)

        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double",
             "priors_type": ["uniform", "custom"],
             "class_priors": "text"}  # Will be used when priors_type is "custom"
        )
        classification_layout.addWidget(nb_group)

        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)

        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)

        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)

        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)

        return widget

    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()

        # PCA parameters
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)

        # Manual PCA calculation button
        manual_pca_btn = QPushButton("Manual PCA (2D Example)")
        manual_pca_btn.clicked.connect(self.manual_pca_calculation)
        pca_layout.addWidget(manual_pca_btn)

        # PCA visualization button
        pca_viz_btn = QPushButton("Visualize PCA")
        pca_viz_btn.clicked.connect(self.visualize_pca)
        pca_layout.addWidget(pca_viz_btn)

        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 0)

        # LDA section
        lda_group = QGroupBox("Linear Discriminant Analysis")
        lda_layout = QVBoxLayout()

        # LDA parameters
        lda_params = self.create_algorithm_group(
            "LDA Parameters",
            {"n_components": "int",
             "solver": ["svd", "lsqr", "eigen"]}
        )
        lda_layout.addWidget(lda_params)

        # LDA visualization button
        lda_viz_btn = QPushButton("Visualize LDA")
        lda_viz_btn.clicked.connect(self.visualize_lda)
        lda_layout.addWidget(lda_viz_btn)

        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 0, 1)

        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()

        # K-Means parameters
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)

        # Elbow method button
        elbow_btn = QPushButton("Find Optimal K (Elbow Method)")
        elbow_btn.clicked.connect(self.visualize_elbow_method)
        kmeans_layout.addWidget(elbow_btn)

        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 1, 0)

        # 2D/3D Projection section
        projection_group = QGroupBox("2D/3D Projection")
        projection_layout = QVBoxLayout()

        # Projection method selection
        projection_method = QComboBox()
        projection_method.addItems(["t-SNE", "PCA", "LDA"])
        projection_layout.addWidget(QLabel("Projection Method:"))
        projection_layout.addWidget(projection_method)

        # Dimension selection
        dimension_combo = QComboBox()
        dimension_combo.addItems(["2D", "3D"])
        projection_layout.addWidget(QLabel("Dimension:"))
        projection_layout.addWidget(dimension_combo)

        # t-SNE specific parameters
        tsne_params = QGroupBox("t-SNE Parameters")
        tsne_layout = QVBoxLayout()

        perplexity_spin = QDoubleSpinBox()
        perplexity_spin.setRange(5.0, 50.0)
        perplexity_spin.setValue(30.0)
        perplexity_spin.setSingleStep(1.0)
        tsne_layout.addWidget(QLabel("Perplexity:"))
        tsne_layout.addWidget(perplexity_spin)

        learning_rate_spin = QDoubleSpinBox()
        learning_rate_spin.setRange(10.0, 1000.0)
        learning_rate_spin.setValue(200.0)
        learning_rate_spin.setSingleStep(10.0)
        tsne_layout.addWidget(QLabel("Learning Rate:"))
        tsne_layout.addWidget(learning_rate_spin)

        tsne_params.setLayout(tsne_layout)
        projection_layout.addWidget(tsne_params)

        # Visualization button
        viz_btn = QPushButton("Visualize Projection")
        viz_btn.clicked.connect(lambda: self.visualize_projection(
            projection_method.currentText(),
            dimension_combo.currentText(),
            perplexity_spin.value(),
            learning_rate_spin.value()
        ))
        projection_layout.addWidget(viz_btn)

        projection_group.setLayout(projection_layout)
        layout.addWidget(projection_group, 1, 1)

        # UMAP section
        umap_group = QGroupBox("UMAP")
        umap_layout = QVBoxLayout()

        # UMAP parameters
        self.umap_params = {
            'n_neighbors': QSpinBox(),
            'min_dist': QDoubleSpinBox(),
            'n_components': QSpinBox(),
            'metric': QComboBox()
        }

        # Configure parameter widgets
        self.umap_params['n_neighbors'].setRange(2, 200)
        self.umap_params['n_neighbors'].setValue(15)
        self.umap_params['n_neighbors'].setToolTip(
            "Size of local neighborhood (higher: more global structure)")

        self.umap_params['min_dist'].setRange(0.0, 1.0)
        self.umap_params['min_dist'].setSingleStep(0.1)
        self.umap_params['min_dist'].setValue(0.1)
        self.umap_params['min_dist'].setToolTip(
            "Minimum distance between points (higher: more even spacing)")

        self.umap_params['n_components'].setRange(2, 3)
        self.umap_params['n_components'].setValue(2)
        self.umap_params['n_components'].setToolTip(
            "Number of dimensions in the output")

        self.umap_params['metric'].addItems(
            ['euclidean', 'manhattan', 'cosine', 'correlation'])
        self.umap_params['metric'].setToolTip("Distance metric to use")

        # Add parameters to layout
        params_grid = QGridLayout()
        params_grid.addWidget(QLabel("n_neighbors:"), 0, 0)
        params_grid.addWidget(self.umap_params['n_neighbors'], 0, 1)
        params_grid.addWidget(QLabel("min_dist:"), 1, 0)
        params_grid.addWidget(self.umap_params['min_dist'], 1, 1)
        params_grid.addWidget(QLabel("n_components:"), 2, 0)
        params_grid.addWidget(self.umap_params['n_components'], 2, 1)
        params_grid.addWidget(QLabel("metric:"), 3, 0)
        params_grid.addWidget(self.umap_params['metric'], 3, 1)

        umap_layout.addLayout(params_grid)

        # UMAP visualization button
        umap_viz_btn = QPushButton("Visualize UMAP")
        umap_viz_btn.clicked.connect(self.visualize_umap)
        umap_layout.addWidget(umap_viz_btn)

        # Real-time update checkbox
        self.umap_realtime = QCheckBox("Real-time Updates")
        self.umap_realtime.setToolTip(
            "Update visualization when parameters change (may be slow with large datasets)")
        umap_layout.addWidget(self.umap_realtime)

        # Connect parameter changes to update function
        for param in self.umap_params.values():
            if isinstance(param, QSpinBox):
                param.valueChanged.connect(self.update_umap_if_realtime)
            elif isinstance(param, QDoubleSpinBox):
                param.valueChanged.connect(self.update_umap_if_realtime)
            elif isinstance(param, QComboBox):
                param.currentTextChanged.connect(self.update_umap_if_realtime)

        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group, 2, 0)

        return widget

    def update_umap_if_realtime(self):
        """Update UMAP visualization if real-time updates are enabled"""
        if self.umap_realtime.isChecked():
            self.visualize_umap()

    def visualize_umap(self):
        """Visualize data using UMAP"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Get UMAP parameters
            n_neighbors = self.umap_params['n_neighbors'].value()
            min_dist = self.umap_params['min_dist'].value()
            n_components = self.umap_params['n_components'].value()
            metric = self.umap_params['metric'].currentText()

            # Create and fit UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=42
            )

            # Combine train and validation data if available
            if hasattr(self, 'X_val') and self.X_val is not None:
                X = np.vstack((self.X_train, self.X_val))
                if self.y_train is not None and self.y_val is not None:
                    y = np.concatenate((self.y_train, self.y_val))
                else:
                    y = None
            else:
                X = self.X_train
                y = self.y_train if hasattr(self, 'y_train') else None

            # Fit and transform data
            X_umap = reducer.fit_transform(X)

            # Visualize results
            self.figure.clear()

            if n_components == 2:
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                                     c=y if y is not None else 'blue',
                                     cmap='viridis')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')

            else:  # 3D
                ax = self.figure.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
                                     c=y if y is not None else 'blue',
                                     cmap='viridis')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_zlabel('UMAP 3')

                # Enable 3D rotation
                ax.mouse_init()

            if y is not None:
                self.figure.colorbar(scatter)

            ax.set_title('UMAP Visualization')
            ax.grid(True)

            self.figure.tight_layout()
            self.canvas.draw()

            # Update metrics
            metrics_text = "UMAP Parameters:\n\n"
            metrics_text += f"n_neighbors: {n_neighbors}\n"
            metrics_text += f"min_dist: {min_dist}\n"
            metrics_text += f"n_components: {n_components}\n"
            metrics_text += f"metric: {metric}\n\n"

            if hasattr(reducer, 'embedding_'):
                metrics_text += "Embedding Statistics:\n"
                metrics_text += f"Shape: {reducer.embedding_.shape}\n"
                metrics_text += f"Min: {reducer.embedding_.min(): .4f}\n"
                metrics_text += f"Max: {reducer.embedding_.max(): .4f}\n"
                metrics_text += f"Mean: {reducer.embedding_.mean(): .4f}\n"
                metrics_text += f"Std: {reducer.embedding_.std(): .4f}\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in UMAP visualization: {str(e)}")

    def create_cross_validation_tab(self):
        """Create the cross validation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # K-fold configuration
        fold_group = QGroupBox("K-Fold Configuration")
        fold_layout = QGridLayout()

        # K selection
        self.k_fold_spin = QSpinBox()
        self.k_fold_spin.setRange(2, 20)
        self.k_fold_spin.setValue(5)
        fold_layout.addWidget(QLabel("Number of Folds (k):"), 0, 0)
        fold_layout.addWidget(self.k_fold_spin, 0, 1)

        # Random state for reproducibility
        self.cv_random_state = QSpinBox()
        self.cv_random_state.setRange(0, 999)
        self.cv_random_state.setValue(42)
        fold_layout.addWidget(QLabel("Random Seed:"), 1, 0)
        fold_layout.addWidget(self.cv_random_state, 1, 1)

        # Shuffle option
        self.cv_shuffle = QCheckBox("Shuffle Data")
        self.cv_shuffle.setChecked(True)
        fold_layout.addWidget(self.cv_shuffle, 2, 0, 1, 2)

        fold_group.setLayout(fold_layout)
        layout.addWidget(fold_group)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        self.cv_model_combo = QComboBox()
        self.cv_model_combo.addItems([
            "Linear Regression",
            "Logistic Regression",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors"
        ])
        model_layout.addWidget(self.cv_model_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Run button
        run_btn = QPushButton("Run Cross-Validation")
        run_btn.clicked.connect(self.run_cross_validation)
        layout.addWidget(run_btn)

        return widget

    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()

        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)

        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()

        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)

        return widget

    def create_visualization(self):
        """Create the visualization section with interactive features"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()

        # Create matplotlib figure with navigation toolbar
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        # Add navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create a widget to hold the canvas and toolbar
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        viz_layout.addWidget(plot_widget)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()

        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}: "))

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)

            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)

        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(
            lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)

        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()

        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)

        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)

        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)

        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)

        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()

        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)

        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)

        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()

        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)

        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)

        return widget

    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(
            ["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()

        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}

        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(
                    ["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input

                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input

                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)

            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()

            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(
                            map(int, widget.text().split(',')))

            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })

            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)

        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        # Loss function selection
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Function:"))
        self.loss_combo = QComboBox()
        self.loss_combo.addItems([
            "Categorical Cross-Entropy",
            "Binary Cross-Entropy",
            "Hinge Loss"
        ])
        loss_layout.addWidget(self.loss_combo)
        layout.addLayout(loss_layout)

        group.setLayout(layout)
        return group

    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()

        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()

        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return

        try:
            # Create and compile model
            model = self.create_neural_network()

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            loss_function = self.loss_combo.currentText()

            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test

            # Get number of classes
            num_classes = len(np.unique(self.y_train))

            # Validate loss function selection based on number of classes
            if loss_function == "Binary Cross-Entropy" and num_classes > 2:
                self.show_error(
                    "Binary Cross-Entropy can only be used for binary classification")
                return

            elif loss_function in ["Categorical Cross-Entropy", "Hinge Loss"] and num_classes == 2:
                self.show_error(
                    "Categorical Cross-Entropy and Hinge Loss are better suited for multi-class classification")
                return

            # Prepare target data based on loss function
            if loss_function == "Binary Cross-Entropy":
                y_train = self.y_train.reshape(-1, 1)
                y_test = self.y_test.reshape(-1, 1)
            else:  # Categorical Cross-Entropy or Hinge Loss
                y_train = tf.keras.utils.to_categorical(
                    self.y_train, num_classes=num_classes)
                y_test = tf.keras.utils.to_categorical(
                    self.y_test, num_classes=num_classes)

            # Compile model with selected loss function
            optimizer = optimizers.Adam(learning_rate=learning_rate)

            # Map loss function names to Keras loss functions
            loss_mapping = {
                "Categorical Cross-Entropy": 'categorical_crossentropy',
                "Binary Cross-Entropy": 'binary_crossentropy',
                "Hinge Loss": 'categorical_hinge'
            }

            loss = loss_mapping[loss_function]

            # Add appropriate metrics based on loss function
            metrics = ['accuracy']
            if loss_function == "Binary Cross-Entropy":
                metrics.append('AUC')

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)

            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])

            # Update visualization with training history
            self.plot_training_history(history)

            self.status_bar.showMessage("Neural Network Training Complete")

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()

        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]

            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))

        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model

    def train_model(self, model_name, params):
        """Train the selected model"""
        try:
            # Model Initialization
            if model_name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=params['fit_intercept'].isChecked())
            elif model_name == "Support Vector Regression":
                # Handle gamma parameter
                gamma = params['gamma'].currentText()  # If gamma is selected
                # Check if gamma is scale, auto or a custom value
                if gamma not in ['scale', 'auto']:
                    gamma = params['gamma'].value()  # Custom gamma value
                else:
                    gamma = params['gamma'].value()

                model = SVR(kernel=params['kernel'].currentText(),
                            C=params['C'].value(),
                            epsilon=params['epsilon'].value(),
                            degree=params['degree'].value(),
                            gamma=gamma)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=params['C'].value(),
                    max_iter=params['max_iter'].value(),
                    multi_class=params['multi_class'].currentText())
            elif model_name == "Naive Bayes":
                # Handle priors based on selection
                priors = None
                if params['priors_type'].currentText() == "custom":
                    try:
                        # Parse custom priors from text input
                        priors = [
                            float(x) for x in params['class_priors'].text().split(',')]
                        # Normalize priors to sum to 1
                        priors = np.array(priors) / np.sum(priors)
                    except Exception as e:
                        self.show_error(
                            f"Error parsing class priors: {str(e)}")
                        return

                model = GaussianNB(
                    var_smoothing=params['var_smoothing'].value(),
                    priors=priors)
            elif model_name == "Support Vector Machine":
                model = SVC(C=params['C'].value(),
                            kernel=params['kernel'].currentText(),
                            degree=params['degree'].value())
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=params['max_depth'].value(),
                    min_samples_split=params['min_samples_split'].value(),
                    criterion=params['criterion'].currentText())
            elif model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=params['n_estimators'].value(),
                    max_depth=params['max_depth'].value(),
                    min_samples_split=params['min_samples_split'].value())
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(
                    n_neighbors=params['n_neighbors'].value(),
                    weights=params['weights'].currentText(),
                    metric=params['metric'].currentText())
            elif model_name == "K-Means Parameters":
                model = KMeans(n_clusters=params['n_clusters'].value(),
                               max_iter=params['max_iter'].value(),
                               n_init=params['n_init'].value())
            elif model_name == "PCA Parameters":
                model = PCA(n_components=params['n_components'].value(),
                            whiten=params['whiten'].isChecked())
            elif model_name == "LDA Parameters":
                model = LinearDiscriminantAnalysis(
                    n_components=params['n_components'].value(),
                    solver=params['solver'].currentText())
            else:
                raise ValueError(f"Unknown model: {model_name}")

            self.current_model = model

            # Train model
            if model_name == "PCA Parameters":
                # For PCA, we only need to fit the model
                model.fit(self.X_train)
                # Transform the data
                X_train_transformed = model.transform(self.X_train)
                X_test_transformed = model.transform(self.X_test)
                # Update visualization with transformed data
                self.visualize_pca(params)
                return
            elif model_name == "LDA Parameters":
                # For LDA, we need both X and y for fitting
                model.fit(self.X_train, self.y_train)
                # Transform the data
                X_train_transformed = model.transform(self.X_train)
                X_test_transformed = model.transform(self.X_test)
                # Update visualization
                self.visualize_lda(params)
                return
            else:
                model.fit(self.X_train, self.y_train)
                # Make predictions
                y_pred = model.predict(self.X_test)

                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred)

            self.status_bar.showMessage(f"Trained {model_name}")

        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")

    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar

            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)

        return ProgressCallback(self.progress_bar)

    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()

        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")

        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)

                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                     c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)

            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                     c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)

        self.canvas.draw()

    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"

        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 8:  # Regression
            # Basic regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(self.y_test - y_pred))
            r2 = self.current_model.score(self.X_test, self.y_test)

            # Additional regression metrics
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            residuals = self.y_test - y_pred
            rss = np.sum(residuals ** 2)  # Residual Sum of Squares
            tss = np.sum((self.y_test - np.mean(self.y_test))
                         ** 2)  # Total Sum of Squares
            adj_r2 = 1 - (1 - r2) * (len(self.y_test) - 1) / \
                (len(self.y_test) - self.X_test.shape[1] - 1)  # Adjusted R²

            metrics_text += "Regression Metrics:\n"
            metrics_text += f"Mean Squared Error (MSE): {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            metrics_text += f"Mean Absolute Error (MAE): {mae:.4f}\n"
            metrics_text += f"Mean Absolute Percentage Error (MAPE): {
                mape:.2f}%\n"
            metrics_text += f"R² Score: {r2:.4f}\n"
            metrics_text += f"Adjusted R² Score: {adj_r2:.4f}\n"
            metrics_text += f"Residual Sum of Squares (RSS): {rss:.4f}\n"
            metrics_text += f"Total Sum of Squares (TSS): {tss:.4f}\n"

            # Model coefficients if available
            if hasattr(self.current_model, 'coef_'):
                metrics_text += "\nModel Coefficients:\n"
                for i, coef in enumerate(self.current_model.coef_):
                    metrics_text += f"Feature {i + 1}: {coef:.4f}\n"
                if hasattr(self.current_model, 'intercept_'):
                    metrics_text += f"Intercept: {
                        self.current_model.intercept_:.4f}\n"

        else:  # Classification
            # Basic classification metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            # Calculate per-class metrics
            n_classes = len(np.unique(self.y_test))
            precision = np.zeros(n_classes)
            recall = np.zeros(n_classes)
            f1 = np.zeros(n_classes)

            for i in range(n_classes):
                tp = conf_matrix[i, i]
                fp = np.sum(conf_matrix[:, i]) - tp
                fn = np.sum(conf_matrix[i, :]) - tp

                precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + \
                             recall[i]) if (precision[i] + recall[i]) > 0 else 0

            # Calculate macro and weighted averages
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)

            # Calculate weighted averages
            class_counts = np.sum(conf_matrix, axis=1)
            weighted_precision = np.sum(
                precision * class_counts) / np.sum(class_counts)
            weighted_recall = np.sum(
                recall * class_counts) / np.sum(class_counts)
            weighted_f1 = np.sum(f1 * class_counts) / np.sum(class_counts)

            metrics_text += "Classification Metrics:\n"
            metrics_text += f"Overall Accuracy: {accuracy:.4f}\n\n"

            metrics_text += "Per-Class Metrics:\n"
            for i in range(n_classes):
                metrics_text += f"\nClass {i}:\n"
                metrics_text += f"Precision: {precision[i]:.4f}\n"
                metrics_text += f"Recall: {recall[i]:.4f}\n"
                metrics_text += f"F1-Score: {f1[i]:.4f}\n"

            metrics_text += "\nMacro Averages:\n"
            metrics_text += f"Macro Precision: {macro_precision:.4f}\n"
            metrics_text += f"Macro Recall: {macro_recall:.4f}\n"
            metrics_text += f"Macro F1-Score: {macro_f1:.4f}\n"

            metrics_text += "\nWeighted Averages:\n"
            metrics_text += f"Weighted Precision: {weighted_precision:.4f}\n"
            metrics_text += f"Weighted Recall: {weighted_recall:.4f}\n"
            metrics_text += f"Weighted F1-Score: {weighted_f1:.4f}\n"

            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)

        self.metrics_text.setText(metrics_text)

    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()

        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])

        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])

        self.figure.tight_layout()
        self.canvas.draw()

    def visualize_pca(self, params=None):
        """Visualize PCA results with scree plot and explained variance"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Handle both direct calls and calls with parameters
            if params is None or not isinstance(params, dict):
                # Default values when called directly
                # Use minimum of 10 or number of features
                n_components = min(10, self.X_train.shape[1])
                whiten = False
            else:
                # Get PCA parameters from params dict
                n_components = params['n_components'].value()
                whiten = params['whiten'].isChecked()

            # Fit PCA
            pca = PCA(n_components=n_components, whiten=whiten)
            pca.fit(self.X_train)

            # Create visualization
            self.figure.clear()

            # Scree plot
            ax1 = self.figure.add_subplot(121)
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

            ax1.bar(range(1, len(explained_variance_ratio) + 1),
                    explained_variance_ratio)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('Scree Plot')

            # Cumulative explained variance
            ax2 = self.figure.add_subplot(122)
            ax2.plot(range(1, len(cumulative_variance_ratio) + 1),
                     cumulative_variance_ratio, 'bo-')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title('Cumulative Explained Variance')

            self.figure.tight_layout()
            self.canvas.draw()

            # Update metrics
            metrics_text = "PCA Results:\n\n"
            metrics_text += f"Total Explained Variance: {
                sum(explained_variance_ratio):.4f}\n"
            metrics_text += f"Number of Components: {n_components}\n"
            metrics_text += "\nExplained Variance by Component:\n"
            for i, var in enumerate(explained_variance_ratio):
                metrics_text += f"PC{i + 1}: {var:.4f}\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in PCA visualization: {str(e)}")

    def visualize_lda(self, params=None):
        """Visualize LDA results with class separation"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Handle both direct calls and calls with parameters
            if params is None or not isinstance(params, dict):
                # Default values when called directly
                # LDA components limited by number of classes - 1
                n_components = min(2, len(np.unique(self.y_train)) - 1)
                solver = 'svd'  # Default solver
            else:
                # Get LDA parameters from params dict
                n_components = params['n_components'].value()
                solver = params['solver'].currentText()

            # Fit LDA
            lda = LinearDiscriminantAnalysis(
                n_components=n_components, solver=solver)
            X_lda = lda.fit_transform(self.X_train, self.y_train)

            # Create visualization
            self.figure.clear()

            if n_components == 1:
                # 1D plot
                ax = self.figure.add_subplot(111)
                for i in range(len(np.unique(self.y_train))):
                    ax.hist(X_lda[self.y_train == i],
                            alpha=0.5, label=f'Class {i}')
                ax.set_xlabel('LDA Component')
                ax.set_ylabel('Frequency')
                ax.set_title('LDA Class Separation')
                ax.legend()
            else:
                # 2D plot
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(
                    X_lda[:, 0], X_lda[:, 1], c=self.y_train, cmap='viridis')
                ax.set_xlabel('LDA Component 1')
                ax.set_ylabel('LDA Component 2')
                ax.set_title('LDA Class Separation')
                self.figure.colorbar(scatter)

            self.figure.tight_layout()
            self.canvas.draw()

            # Update metrics
            metrics_text = "LDA Results:\n\n"
            metrics_text += f"Number of Components: {n_components}\n"
            metrics_text += f"Solver: {solver}\n"
            metrics_text += f"Explained Variance Ratio: {
                sum(
                    lda.explained_variance_ratio_):.4f}\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in LDA visualization: {str(e)}")

    def visualize_elbow_method(self):
        """Visualize elbow method for K-Means clustering"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Calculate inertia and clustering metrics for different k values
            k_range = range(1, 11)
            inertias = []
            silhouette_scores = []
            davies_bouldin_scores = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(self.X_train)
                inertias.append(kmeans.inertia_)

                if k > 1:  # Silhouette and Davies-Bouldin scores require at least 2 clusters
                    labels = kmeans.labels_
                    silhouette_scores.append(
                        silhouette_score(self.X_train, labels))
                    davies_bouldin_scores.append(
                        davies_bouldin_score(self.X_train, labels))
                else:
                    silhouette_scores.append(0)  # Placeholder for k=1
                    davies_bouldin_scores.append(0)  # Placeholder for k=1

            # Create visualization with three subplots
            self.figure.clear()

            # Elbow plot
            ax1 = self.figure.add_subplot(131)
            ax1.plot(k_range, inertias, 'bo-')
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Inertia')
            ax1.set_title('Elbow Method')
            ax1.grid(True)

            # Silhouette score plot
            ax2 = self.figure.add_subplot(132)
            ax2.plot(k_range[1:], silhouette_scores[1:], 'go-')  # Skip k=1
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            ax2.grid(True)

            # Davies-Bouldin score plot
            ax3 = self.figure.add_subplot(133)
            ax3.plot(k_range[1:], davies_bouldin_scores[1:], 'ro-')  # Skip k=1
            ax3.set_xlabel('Number of Clusters (k)')
            ax3.set_ylabel('Davies-Bouldin Index')
            ax3.set_title('Davies-Bouldin Analysis')
            ax3.grid(True)

            self.figure.tight_layout()
            self.canvas.draw()

            # Update metrics text with detailed analysis
            metrics_text = "Clustering Evaluation Results:\n\n"

            # Find optimal k values based on different metrics
            optimal_k_elbow = self.find_elbow_point(k_range, inertias)
            optimal_k_silhouette = k_range[np.argmax(
                silhouette_scores) + 1]  # Add 1 because we skip k=1
            # Add 1 because we skip k=1
            optimal_k_davies = k_range[np.argmin(
                davies_bouldin_scores[1:]) + 1]

            metrics_text += "Optimal number of clusters (k):\n"
            metrics_text += f"- Based on Elbow Method: {optimal_k_elbow}\n"
            metrics_text += f"- Based on Silhouette Score: {optimal_k_silhouette}\n"
            metrics_text += f"- Based on Davies-Bouldin Index: {optimal_k_davies}\n\n"

            metrics_text += "Detailed metrics for each k:\n"
            for k in k_range:
                metrics_text += f"\nk = {k}:\n"
                metrics_text += f"Inertia: {inertias[k - 1]:.2f}\n"
                if k > 1:
                    metrics_text += f"Silhouette Score: {silhouette_scores[k - 1]:.4f}\n"
                    metrics_text += f"Davies-Bouldin Index: {davies_bouldin_scores[k - 1]:.4f}\n"

            metrics_text += "\nInterpretation:\n"
            metrics_text += "- Silhouette Score: Higher is better (range: -1 to 1)\n"
            metrics_text += "- Davies-Bouldin Index: Lower is better\n"
            metrics_text += "- Elbow Method: Look for the 'elbow' point where inertia decrease slows down\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in clustering evaluation: {str(e)}")

    def find_elbow_point(self, k_range, inertias):
        """Find the elbow point using the maximum curvature method"""
        # Convert to numpy arrays for easier manipulation
        k_array = np.array(list(k_range))
        inertia_array = np.array(inertias)

        # Normalize the data
        k_norm = (k_array - k_array.min()) / (k_array.max() - k_array.min())
        inertia_norm = (inertia_array - inertia_array.min()) / \
            (inertia_array.max() - inertia_array.min())

        # Calculate the angle between consecutive segments
        angles = []
        for i in range(1, len(k_norm) - 1):
            vector1 = np.array(
                [k_norm[i] - k_norm[i - 1], inertia_norm[i] - inertia_norm[i - 1]])
            vector2 = np.array(
                [k_norm[i + 1] - k_norm[i], inertia_norm[i + 1] - inertia_norm[i]])

            # Normalize vectors
            vector1 = vector1 / np.linalg.norm(vector1)
            vector2 = vector2 / np.linalg.norm(vector2)

            # Calculate angle
            angle = np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))
            angles.append(angle)

        # The elbow point is where the angle is maximum
        # Add 1 because we skip the first point in angle calculation
        elbow_idx = np.argmax(angles) + 1
        return k_range[elbow_idx]

    def visualize_projection(self, method, dimension,
                             perplexity, learning_rate):
        """Visualize 2D/3D projection using selected method with interactive features"""
        if self.X_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Create visualization
            self.figure.clear()

            # Enable tight layout for better use of space
            self.figure.set_tight_layout(True)

            if dimension == "2D":
                ax = self.figure.add_subplot(111)
            else:  # 3D
                ax = self.figure.add_subplot(111, projection='3d')
                # Enable 3D rotation
                ax.mouse_init()

            if method == "t-SNE":
                # Apply t-SNE
                tsne = TSNE(n_components=2 if dimension == "2D" else 3,
                            perplexity=perplexity,
                            learning_rate=learning_rate)
                X_proj = tsne.fit_transform(self.X_train)

                # Plot with interactive features
                if dimension == "2D":
                    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1],
                                         c=self.y_train if self.y_train is not None else 'blue',
                                         cmap='viridis',
                                         picker=True)  # Enable point picking
                else:
                    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
                                         c=self.y_train if self.y_train is not None else 'blue',
                                         cmap='viridis',
                                         picker=True)  # Enable point picking

            elif method == "PCA":
                # Apply PCA
                pca = PCA(n_components=2 if dimension == "2D" else 3)
                X_proj = pca.fit_transform(self.X_train)

                # Plot with interactive features
                if dimension == "2D":
                    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1],
                                         c=self.y_train if self.y_train is not None else 'blue',
                                         cmap='viridis',
                                         picker=True)
                else:
                    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
                                         c=self.y_train if self.y_train is not None else 'blue',
                                         cmap='viridis',
                                         picker=True)

            elif method == "LDA":
                if self.y_train is None:
                    self.show_error("LDA requires labeled data")
                    return

                # Apply LDA
                lda = LinearDiscriminantAnalysis(
                    n_components=2 if dimension == "2D" else 3)
                X_proj = lda.fit_transform(self.X_train, self.y_train)

                # Plot with interactive features
                if dimension == "2D":
                    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1],
                                         c=self.y_train,
                                         cmap='viridis',
                                         picker=True)
                else:
                    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
                                         c=self.y_train,
                                         cmap='viridis',
                                         picker=True)

            # Set labels and title
            if dimension == "2D":
                ax.set_xlabel(f'{method} Component 1')
                ax.set_ylabel(f'{method} Component 2')
            else:
                ax.set_xlabel(f'{method} Component 1')
                ax.set_ylabel(f'{method} Component 2')
                ax.set_zlabel(f'{method} Component 3')
                # Add a grid for better 3D visualization
                ax.grid(True)

            ax.set_title(f'{method} {dimension} Projection')

            if self.y_train is not None:
                self.figure.colorbar(scatter)

            # Add point picking functionality
            def on_pick(event):
                ind = event.ind[0]
                if self.y_train is not None:
                    label = self.y_train[ind]
                    self.status_bar.showMessage(
                        f"Selected point - Class: {label}")
                else:
                    self.status_bar.showMessage(f"Selected point index: {ind}")

            self.canvas.mpl_connect('pick_event', on_pick)

            # Enable automatic view adjustment for 3D plots
            if dimension == "3D":
                ax.view_init(elev=20, azim=45)  # Set initial viewing angle

            self.canvas.draw()

            # Update metrics
            metrics_text = f"{method} {dimension} Projection Results:\n\n"
            if method == "t-SNE":
                metrics_text += f"Perplexity: {perplexity}\n"
                metrics_text += f"Learning Rate: {learning_rate}\n"
            elif method == "PCA":
                metrics_text += f"Explained Variance Ratio: {
                    sum(
                        pca.explained_variance_ratio_):.4f}\n"
            elif method == "LDA":
                metrics_text += f"Explained Variance Ratio: {
                    sum(
                        lda.explained_variance_ratio_):.4f}\n"

            # Add interaction instructions
            if dimension == "3D":
                metrics_text += "\nInteraction Instructions:\n"
                metrics_text += "- Left click and drag to rotate\n"
                metrics_text += "- Right click and drag to zoom\n"
                metrics_text += "- Middle click and drag to pan\n"
            metrics_text += "- Click on points to see their details\n"
            metrics_text += "- Use toolbar for additional controls\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in projection visualization: {str(e)}")

    def manual_pca_calculation(self):
        """Perform manual PCA calculation for the given covariance matrix and visualize results"""
        try:
            # Given covariance matrix
            cov_matrix = np.array([[5, 2],
                                   [2, 3]])

            # Manual eigenvalue calculation
            # |Σ - λI| = 0
            # |5-λ  2  | = 0
            # |2    3-λ|
            # (5-λ)(3-λ) - 4 = 0
            # λ^2 - 8λ + 11 = 0

            # Using quadratic formula: λ = (-b ± √(b^2 - 4ac))/2a
            a = 1
            b = -8
            c = 11

            eigenval1 = (-b + np.sqrt(b**2 - 4 * a * c)) / \
                (2 * a)  # Larger eigenvalue
            eigenval2 = (-b - np.sqrt(b**2 - 4 * a * c)) / \
                (2 * a)  # Smaller eigenvalue

            # For principal eigenvector (v1), solve (Σ - λ1I)v1 = 0
            # [5-λ1  2   ][x] = [0]
            # [2     3-λ1][y]   [0]
            # Use first equation: (5-λ1)x + 2y = 0
            x1 = 2  # Choose arbitrary x1
            y1 = -(5 - eigenval1) * x1 / 2
            principal_eigenvec = np.array([x1, y1])
            principal_eigenvec = principal_eigenvec / \
                np.linalg.norm(principal_eigenvec)  # Normalize

            # For second eigenvector (v2), solve (Σ - λ2I)v2 = 0
            x2 = 2  # Choose arbitrary x2
            y2 = -(5 - eigenval2) * x2 / 2
            second_eigenvec = np.array([x2, y2])
            second_eigenvec = second_eigenvec / \
                np.linalg.norm(second_eigenvec)  # Normalize

            # Generate some 2D data for visualization
            np.random.seed(42)
            mean = [0, 0]
            n_points = 200
            data = np.random.multivariate_normal(mean, cov_matrix, n_points)

            # Project data onto principal eigenvector
            projection = np.dot(data, principal_eigenvec)

            # Visualize original data and projection
            self.figure.clear()

            # Plot original data and eigenvectors
            ax1 = self.figure.add_subplot(121)
            ax1.scatter(data[:, 0], data[:, 1],
                        alpha=0.5, label='Original Data')

            # Plot eigenvectors scaled by their eigenvalues
            scale = 3
            ax1.arrow(0, 0,
                      principal_eigenvec[0] * scale * np.sqrt(eigenval1),
                      principal_eigenvec[1] * scale * np.sqrt(eigenval1),
                      head_width=0.2, head_length=0.3, fc='r', ec='r',
                      label='Principal Eigenvector')
            ax1.arrow(0, 0,
                      second_eigenvec[0] * scale * np.sqrt(eigenval2),
                      second_eigenvec[1] * scale * np.sqrt(eigenval2),
                      head_width=0.2, head_length=0.3, fc='g', ec='g',
                      label='Second Eigenvector')

            # Plot projection lines for a few points
            for i in range(0, n_points, 20):
                point = data[i]
                proj_point = projection[i] * principal_eigenvec
                ax1.plot([point[0], proj_point[0]], [point[1], proj_point[1]],
                         'k--', alpha=0.2)

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Original Data with Eigenvectors')
            ax1.legend()
            ax1.grid(True)
            ax1.axis('equal')

            # Plot 1D projection
            ax2 = self.figure.add_subplot(122)
            ax2.hist(projection, bins=30, orientation='horizontal')
            ax2.set_ylabel('Projection onto Principal Eigenvector')
            ax2.set_xlabel('Frequency')
            ax2.set_title('1D Projection Histogram')
            ax2.grid(True)

            self.figure.tight_layout()
            self.canvas.draw()

            # Update metrics text with calculations
            metrics_text = "Manual PCA Calculations:\n\n"
            metrics_text += "Covariance Matrix:\n"
            metrics_text += f"[{cov_matrix[0, 0]}, {cov_matrix[0, 1]}]\n"
            metrics_text += f"[{cov_matrix[1, 0]}, {cov_matrix[1, 1]}]\n\n"

            metrics_text += "Eigenvalues:\n"
            metrics_text += f"λ1 = {eigenval1:.4f} (Principal)\n"
            metrics_text += f"λ2 = {eigenval2:.4f}\n\n"

            metrics_text += "Eigenvectors:\n"
            metrics_text += f"v1 = [{
                principal_eigenvec[0]:.4f}, {
                principal_eigenvec[1]:.4f}] (Principal)\n"
            metrics_text += f"v2 = [{
                second_eigenvec[0]:.4f}, {
                second_eigenvec[1]:.4f}]\n\n"

            metrics_text += "Verification:\n"
            # Verify Σv = λv for principal eigenvector
            Σv = np.dot(cov_matrix, principal_eigenvec)
            λv = eigenval1 * principal_eigenvec
            metrics_text += "Σv1 = λ1v1 check:\n"
            metrics_text += f"Σv1 = [{Σv[0]:.4f}, {Σv[1]:.4f}]\n"
            metrics_text += f"λ1v1 = [{λv[0]:.4f}, {λv[1]:.4f}]\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in manual PCA calculation: {str(e)}")

    def run_cross_validation(self):
        """Run k-fold cross-validation with selected parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first")
            return

        try:
            # Get parameters
            k = self.k_fold_spin.value()
            random_state = self.cv_random_state.value()
            shuffle = self.cv_shuffle.isChecked()
            model_name = self.cv_model_combo.currentText()

            # Initialize model based on selection
            if model_name == "Linear Regression":
                model = LinearRegression()
                is_regression = True
            elif model_name == "Logistic Regression":
                model = LogisticRegression(random_state=random_state)
                is_regression = False
            elif model_name == "Support Vector Machine":
                model = SVC(random_state=random_state)
                is_regression = False
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(random_state=random_state)
                is_regression = False
            elif model_name == "Random Forest":
                model = RandomForestClassifier(random_state=random_state)
                is_regression = False
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
                is_regression = False
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Combine training and validation sets for cross-validation
            if hasattr(self, 'X_val') and self.X_val is not None:
                X = np.vstack((self.X_train, self.X_val))
                y = np.concatenate((self.y_train, self.y_val))
            else:
                X = self.X_train
                y = self.y_train

            # Create k-fold splitter
            kf = model_selection.KFold(
                n_splits=k, shuffle=shuffle, random_state=random_state)

            # Initialize metrics storage
            accuracies = []
            mses = []
            rmses = []
            fold_sizes = []

            # Perform k-fold cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]

                # Train model
                model.fit(X_fold_train, y_fold_train)

                # Make predictions
                y_pred = model.predict(X_fold_val)

                # Calculate metrics
                if is_regression:
                    mse = mean_squared_error(y_fold_val, y_pred)
                    rmse = np.sqrt(mse)
                    accuracy = model.score(
                        X_fold_val, y_fold_val)  # R² for regression
                else:
                    accuracy = accuracy_score(y_fold_val, y_pred)
                    mse = mean_squared_error(y_fold_val, y_pred)
                    rmse = np.sqrt(mse)

                accuracies.append(accuracy)
                mses.append(mse)
                rmses.append(rmse)
                fold_sizes.append(len(val_idx))

            # Calculate statistics
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            mean_mse = np.mean(mses)
            std_mse = np.std(mses)
            mean_rmse = np.mean(rmses)
            std_rmse = np.std(rmses)

            # Visualize results
            self.figure.clear()

            # Plot metrics across folds
            ax1 = self.figure.add_subplot(121)
            folds = range(1, k + 1)

            if is_regression:
                ax1.plot(folds, accuracies, 'bo-', label='R² Score')
            else:
                ax1.plot(folds, accuracies, 'bo-', label='Accuracy')
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Across Folds')
            ax1.grid(True)
            ax1.legend()

            # Plot error metrics
            ax2 = self.figure.add_subplot(122)
            ax2.plot(folds, mses, 'ro-', label='MSE')
            ax2.plot(folds, rmses, 'go-', label='RMSE')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('Error')
            ax2.set_title('Error Metrics Across Folds')
            ax2.grid(True)
            ax2.legend()

            self.figure.tight_layout()
            self.canvas.draw()

            # Update metrics text
            metrics_text = f"K-Fold Cross-Validation Results (k={k}):\n\n"
            metrics_text += "Fold Sizes:\n"
            metrics_text += f"{fold_sizes}\n\n"

            if is_regression:
                metrics_text += "R² Score:\n"
            else:
                metrics_text += "Accuracy:\n"
            metrics_text += f"Mean: {mean_accuracy:.4f}\n"
            metrics_text += f"Std: {std_accuracy:.4f}\n"
            metrics_text += f"Per Fold: {[f'{acc:.4f}' for acc in accuracies]}\n\n"

            metrics_text += "Mean Squared Error (MSE):\n"
            metrics_text += f"Mean: {mean_mse:.4f}\n"
            metrics_text += f"Std: {std_mse:.4f}\n"
            metrics_text += f"Per Fold: {[f'{mse:.4f}' for mse in mses]}\n\n"

            metrics_text += "Root Mean Squared Error (RMSE):\n"
            metrics_text += f"Mean: {mean_rmse:.4f}\n"
            metrics_text += f"Std: {std_rmse:.4f}\n"
            metrics_text += f"Per Fold: {[f'{rmse:.4f}' for rmse in rmses]}\n"

            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Error in cross-validation: {str(e)}")


def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
