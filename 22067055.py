import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit, QTableWidgetItem, QListWidget,
                             QTableWidget)
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
import json
import datetime


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)

        # Create a scroll area to be the central widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # Important for layout to expand correctly
        self.setCentralWidget(self.scroll_area)

        # Initialize main widget and layout (this will go *inside* the scroll area)
        self.main_widget = QWidget()
        # self.setCentralWidget(self.main_widget) # No longer set main_widget directly
        self.layout = QVBoxLayout(self.main_widget) # layout is for main_widget
        self.scroll_area.setWidget(self.main_widget) # Put main_widget inside scroll_area

        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None

        # Neural network configuration
        self.layer_config = []
        self.base_model = None # For pre-trained models

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

        # Image Augmentation Section (within Preprocessing)
        self.augmentation_group = QGroupBox("Image Augmentation (for image datasets)")
        aug_layout = QGridLayout()

        self.enable_augmentation_checkbox = QCheckBox("Enable Image Augmentation")
        self.enable_augmentation_checkbox.stateChanged.connect(self.toggle_augmentation_controls)
        aug_layout.addWidget(self.enable_augmentation_checkbox, 0, 0, 1, 2)

        # Rotation
        self.aug_rotate_checkbox = QCheckBox("Rotation")
        aug_layout.addWidget(self.aug_rotate_checkbox, 1, 0)
        self.aug_rotate_angle_spinbox = QDoubleSpinBox()
        self.aug_rotate_angle_spinbox.setRange(0, 45.0) # Max rotation angle in degrees
        self.aug_rotate_angle_spinbox.setValue(10.0)
        self.aug_rotate_angle_spinbox.setSuffix(" °")
        aug_layout.addWidget(self.aug_rotate_angle_spinbox, 1, 1)

        # Horizontal Flip
        self.aug_hflip_checkbox = QCheckBox("Horizontal Flip")
        aug_layout.addWidget(self.aug_hflip_checkbox, 2, 0)

        # Vertical Flip
        self.aug_vflip_checkbox = QCheckBox("Vertical Flip")
        aug_layout.addWidget(self.aug_vflip_checkbox, 2, 1)
        
        # Scaling/Zoom
        self.aug_scale_checkbox = QCheckBox("Scaling (Zoom)")
        aug_layout.addWidget(self.aug_scale_checkbox, 3, 0)
        self.aug_scale_factor_spinbox = QDoubleSpinBox()
        self.aug_scale_factor_spinbox.setRange(0.0, 0.5)  # Max zoom factor (e.g., 0.2 for +/- 20%)
        self.aug_scale_factor_spinbox.setValue(0.1)
        self.aug_scale_factor_spinbox.setToolTip("Zoom range, e.g., 0.1 means zoom between 0.9 and 1.1")
        aug_layout.addWidget(self.aug_scale_factor_spinbox, 3, 1)

        self.augmentation_group.setLayout(aug_layout)
        data_layout.addWidget(self.augmentation_group)
        
        # Initially disable augmentation controls
        self.toggle_augmentation_controls(Qt.CheckState.Unchecked.value)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def toggle_augmentation_controls(self, state):
        """Enable or disable augmentation parameter controls based on the main checkbox."""
        enabled = (state == Qt.CheckState.Checked.value)
        self.aug_rotate_checkbox.setEnabled(enabled)
        self.aug_rotate_angle_spinbox.setEnabled(enabled)
        self.aug_hflip_checkbox.setEnabled(enabled)
        self.aug_vflip_checkbox.setEnabled(enabled)
        self.aug_scale_checkbox.setEnabled(enabled)
        self.aug_scale_factor_spinbox.setEnabled(enabled)
        
        # Also consider if an image dataset is loaded
        is_image_data = self.dataset_combo.currentText() == "MNIST Dataset" # Add other image types later
        self.augmentation_group.setEnabled(is_image_data)
        if not is_image_data: # If not image data, all sub-controls should be off
            self.enable_augmentation_checkbox.setChecked(False)
            self.aug_rotate_checkbox.setEnabled(False)
            self.aug_rotate_angle_spinbox.setEnabled(False)
            self.aug_hflip_checkbox.setEnabled(False)
            self.aug_vflip_checkbox.setEnabled(False)
            self.aug_scale_checkbox.setEnabled(False)
            self.aug_scale_factor_spinbox.setEnabled(False)

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

            # Apply image augmentation if enabled and applicable
            if self.enable_augmentation_checkbox.isChecked() and self.dataset_combo.currentText() == "MNIST Dataset":
                self.apply_image_augmentation()

            self.status_bar.showMessage("Preprocessing applied successfully")
        except Exception as e:
            self.show_error(f"Error in preprocessing: {str(e)}")

    def apply_image_augmentation(self):
        """Apply selected image augmentations to self.X_train"""
        if self.X_train is None:
            self.show_error("Cannot apply augmentation: X_train is not loaded.")
            return

        # Ensure X_train is in a format suitable for tf.image (e.g., NHWC for MNIST)
        # MNIST is typically (num_samples, 28, 28). Reshape to (num_samples, 28, 28, 1).
        if len(self.X_train.shape) == 3: # Assuming (N, H, W)
            X_augmented = np.expand_dims(self.X_train, axis=-1).astype(np.float32)
        elif len(self.X_train.shape) == 4 and self.X_train.shape[-1] == 1: # Already (N,H,W,C)
            X_augmented = self.X_train.astype(np.float32)
        else:
            self.show_error(f"Unsupported image data shape for augmentation: {self.X_train.shape}")
            return
        
        # Create a sequence of Keras preprocessing layers
        augmentation_layers = []

        if self.aug_rotate_checkbox.isChecked():
            # Keras RandomRotation: if factor is a float, range is [-factor*2*pi, factor*2*pi].
            # If our spinbox is in degrees, we want rotation in [-angle_degrees, +angle_degrees].
            # So, factor * 2 * pi = angle_degrees * pi / 180 (converting degrees to radians)
            # factor = (angle_degrees / 180) / 2 = angle_degrees / 360
            angle_degrees = self.aug_rotate_angle_spinbox.value()
            rotation_factor = angle_degrees / 360.0
            augmentation_layers.append(layers.RandomRotation(factor=rotation_factor))

        if self.aug_hflip_checkbox.isChecked():
            augmentation_layers.append(layers.RandomFlip(mode="horizontal"))

        if self.aug_vflip_checkbox.isChecked():
            augmentation_layers.append(layers.RandomFlip(mode="vertical"))

        if self.aug_scale_checkbox.isChecked():
            zoom_factor = self.aug_scale_factor_spinbox.value()
            # RandomZoom takes a tuple for (min_zoom, max_zoom) relative to 1.0
            # Or a single float for (-factor, factor) meaning zoom in (factor) or zoom out (-factor)
            # Height_factor = (-zoom_factor, zoom_factor), width_factor = (-zoom_factor, zoom_factor)
            augmentation_layers.append(layers.RandomZoom(height_factor=zoom_factor, width_factor=zoom_factor))

        if not augmentation_layers:
            self.status_bar.showMessage("Image augmentation enabled, but no specific augmentations selected.")
            return

        # Build a sequential model for augmentation
        augmentation_model = tf.keras.Sequential(augmentation_layers)

        # Apply augmentations. 
        # This will augment the entire training dataset once. 
        # For on-the-fly augmentation during training, this should be part of the tf.data.Dataset pipeline.
        # For now, we augment it here as a preprocessing step.
        augmented_images = []
        # Process in batches to avoid memory issues if X_train is very large
        batch_size = 128 
        for i in range(0, X_augmented.shape[0], batch_size):
            batch = X_augmented[i:i+batch_size]
            augmented_batch = augmentation_model(batch, training=True) # training=True to activate layers
            augmented_images.append(augmented_batch.numpy())
        
        self.X_train = np.concatenate(augmented_images, axis=0)
        
        # If original was (N,H,W), squeeze back. Otherwise keep (N,H,W,C)
        if len(self.X_train.shape) == 4 and self.X_train.shape[-1] == 1 and len(self.X_train.shape) == 3:
             self.X_train = np.squeeze(self.X_train, axis=-1)

        self.status_bar.showMessage("Image augmentation applied to training data.")

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
                self.toggle_augmentation_controls(self.enable_augmentation_checkbox.checkState().value) # Update UI
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
            self.toggle_augmentation_controls(self.enable_augmentation_checkbox.checkState().value) # Update UI

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
                    self.toggle_augmentation_controls(self.enable_augmentation_checkbox.checkState().value) # Update UI

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
        
        # Set tab widget properties for larger tabs
        # self.tab_widget.setStyleSheet("""
        #     QTabWidget::pane {
        #         border: 1px solid #cccccc;
        #         background: white;
        #     }
        # """)
        
        # Set minimum size for the tab widget
        self.tab_widget.setMinimumHeight(250)
        #self.tab_widget.setMinimumWidth(800)

        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Cross Validation", self.create_cross_validation_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Training Visualization", self.create_training_visualization_tab)
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_model_design_tab(self):
        """Create the model design tab for configuring neural network architectures"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model Architecture Section
        arch_group = QGroupBox("Model Architecture")
        arch_layout = QVBoxLayout()

        # Layer list with scroll area
        layer_list_group = QGroupBox("Layer Configuration")
        layer_list_layout = QVBoxLayout()
        
        # Create scroll area for layer list
        layer_scroll = QScrollArea()
        layer_scroll.setWidgetResizable(True)
        layer_scroll.setMinimumHeight(200)  # Set minimum height for layer configuration
        self.layer_list_widget = QWidget()
        self.layer_list_layout = QVBoxLayout(self.layer_list_widget)
        layer_scroll.setWidget(self.layer_list_widget)
        layer_list_layout.addWidget(layer_scroll)

        # Layer controls
        layer_controls = QHBoxLayout()
        
        # Add layer button
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer_dialog)
        layer_controls.addWidget(add_layer_btn)
        
        # Remove layer button
        remove_layer_btn = QPushButton("Remove Selected Layer")
        remove_layer_btn.clicked.connect(self.remove_selected_layer)
        layer_controls.addWidget(remove_layer_btn)
        
        # Move layer up/down buttons
        move_up_btn = QPushButton("Move Up")
        move_up_btn.clicked.connect(lambda: self.move_layer(-1))
        layer_controls.addWidget(move_up_btn)
        
        move_down_btn = QPushButton("Move Down")
        move_down_btn.clicked.connect(lambda: self.move_layer(1))
        layer_controls.addWidget(move_down_btn)
        
        layer_list_layout.addLayout(layer_controls)
        layer_list_group.setLayout(layer_list_layout)
        arch_layout.addWidget(layer_list_group)

        # Model summary section
        summary_group = QGroupBox("Model Summary")
        summary_layout = QVBoxLayout()
        self.model_summary_text = QTextEdit()
        self.model_summary_text.setReadOnly(True)
        self.model_summary_text.setMinimumHeight(300)  # Set minimum height for model summary
        summary_layout.addWidget(self.model_summary_text)
        summary_group.setLayout(summary_layout)
        arch_layout.addWidget(summary_group)

        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group)

        # Model save/load section
        save_load_group = QGroupBox("Save/Load Model")
        save_load_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model_architecture)
        save_load_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model_architecture)
        save_load_layout.addWidget(load_btn)
        
        save_load_group.setLayout(save_load_layout)
        layout.addWidget(save_load_group)

        return widget

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

        # Left side: Plotting area
        plot_area_widget = QWidget()
        plot_area_layout = QVBoxLayout(plot_area_widget)

        # Create matplotlib figure with navigation toolbar
        self.figure = Figure(figsize=(8, 6)) # Adjusted figure size for potentially less width
        self.canvas = FigureCanvas(self.figure)

        # Add navigation toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_area_layout.addWidget(self.toolbar)
        plot_area_layout.addWidget(self.canvas)
        
        viz_layout.addWidget(plot_area_widget, 2) # Give more space to plot

        # Right side: Text details (Epoch info and Final Metrics)
        text_details_widget = QWidget()
        text_details_layout = QVBoxLayout(text_details_widget)

        # Epoch details display
        epoch_details_group = QGroupBox("Epoch Details (During Training)")
        epoch_details_layout = QVBoxLayout()
        self.epoch_details_text = QTextEdit()
        self.epoch_details_text.setReadOnly(True)
        self.epoch_details_text.setMinimumHeight(150) # Initial height
        epoch_details_layout.addWidget(self.epoch_details_text)
        epoch_details_group.setLayout(epoch_details_layout)
        text_details_layout.addWidget(epoch_details_group)

        # Final Metrics display
        final_metrics_group = QGroupBox("Final Model Metrics")
        final_metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(150) # Initial height
        final_metrics_layout.addWidget(self.metrics_text)
        final_metrics_group.setLayout(final_metrics_layout)
        text_details_layout.addWidget(final_metrics_group)
        
        viz_layout.addWidget(text_details_widget, 1) # Give less space to text initially

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
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Model architecture section
        model_group = QGroupBox("Model Architecture")
        model_layout = QVBoxLayout()

        # Layer list with scroll area
        layer_list_group = QGroupBox("Layer Configuration")
        layer_list_layout = QVBoxLayout()
        
        # Create scroll area for layer list
        layer_scroll = QScrollArea()
        layer_scroll.setWidgetResizable(True)
        self.layer_list_widget = QWidget()
        self.layer_list_layout = QVBoxLayout(self.layer_list_widget)
        layer_scroll.setWidget(self.layer_list_widget)
        layer_list_layout.addWidget(layer_scroll)

        # Layer controls
        layer_controls = QHBoxLayout()
        
        # Add layer button
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer_dialog)
        layer_controls.addWidget(add_layer_btn)
        
        # Remove layer button
        remove_layer_btn = QPushButton("Remove Selected Layer")
        remove_layer_btn.clicked.connect(self.remove_selected_layer)
        layer_controls.addWidget(remove_layer_btn)
        
        # Move layer up/down buttons
        move_up_btn = QPushButton("Move Up")
        move_up_btn.clicked.connect(lambda: self.move_layer(-1))
        layer_controls.addWidget(move_up_btn)
        
        move_down_btn = QPushButton("Move Down")
        move_down_btn.clicked.connect(lambda: self.move_layer(1))
        layer_controls.addWidget(move_down_btn)
        
        layer_list_layout.addLayout(layer_controls)
        layer_list_group.setLayout(layer_list_layout)
        model_layout.addWidget(layer_list_group)

        # Model summary section
        summary_group = QGroupBox("Model Summary")
        summary_layout = QVBoxLayout()
        self.model_summary_text = QTextEdit()
        self.model_summary_text.setReadOnly(True)
        summary_layout.addWidget(self.model_summary_text)
        summary_group.setLayout(summary_layout)
        model_layout.addWidget(summary_group)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Quick MNIST Configuration
        mnist_group = QGroupBox("Quick MNIST Configuration")
        mnist_layout = QVBoxLayout()
        
        # Add a button to load the compute-efficient MNIST configuration
        load_mnist_config_btn = QPushButton("Load Compute-Efficient MNIST Config")
        load_mnist_config_btn.clicked.connect(self.load_efficient_mnist_config)
        mnist_layout.addWidget(load_mnist_config_btn)
        
        mnist_group.setLayout(mnist_layout)
        layout.addWidget(mnist_group)

        # Training configuration section
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout()

        # Training configuration button
        config_btn = QPushButton("Configure Training")
        config_btn.clicked.connect(self.configure_training)
        training_layout.addWidget(config_btn)

        # Training status
        status_layout = QHBoxLayout()
        self.training_status = QLabel("Not trained")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        status_layout.addWidget(self.training_status)
        status_layout.addWidget(self.training_progress)
        training_layout.addLayout(status_layout)

        # Training buttons
        train_btn_layout = QHBoxLayout()
        train_btn = QPushButton("Train Model")
        stop_btn = QPushButton("Stop Training")
        train_btn.clicked.connect(self.train_neural_network)  # Changed to train_neural_network
        stop_btn.clicked.connect(self.stop_training)
        train_btn_layout.addWidget(train_btn)
        train_btn_layout.addWidget(stop_btn)
        training_layout.addLayout(train_btn_layout)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # Model save/load section
        save_load_group = QGroupBox("Save/Load Model")
        save_load_layout = QHBoxLayout()

        save_btn = QPushButton("Save Model")
        load_btn = QPushButton("Load Model")
        save_btn.clicked.connect(self.save_model_architecture)
        load_btn.clicked.connect(self.load_model_architecture)
        save_load_layout.addWidget(save_btn)
        save_load_layout.addWidget(load_btn)

        save_load_group.setLayout(save_load_layout)
        layout.addWidget(save_load_group)

        # Pre-trained Model Fine-tuning Section
        pretrained_group = QGroupBox("Pre-trained Model Fine-tuning")
        pretrained_layout = QVBoxLayout()

        self.pretrained_model_combo = QComboBox()
        self.pretrained_model_combo.addItems(["None", "VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"])
        pretrained_layout.addWidget(QLabel("Select Pre-trained Model:"))
        pretrained_layout.addWidget(self.pretrained_model_combo)

        self.freeze_layers_checkbox = QCheckBox("Freeze Base Model Layers")
        self.freeze_layers_checkbox.setChecked(True) # Usually good to start with frozen layers
        pretrained_layout.addWidget(self.freeze_layers_checkbox)
        
        # Add a field for number of output classes for the head
        self.num_classes_spinbox = QSpinBox()
        self.num_classes_spinbox.setRange(1, 2000) # Max classes
        self.num_classes_spinbox.setValue(10) # Default for MNIST/CIFAR10
        self.num_classes_spinbox.setToolTip("Number of output classes for the new classification head.")
        num_classes_layout = QHBoxLayout()
        num_classes_layout.addWidget(QLabel("Number of Classes (for head):"))
        num_classes_layout.addWidget(self.num_classes_spinbox)
        pretrained_layout.addLayout(num_classes_layout)

        load_pretrained_btn = QPushButton("Load Model & Build Head")
        load_pretrained_btn.clicked.connect(self.load_and_configure_pretrained_model)
        pretrained_layout.addWidget(load_pretrained_btn)

        pretrained_group.setLayout(pretrained_layout)
        layout.addWidget(pretrained_group)

        return tab

    def load_and_configure_pretrained_model(self):
        """Loads a pre-trained model and sets up a classification head."""
        selected_model_name = self.pretrained_model_combo.currentText()
        if selected_model_name == "None":
            self.base_model = None
            self.layer_config = [] # Clear existing layers if switching back to no pre-trained model
            self.update_layer_list()
            self.status_bar.showMessage("Pre-trained model cleared. Build a custom model or select one.")
            return

        try:
            freeze_base = self.freeze_layers_checkbox.isChecked()
            num_classes = self.num_classes_spinbox.value()
            
            # Define standard input shape for many pre-trained models
            # This might need to be configurable or dynamically determined later
            input_shape = (224, 224, 3) 

            if selected_model_name == "VGG16":
                self.base_model = tf.keras.applications.VGG16(
                    include_top=False, weights='imagenet', input_shape=input_shape
                )
            elif selected_model_name == "ResNet50":
                self.base_model = tf.keras.applications.ResNet50(
                    include_top=False, weights='imagenet', input_shape=input_shape
                )
            elif selected_model_name == "MobileNetV2":
                self.base_model = tf.keras.applications.MobileNetV2(
                    include_top=False, weights='imagenet', input_shape=input_shape
                )
            elif selected_model_name == "EfficientNetB0":
                self.base_model = tf.keras.applications.EfficientNetB0(
                    include_top=False, weights='imagenet', input_shape=input_shape
                )
            else:
                self.show_error(f"Pre-trained model {selected_model_name} is not yet supported.")
                return

            if freeze_base:
                self.base_model.trainable = False
            else:
                self.base_model.trainable = True # Ensure it's trainable if not frozen

            # Define the new classification head layers
            self.layer_config = [
                # Note: Flattening is handled by the base_model if it has a pooling layer at the end,
                # or we can add GlobalAveragePooling2D for flexibility before Dense layers.
                # For now, let's assume base_model output needs flattening or GAP.
                # We will add a GlobalAveragePooling2D as a robust way to connect to Dense head.
                {"type": "GlobalAveragePooling2D", "params": {}},
                {"type": "Dense", "params": {"units": 256, "activation": "relu"}},
                {"type": "Dropout", "params": {"rate": 0.5}},
                {"type": "Dense", "params": {"units": num_classes, "activation": "softmax"}}
            ]

            self.update_layer_list()
            self.status_bar.showMessage(f"Loaded {selected_model_name} with new head. Base trainable: {not freeze_base}")

        except Exception as e:
            self.base_model = None # Reset on error
            self.show_error(f"Error loading pre-trained model {selected_model_name}: {str(e)}")
            self.status_bar.showMessage(f"Failed to load {selected_model_name}.")

    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config and not self.base_model:
            self.show_error("Please add at least one layer to the network or select a pre-trained model.")
            return

        try:
            # Create and compile model
            model = self.create_neural_network()

            # Get training parameters
            if not hasattr(self, 'training_config'):
                self.training_config = {
                    "batch_size": 64,
                    "epochs": 5,
                    "validation_split": 0.2,
                    "early_stopping": {
                        "enable": True,
                        "patience": 3,
                        "min_delta": 0.001
                    }
                }

            # Update UI before starting
            self.training_status.setText("Preparing data and model...")
            self.training_progress.setValue(0)
            QApplication.processEvents() # Ensure UI updates

            # --- Data Preprocessing for Pre-trained Models (if applicable) ---
            X_train_processed = self.X_train
            X_test_processed = self.X_test # Assuming X_val will be handled by validation_split
            
            preprocessor = None
            target_size = None

            if self.base_model: # If a pre-trained model is selected
                selected_model_name = self.pretrained_model_combo.currentText()
                # Get the expected input shape from the base model
                # base_model.input_shape is a tuple like (None, H, W, C)
                # We need H and W, which are typically input_shape[1] and input_shape[2]
                if len(self.base_model.input_shape) == 4:
                    target_size = (self.base_model.input_shape[1], self.base_model.input_shape[2])
                else: # Fallback if shape is unexpected, though Keras models are standard
                    target_size = (224, 224) 
                    self.status_bar.showMessage(f"Warning: Could not determine target size from base model, defaulting to {target_size}")

                if selected_model_name == "VGG16":
                    preprocessor = tf.keras.applications.vgg16.preprocess_input
                elif selected_model_name == "ResNet50":
                    preprocessor = tf.keras.applications.resnet.preprocess_input # or resnet50 for older TF
                elif selected_model_name == "MobileNetV2":
                    preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input
                elif selected_model_name == "EfficientNetB0":
                    preprocessor = tf.keras.applications.efficientnet.preprocess_input
                # Add other models here

                if self.X_train is None:
                    self.show_error("Please load a dataset first for fine-tuning.")
                    return

                # 1. Resize images
                # 2. Ensure 3 channels (e.g., for MNIST)
                # 3. Apply model-specific preprocessing
                def preprocess_dataset(dataset_x):
                    # Ensure float32 for tf.image operations
                    images = tf.image.convert_image_dtype(dataset_x, dtype=tf.float32)
                    
                    # Handle grayscale to RGB conversion if original data is (H, W) or (H, W, 1)
                    if len(images.shape) == 3: # (N, H, W) -> (N, H, W, 1)
                        images = tf.expand_dims(images, axis=-1)
                    
                    if images.shape[-1] == 1: # Grayscale
                        images = tf.image.grayscale_to_rgb(images)
                    
                    # Resize
                    images = tf.image.resize(images, target_size)
                    
                    # Apply model-specific preprocessor if available
                    if preprocessor:
                        images = preprocessor(images)
                    return images.numpy() # Convert back to numpy for model.fit
                
                self.training_status.setText("Preprocessing data for fine-tuning...")
                QApplication.processEvents()

                X_train_processed = preprocess_dataset(self.X_train)
                # Assuming X_test is already split and available. 
                # If using validation_split in model.fit, Keras handles splitting X_train_processed.
                # If self.X_val exists, it should also be preprocessed.
                if self.X_val is not None:
                    X_val_processed = preprocess_dataset(self.X_val)
                else:
                    X_val_processed = None # Will be handled by validation_split from X_train_processed
                
                X_test_processed = preprocess_dataset(self.X_test)

            else: # No base model, use data as is (after regular augmentation/preprocessing)
                X_train_processed = self.X_train
                X_val_processed = self.X_val # Could be None
                X_test_processed = self.X_test
            
            # --- End Data Preprocessing for Pre-trained Models ---

            # Prepare data for neural network (Original part of your function)
            if X_train_processed is None:
                self.show_error("Please load and preprocess a dataset first")
                return

            # Reshape data if needed (e.g. for a custom CNN expecting specific input)
            # This part might need adjustment if base_model is used, as preprocessing above handles shape.
            if not self.base_model and len(X_train_processed.shape) == 3:  # MNIST data for custom CNN
                X_train_final = X_train_processed.reshape(-1, 28, 28, 1)
                X_test_final = X_test_processed.reshape(-1, 28, 28, 1)
                X_val_final = X_val_processed.reshape(-1, 28, 28, 1) if X_val_processed is not None else None
            else:
                X_train_final = X_train_processed
                X_test_final = X_test_processed
                X_val_final = X_val_processed

            # Get number of classes
            # Ensure y_train is not None before trying to get unique values
            if self.y_train is None:
                self.show_error("Target variable y_train is not loaded.")
                return
            num_classes = len(np.unique(self.y_train))

            # Prepare target data
            y_train_cat = tf.keras.utils.to_categorical(self.y_train, num_classes=num_classes)
            y_test_cat = tf.keras.utils.to_categorical(self.y_test, num_classes=num_classes)
            y_val_cat = tf.keras.utils.to_categorical(self.y_val, num_classes=num_classes) if self.y_val is not None else None


            # Create and compile model
            model = self.create_neural_network()
            if model is None: # If create_neural_network failed (e.g. base_model invalid)
                self.training_status.setText("Model creation failed.")
                return

            # Create optimizer if not exists
            if not hasattr(self, 'optimizer'):
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            # Compile model
            model.compile(
                optimizer=self.optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Display model summary in the epoch details text area before training starts
            if hasattr(self, 'epoch_details_text'):
                self.epoch_details_text.clear()
                summary_list = []
                model.summary(print_fn=lambda x: summary_list.append(x))
                self.epoch_details_text.append("Model Summary:\n" + "\n".join(summary_list) + "\n" + "-"*20 + "\n")
                QApplication.processEvents()

            # Create callbacks
            callbacks = []
            if self.training_config['early_stopping']['enable']:
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_config['early_stopping']['patience'],
                    min_delta=self.training_config['early_stopping']['min_delta'],
                    restore_best_weights=True
                ))

            # Add progress callback
            callbacks.append(self.create_progress_callback())

            # Clear previous epoch details (already done above if displaying summary)
            # self.epoch_details_text.clear() 

            self.training_status.setText(f"Training... Epoch 1/{self.training_config['epochs']}")
            QApplication.processEvents()

            # Train model
            fit_params = {
                'x': X_train_final,
                'y': y_train_cat,
                'batch_size': self.training_config['batch_size'],
                'epochs': self.training_config['epochs'],
                'callbacks': callbacks
            }
            if X_val_final is not None and y_val_cat is not None:
                fit_params['validation_data'] = (X_val_final, y_val_cat)
            else:
                fit_params['validation_split'] = self.training_config['validation_split']

            self.training_history = model.fit(**fit_params)

            # Evaluate model
            self.training_status.setText("Evaluating model...")
            QApplication.processEvents()
            test_loss, test_accuracy = model.evaluate(X_test_final, y_test_cat)
            
            # Update visualization with training history
            self.plot_training_history(self.training_history)
            
            # Update metrics
            metrics_text = f"Test Loss: {test_loss:.4f}\n"
            metrics_text += f"Test Accuracy: {test_accuracy:.4f}\n"
            self.metrics_text.setText(metrics_text)

            self.status_bar.showMessage("Neural Network Training Complete")
            self.training_status.setText("Training Complete")
            self.training_progress.setValue(100)

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            self.training_status.setText("Training Failed!")
            self.training_progress.setValue(0)

    def create_neural_network(self):
        """Create neural network based on current configuration"""
        if self.base_model:
            # We have a pre-trained base model. The self.layer_config defines the head.
            # The base_model's input is the model's input.
            # The head layers are added on top of the base_model's output.
            
            # Ensure base_model is a Keras Model instance
            if not isinstance(self.base_model, tf.keras.Model):
                self.show_error("Base model is not a valid Keras model instance.")
                # Potentially clear self.base_model here or handle error more gracefully
                return None 

            # Create a new model starting with the base_model
            # It's crucial that base_model.input and base_model.output are used correctly.
            x = self.base_model.output
            # Add layers from self.layer_config (which is now just the head)
            for layer_config_item in self.layer_config: # Renamed to avoid conflict
                layer_type = layer_config_item["type"]
                params = layer_config_item["params"]

                if layer_type == "Dense":
                    x = layers.Dense(**params)(x)
                elif layer_type == "Conv2D": # Should not typically be in a head after a modern base
                    x = layers.Conv2D(**params)(x)
                elif layer_type == "MaxPooling2D": # Should not typically be in a head
                    x = layers.MaxPooling2D(**params)(x)
                elif layer_type == "Flatten":
                    x = layers.Flatten()(x)
                elif layer_type == "Dropout":
                    x = layers.Dropout(**params)(x)
                elif layer_type == "GlobalAveragePooling2D":
                    x = layers.GlobalAveragePooling2D(**params)(x)
                # Add other relevant head layers here if needed
            
            # Create the final model with the base_model's input and the new head's output
            model = models.Model(inputs=self.base_model.input, outputs=x)
        
        else:
            # No pre-trained model, build from scratch using self.layer_config
            model = models.Sequential()
            for layer_config_item in self.layer_config: # Renamed to avoid conflict
                layer_type = layer_config_item["type"]
                params = layer_config_item["params"]

                if layer_type == "Dense":
                    model.add(layers.Dense(**params))
                elif layer_type == "Conv2D":
                    model.add(layers.Conv2D(**params))
                elif layer_type == "MaxPooling2D":
                    model.add(layers.MaxPooling2D(**params))
                elif layer_type == "Flatten":
                    model.add(layers.Flatten())
                elif layer_type == "Dropout":
                    model.add(layers.Dropout(**params))
                elif layer_type == "GlobalAveragePooling2D":
                    model.add(layers.GlobalAveragePooling2D(**params))
                # Ensure all custom layers are handled here
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
            def __init__(self, progress_bar, epoch_details_text_area, training_status_label, training_progress_bar):
                super().__init__()
                self.progress_bar = progress_bar # Main status bar progress
                self.epoch_details_text_area = epoch_details_text_area
                self.training_status_label = training_status_label
                self.training_progress_bar = training_progress_bar # Tab-specific progress bar

            def on_epoch_begin(self, epoch, logs=None):
                # Update status at the beginning of each epoch
                self.training_status_label.setText(f"Training... Epoch {epoch + 1}/{self.params['epochs']}")
                QApplication.processEvents()

            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress) # Update main status bar progress
                self.training_progress_bar.setValue(progress) # Update tab-specific progress bar
                
                # Append epoch details to the text area
                epoch_info = f"Epoch {epoch + 1}/{self.params['epochs']}:\n"
                if logs:
                    for metric, value in logs.items():
                        epoch_info += f"  {metric}: {value:.4f}\n"
                epoch_info += "-"*20 + "\n"
                self.epoch_details_text_area.append(epoch_info)
                QApplication.processEvents() # Ensure UI updates

        return ProgressCallback(self.progress_bar, self.epoch_details_text, self.training_status, self.training_progress)

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
                metrics_text += f"  Precision: {precision[i]:.4f}\n"
                metrics_text += f"  Recall: {recall[i]:.4f}\n"
                metrics_text += f"  F1-Score: {f1[i]:.4f}\n"

            metrics_text += "\nMacro Averages:\n"
            metrics_text += f"  Macro Precision: {macro_precision:.4f}\n"
            metrics_text += f"  Macro Recall: {macro_recall:.4f}\n"
            metrics_text += f"  Macro F1-Score: {macro_f1:.4f}\n"

            metrics_text += "\nWeighted Averages:\n"
            metrics_text += f"  Weighted Precision: {weighted_precision:.4f}\n"
            metrics_text += f"  Weighted Recall: {weighted_recall:.4f}\n"
            metrics_text += f"  Weighted F1-Score: {weighted_f1:.4f}\n\n"

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

    def remove_selected_layer(self):
        """Remove the selected layer from the model architecture"""
        # Get selected layer index from the layer list
        selected_items = self.layer_list_widget.findChildren(QWidget, "layer_item")
        for i, item in enumerate(selected_items):
            if item.property("selected"):
                if 0 <= i < len(self.layer_config):
                    self.layer_config.pop(i)
                    self.update_layer_list()
                break

    def move_layer(self, direction):
        """Move a layer up or down in the architecture"""
        selected_items = self.layer_list_widget.findChildren(QWidget, "layer_item")
        for i, item in enumerate(selected_items):
            if item.property("selected"):
                new_index = i + direction
                if 0 <= new_index < len(self.layer_config):
                    # Swap layer configurations
                    self.layer_config[i], self.layer_config[new_index] = \
                        self.layer_config[new_index], self.layer_config[i]
                    self.update_layer_list()
                break

    def update_layer_list(self):
        """Update the layer list display"""
        # Clear existing layer items
        while self.layer_list_layout.count():
            item = self.layer_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add current layers to the list
        for i, layer in enumerate(self.layer_config):
            layer_item = QWidget()
            layer_item.setObjectName("layer_item")
            layer_layout = QHBoxLayout(layer_item)

            # Layer type and parameters
            layer_info = f"{i+1}. {layer['type']}"
            if 'params' in layer:
                params_str = ", ".join(f"{k}={v}" for k, v in layer['params'].items())
                layer_info += f" ({params_str})"
            
            layer_label = QLabel(layer_info)
            layer_layout.addWidget(layer_label)

            # Selection checkbox
            select_checkbox = QCheckBox()
            select_checkbox.stateChanged.connect(
                lambda state, item=layer_item: self.toggle_layer_selection(item, state))
            layer_layout.addWidget(select_checkbox)

            self.layer_list_layout.addWidget(layer_item)
        
        # Update the model summary
        self.update_model_summary()

    def toggle_layer_selection(self, layer_item, state):
        """Toggle selection state of a layer item"""
        # Deselect all other layers
        for item in self.layer_list_widget.findChildren(QWidget, "layer_item"):
            if item != layer_item:
                item.setProperty("selected", False)
                item.style().unpolish(item)
                item.style().polish(item)

        # Set selection state for this layer
        layer_item.setProperty("selected", state == Qt.CheckState.Checked.value)
        layer_item.style().unpolish(layer_item)
        layer_item.style().polish(layer_item)

    def save_model_architecture(self):
        """Save the current model architecture to a file"""
        if not self.layer_config:
            self.show_error("No model architecture to save")
            return

        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model Architecture",
                "",
                "JSON files (*.json);;HDF5 files (*.h5)"
            )

            if not file_path:
                return

            if file_path.endswith('.json'):
                # Save as JSON
                model_json = {
                    'layers': self.layer_config,
                    'input_shape': self.X_train.shape[1:] if self.X_train is not None else None
                }
                with open(file_path, 'w') as f:
                    json.dump(model_json, f, indent=2)
            else:
                # Save as HDF5
                model = self.create_neural_network()
                model.save(file_path)

            self.status_bar.showMessage(f"Model architecture saved to {file_path}")

        except Exception as e:
            self.show_error(f"Error saving model architecture: {str(e)}")

    def load_model_architecture(self):
        """Load a model architecture from a file"""
        try:
            # Get load file path
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model Architecture",
                "",
                "JSON files (*.json);;HDF5 files (*.h5)"
            )

            if not file_path:
                return

            if file_path.endswith('.json'):
                # Load from JSON
                with open(file_path, 'r') as f:
                    model_json = json.load(f)
                self.layer_config = model_json['layers']
            else:
                # Load from HDF5
                model = tf.keras.models.load_model(file_path)
                self.layer_config = []
                
                # Convert model layers to configuration
                for layer in model.layers:
                    if isinstance(layer, (layers.Dense, layers.Conv2D, layers.MaxPooling2D,
                                        layers.Flatten, layers.Dropout, layers.LSTM, layers.GRU)):
                        layer_config = {
                            'type': layer.__class__.__name__,
                            'params': self._get_layer_params(layer)
                        }
                        self.layer_config.append(layer_config)

            self.update_layer_list()
            self.status_bar.showMessage(f"Model architecture loaded from {file_path}")

        except Exception as e:
            self.show_error(f"Error loading model architecture: {str(e)}")

    def _get_layer_params(self, layer):
        """Extract parameters from a Keras layer"""
        params = {}
        
        if isinstance(layer, layers.Dense):
            params['units'] = layer.units
            params['activation'] = layer.activation.__name__
        elif isinstance(layer, layers.Conv2D):
            params['filters'] = layer.filters
            params['kernel_size'] = layer.kernel_size
            params['activation'] = layer.activation.__name__
        elif isinstance(layer, layers.MaxPooling2D):
            params['pool_size'] = layer.pool_size
        elif isinstance(layer, layers.Dropout):
            params['rate'] = layer.rate
        elif isinstance(layer, (layers.LSTM, layers.GRU)):
            params['units'] = layer.units
            params['return_sequences'] = layer.return_sequences
            
        return params

    def load_pretrained_model(self):
        """Load and configure a pretrained model"""
        try:
            model_name = self.pretrained_model_combo.currentText()
            
            # Load pretrained model
            if model_name == "VGG16":
                base_model = tf.keras.applications.VGG16(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3)
                )
            elif model_name == "ResNet50":
                base_model = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3)
                )
            elif model_name == "MobileNetV2":
                base_model = tf.keras.applications.MobileNetV2(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3)
                )
            
            # Freeze base layers if requested
            if self.freeze_layers.isChecked():
                base_model.trainable = False
            
            # Clear existing layer configuration
            self.layer_config = []
            
            # Add base model layers
            for layer in base_model.layers:
                layer_config = {
                    'type': layer.__class__.__name__,
                    'params': self._get_layer_params(layer)
                }
                self.layer_config.append(layer_config)
            
            # Add classification head
            self.layer_config.extend([
                {
                    'type': 'Flatten',
                    'params': {}
                },
                {
                    'type': 'Dense',
                    'params': {
                        'units': 512,
                        'activation': 'relu'
                    }
                },
                {
                    'type': 'Dropout',
                    'params': {
                        'rate': 0.5
                    }
                },
                {
                    'type': 'Dense',
                    'params': {
                        'units': len(np.unique(self.y_train)) if self.y_train is not None else 1000,
                        'activation': 'softmax'
                    }
                }
            ])
            
            self.update_layer_list()
            self.status_bar.showMessage(f"Loaded {model_name} pretrained model")
            
        except Exception as e:
            self.show_error(f"Error loading pretrained model: {str(e)}")

    def configure_training(self):
        """Open a dialog to configure training parameters"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Training")
        layout = QVBoxLayout(dialog)

        # Optimizer section
        optimizer_group = QGroupBox("Optimizer")
        optimizer_layout = QVBoxLayout()

        # Optimizer type
        opt_type_layout = QHBoxLayout()
        opt_type_label = QLabel("Optimizer:")
        opt_type_combo = QComboBox()
        opt_type_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        # Set default optimizer if it exists
        if hasattr(self, 'optimizer'):
            if isinstance(self.optimizer, tf.keras.optimizers.Adam):
                opt_type_combo.setCurrentText("Adam")
            elif isinstance(self.optimizer, tf.keras.optimizers.SGD):
                opt_type_combo.setCurrentText("SGD")
            elif isinstance(self.optimizer, tf.keras.optimizers.RMSprop):
                opt_type_combo.setCurrentText("RMSprop")
            elif isinstance(self.optimizer, tf.keras.optimizers.AdamW):
                opt_type_combo.setCurrentText("AdamW")
        opt_type_layout.addWidget(opt_type_label)
        opt_type_layout.addWidget(opt_type_combo)
        optimizer_layout.addLayout(opt_type_layout)

        # Learning rate
        lr_layout = QHBoxLayout()
        lr_label = QLabel("Learning Rate:")
        lr_input = QDoubleSpinBox()
        lr_input.setRange(0.00001, 1.0)
        lr_input.setValue(0.001)
        lr_input.setSingleStep(0.0001)
        lr_input.setDecimals(5)
        # Set current learning rate if optimizer exists
        if hasattr(self, 'optimizer'):
            lr_input.setValue(float(self.optimizer.learning_rate.numpy()))
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(lr_input)
        optimizer_layout.addLayout(lr_layout)

        # Optimizer-specific parameters
        opt_params_group = QGroupBox("Optimizer Parameters")
        opt_params_layout = QVBoxLayout()
        self.opt_param_inputs = {}

        def update_opt_params():
            # Clear existing parameter inputs
            for widget in list(self.opt_param_inputs.values()):
                opt_params_layout.removeWidget(widget)
                widget.deleteLater()
            self.opt_param_inputs.clear()

            opt_type = opt_type_combo.currentText()
            
            if opt_type == "Adam":
                # Beta1
                beta1_label = QLabel("Beta1:")
                beta1_input = QDoubleSpinBox()
                beta1_input.setRange(0.0, 1.0)
                beta1_input.setValue(0.9)
                beta1_input.setSingleStep(0.01)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.Adam):
                    beta1 = self.optimizer.beta_1
                    beta1_input.setValue(float(beta1.numpy()) if hasattr(beta1, 'numpy') else float(beta1))
                self.opt_param_inputs["beta_1"] = beta1_input
                opt_params_layout.addWidget(beta1_label)
                opt_params_layout.addWidget(beta1_input)

                # Beta2
                beta2_label = QLabel("Beta2:")
                beta2_input = QDoubleSpinBox()
                beta2_input.setRange(0.0, 1.0)
                beta2_input.setValue(0.999)
                beta2_input.setSingleStep(0.001)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.Adam):
                    beta2 = self.optimizer.beta_2
                    beta2_input.setValue(float(beta2.numpy()) if hasattr(beta2, 'numpy') else float(beta2))
                self.opt_param_inputs["beta_2"] = beta2_input
                opt_params_layout.addWidget(beta2_label)
                opt_params_layout.addWidget(beta2_input)

                # Epsilon
                epsilon_label = QLabel("Epsilon:")
                epsilon_input = QDoubleSpinBox()
                epsilon_input.setRange(1e-10, 1e-6)
                epsilon_input.setValue(1e-7)
                epsilon_input.setSingleStep(1e-8)
                epsilon_input.setDecimals(10)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.Adam):
                    epsilon = self.optimizer.epsilon
                    epsilon_input.setValue(float(epsilon.numpy()) if hasattr(epsilon, 'numpy') else float(epsilon))
                self.opt_param_inputs["epsilon"] = epsilon_input
                opt_params_layout.addWidget(epsilon_label)
                opt_params_layout.addWidget(epsilon_input)

            elif opt_type == "SGD":
                # Momentum
                momentum_label = QLabel("Momentum:")
                momentum_input = QDoubleSpinBox()
                momentum_input.setRange(0.0, 1.0)
                momentum_input.setValue(0.9)
                momentum_input.setSingleStep(0.1)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.SGD):
                    momentum = self.optimizer.momentum
                    momentum_input.setValue(float(momentum.numpy()) if hasattr(momentum, 'numpy') else float(momentum))
                self.opt_param_inputs["momentum"] = momentum_input
                opt_params_layout.addWidget(momentum_label)
                opt_params_layout.addWidget(momentum_input)

                # Nesterov
                nesterov_label = QLabel("Nesterov Momentum:")
                nesterov_checkbox = QCheckBox()
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.SGD):
                    nesterov_checkbox.setChecked(self.optimizer.nesterov)
                self.opt_param_inputs["nesterov"] = nesterov_checkbox
                opt_params_layout.addWidget(nesterov_label)
                opt_params_layout.addWidget(nesterov_checkbox)

            elif opt_type == "RMSprop":
                # Rho
                rho_label = QLabel("Rho:")
                rho_input = QDoubleSpinBox()
                rho_input.setRange(0.0, 1.0)
                rho_input.setValue(0.9)
                rho_input.setSingleStep(0.01)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.RMSprop):
                    rho = self.optimizer.rho
                    rho_input.setValue(float(rho.numpy()) if hasattr(rho, 'numpy') else float(rho))
                self.opt_param_inputs["rho"] = rho_input
                opt_params_layout.addWidget(rho_label)
                opt_params_layout.addWidget(rho_input)

                # Epsilon
                epsilon_label = QLabel("Epsilon:")
                epsilon_input = QDoubleSpinBox()
                epsilon_input.setRange(1e-10, 1e-6)
                epsilon_input.setValue(1e-7)
                epsilon_input.setSingleStep(1e-8)
                epsilon_input.setDecimals(10)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.RMSprop):
                    epsilon = self.optimizer.epsilon
                    epsilon_input.setValue(float(epsilon.numpy()) if hasattr(epsilon, 'numpy') else float(epsilon))
                self.opt_param_inputs["epsilon"] = epsilon_input
                opt_params_layout.addWidget(epsilon_label)
                opt_params_layout.addWidget(epsilon_input)

            elif opt_type == "AdamW":
                # Weight decay
                wd_label = QLabel("Weight Decay:")
                wd_input = QDoubleSpinBox()
                wd_input.setRange(0.0, 0.1)
                wd_input.setValue(0.01)
                wd_input.setSingleStep(0.001)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.AdamW):
                    wd = self.optimizer.weight_decay
                    wd_input.setValue(float(wd.numpy()) if hasattr(wd, 'numpy') else float(wd))
                self.opt_param_inputs["weight_decay"] = wd_input
                opt_params_layout.addWidget(wd_label)
                opt_params_layout.addWidget(wd_input)

                # Beta1
                beta1_label = QLabel("Beta1:")
                beta1_input = QDoubleSpinBox()
                beta1_input.setRange(0.0, 1.0)
                beta1_input.setValue(0.9)
                beta1_input.setSingleStep(0.01)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.AdamW):
                    beta1 = self.optimizer.beta_1
                    beta1_input.setValue(float(beta1.numpy()) if hasattr(beta1, 'numpy') else float(beta1))
                self.opt_param_inputs["beta_1"] = beta1_input
                opt_params_layout.addWidget(beta1_label)
                opt_params_layout.addWidget(beta1_input)

                # Beta2
                beta2_label = QLabel("Beta2:")
                beta2_input = QDoubleSpinBox()
                beta2_input.setRange(0.0, 1.0)
                beta2_input.setValue(0.999)
                beta2_input.setSingleStep(0.001)
                if hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.AdamW):
                    beta2 = self.optimizer.beta_2
                    beta2_input.setValue(float(beta2.numpy()) if hasattr(beta2, 'numpy') else float(beta2))
                self.opt_param_inputs["beta_2"] = beta2_input
                opt_params_layout.addWidget(beta2_label)
                opt_params_layout.addWidget(beta2_input)

        opt_type_combo.currentIndexChanged.connect(update_opt_params)
        update_opt_params()

        opt_params_group.setLayout(opt_params_layout)
        optimizer_layout.addWidget(opt_params_group)
        optimizer_group.setLayout(optimizer_layout)
        layout.addWidget(optimizer_group)

        # Training configuration section
        training_group = QGroupBox("Training Configuration")
        training_layout = QVBoxLayout()

        # Batch size
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        batch_input = QSpinBox()
        batch_input.setRange(1, 1024)
        batch_input.setValue(32)
        batch_input.setSingleStep(8)
        if hasattr(self, 'training_config'):
            batch_input.setValue(self.training_config['batch_size'])
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(batch_input)
        training_layout.addLayout(batch_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("Epochs:")
        epochs_input = QSpinBox()
        epochs_input.setRange(1, 1000)
        epochs_input.setValue(100)
        if hasattr(self, 'training_config'):
            epochs_input.setValue(self.training_config['epochs'])
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(epochs_input)
        training_layout.addLayout(epochs_layout)

        # Validation split
        val_split_layout = QHBoxLayout()
        val_split_label = QLabel("Validation Split:")
        val_split_input = QDoubleSpinBox()
        val_split_input.setRange(0.0, 0.5)
        val_split_input.setValue(0.2)
        val_split_input.setSingleStep(0.05)
        if hasattr(self, 'training_config'):
            val_split_input.setValue(self.training_config['validation_split'])
        val_split_layout.addWidget(val_split_label)
        val_split_layout.addWidget(val_split_input)
        training_layout.addLayout(val_split_layout)

        # Early stopping
        early_stop_group = QGroupBox("Early Stopping")
        early_stop_layout = QVBoxLayout()

        # Enable early stopping
        enable_es_label = QLabel("Enable Early Stopping:")
        enable_es_checkbox = QCheckBox()
        if hasattr(self, 'training_config'):
            enable_es_checkbox.setChecked(self.training_config['early_stopping']['enable'])
        early_stop_layout.addWidget(enable_es_label)
        early_stop_layout.addWidget(enable_es_checkbox)

        # Patience
        patience_layout = QHBoxLayout()
        patience_label = QLabel("Patience:")
        patience_input = QSpinBox()
        patience_input.setRange(1, 100)
        patience_input.setValue(10)
        if hasattr(self, 'training_config'):
            patience_input.setValue(self.training_config['early_stopping']['patience'])
        patience_layout.addWidget(patience_label)
        patience_layout.addWidget(patience_input)
        early_stop_layout.addLayout(patience_layout)

        # Min delta
        min_delta_layout = QHBoxLayout()
        min_delta_label = QLabel("Min Delta:")
        min_delta_input = QDoubleSpinBox()
        min_delta_input.setRange(0.0, 0.1)
        min_delta_input.setValue(0.001)
        min_delta_input.setSingleStep(0.001)
        if hasattr(self, 'training_config'):
            min_delta_input.setValue(self.training_config['early_stopping']['min_delta'])
        min_delta_layout.addWidget(min_delta_label)
        min_delta_layout.addWidget(min_delta_input)
        early_stop_layout.addLayout(min_delta_layout)

        early_stop_group.setLayout(early_stop_layout)
        training_layout.addWidget(early_stop_group)

        # Learning rate schedule
        lr_schedule_group = QGroupBox("Learning Rate Schedule")
        lr_schedule_layout = QVBoxLayout()

        # Schedule type
        schedule_type_layout = QHBoxLayout()
        schedule_type_label = QLabel("Schedule Type:")
        schedule_type_combo = QComboBox()
        schedule_type_combo.addItems(["None", "Exponential Decay", "Cosine Decay", "Step Decay"])
        if hasattr(self, 'lr_schedule'):
            if isinstance(self.lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay):
                schedule_type_combo.setCurrentText("Exponential Decay")
            elif isinstance(self.lr_schedule, tf.keras.optimizers.schedules.CosineDecay):
                schedule_type_combo.setCurrentText("Cosine Decay")
            elif isinstance(self.lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay) and getattr(self.lr_schedule, 'staircase', False):
                schedule_type_combo.setCurrentText("Step Decay")
        schedule_type_layout.addWidget(schedule_type_label)
        schedule_type_layout.addWidget(schedule_type_combo)
        lr_schedule_layout.addLayout(schedule_type_layout)

        # Schedule parameters
        schedule_params_group = QGroupBox("Schedule Parameters")
        schedule_params_layout = QVBoxLayout()
        self.schedule_param_inputs = {}

        def update_schedule_params():
            # Clear existing parameter inputs
            for widget in list(self.schedule_param_inputs.values()):
                schedule_params_layout.removeWidget(widget)
                widget.deleteLater()
            self.schedule_param_inputs.clear()

            schedule_type = schedule_type_combo.currentText()
            
            if schedule_type == "Exponential Decay":
                # Decay rate
                decay_rate_label = QLabel("Decay Rate:")
                decay_rate_input = QDoubleSpinBox()
                decay_rate_input.setRange(0.0, 1.0)
                decay_rate_input.setValue(0.9)
                decay_rate_input.setSingleStep(0.01)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay):
                    decay_rate_input.setValue(self.lr_schedule.decay_rate.numpy())
                self.schedule_param_inputs["decay_rate"] = decay_rate_input
                schedule_params_layout.addWidget(decay_rate_label)
                schedule_params_layout.addWidget(decay_rate_input)

                # Decay steps
                decay_steps_label = QLabel("Decay Steps:")
                decay_steps_input = QSpinBox()
                decay_steps_input.setRange(1, 10000)
                decay_steps_input.setValue(1000)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay):
                    decay_steps_input.setValue(self.lr_schedule.decay_steps.numpy())
                self.schedule_param_inputs["decay_steps"] = decay_steps_input
                schedule_params_layout.addWidget(decay_steps_label)
                schedule_params_layout.addWidget(decay_steps_input)

            elif schedule_type == "Cosine Decay":
                # Initial learning rate
                initial_lr_label = QLabel("Initial Learning Rate:")
                initial_lr_input = QDoubleSpinBox()
                initial_lr_input.setRange(0.00001, 1.0)
                initial_lr_input.setValue(0.001)
                initial_lr_input.setSingleStep(0.0001)
                initial_lr_input.setDecimals(5)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.CosineDecay):
                    initial_lr_input.setValue(self.lr_schedule.initial_learning_rate.numpy())
                self.schedule_param_inputs["initial_learning_rate"] = initial_lr_input
                schedule_params_layout.addWidget(initial_lr_label)
                schedule_params_layout.addWidget(initial_lr_input)

                # Decay steps
                decay_steps_label = QLabel("Decay Steps:")
                decay_steps_input = QSpinBox()
                decay_steps_input.setRange(1, 10000)
                decay_steps_input.setValue(1000)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.CosineDecay):
                    decay_steps_input.setValue(self.lr_schedule.decay_steps.numpy())
                self.schedule_param_inputs["decay_steps"] = decay_steps_input
                schedule_params_layout.addWidget(decay_steps_label)
                schedule_params_layout.addWidget(decay_steps_input)

                # Alpha
                alpha_label = QLabel("Alpha (min lr factor):")
                alpha_input = QDoubleSpinBox()
                alpha_input.setRange(0.0, 1.0)
                alpha_input.setValue(0.0001)
                alpha_input.setSingleStep(0.0001)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.CosineDecay):
                    alpha_input.setValue(self.lr_schedule.alpha.numpy())
                self.schedule_param_inputs["alpha"] = alpha_input
                schedule_params_layout.addWidget(alpha_label)
                schedule_params_layout.addWidget(alpha_input)

            elif schedule_type == "Step Decay":
                # Decay rate
                decay_rate_label = QLabel("Decay Rate:")
                decay_rate_input = QDoubleSpinBox()
                decay_rate_input.setRange(0.0, 1.0)
                decay_rate_input.setValue(0.1)
                decay_rate_input.setSingleStep(0.1)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay):
                    decay_rate_input.setValue(self.lr_schedule.decay_rate.numpy())
                self.schedule_param_inputs["decay_rate"] = decay_rate_input
                schedule_params_layout.addWidget(decay_rate_label)
                schedule_params_layout.addWidget(decay_rate_input)

                # Decay steps
                decay_steps_label = QLabel("Decay Steps:")
                decay_steps_input = QSpinBox()
                decay_steps_input.setRange(1, 1000)
                decay_steps_input.setValue(10)
                if hasattr(self, 'lr_schedule') and isinstance(self.lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay):
                    decay_steps_input.setValue(self.lr_schedule.decay_steps.numpy())
                self.schedule_param_inputs["decay_steps"] = decay_steps_input
                schedule_params_layout.addWidget(decay_steps_label)
                schedule_params_layout.addWidget(decay_steps_input)

        schedule_type_combo.currentIndexChanged.connect(update_schedule_params)
        update_schedule_params()

        schedule_params_group.setLayout(schedule_params_layout)
        lr_schedule_layout.addWidget(schedule_params_group)
        lr_schedule_group.setLayout(lr_schedule_layout)
        training_layout.addWidget(lr_schedule_group)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Configuration")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def save_config():
            # Collect optimizer configuration
            opt_type = opt_type_combo.currentText()
            opt_params = {
                "learning_rate": lr_input.value()
            }

            # Add optimizer-specific parameters
            for param_name, widget in self.opt_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    opt_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    opt_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    opt_params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    opt_params[param_name] = widget.isChecked()

            # Create optimizer
            if opt_type == "Adam":
                self.optimizer = tf.keras.optimizers.Adam(**opt_params)
            elif opt_type == "SGD":
                self.optimizer = tf.keras.optimizers.SGD(**opt_params)
            elif opt_type == "RMSprop":
                self.optimizer = tf.keras.optimizers.RMSprop(**opt_params)
            elif opt_type == "AdamW":
                self.optimizer = tf.keras.optimizers.AdamW(**opt_params)

            # Collect training configuration
            self.training_config = {
                "batch_size": batch_input.value(),
                "epochs": epochs_input.value(),
                "validation_split": val_split_input.value(),
                "early_stopping": {
                    "enable": enable_es_checkbox.isChecked(),
                    "patience": patience_input.value(),
                    "min_delta": min_delta_input.value()
                }
            }

            # Collect learning rate schedule configuration
            schedule_type = schedule_type_combo.currentText()
            if schedule_type != "None":
                schedule_params = {}
                for param_name, widget in self.schedule_param_inputs.items():
                    if isinstance(widget, QSpinBox):
                        schedule_params[param_name] = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):
                        schedule_params[param_name] = widget.value()

                if schedule_type == "Exponential Decay":
                    self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=opt_params["learning_rate"],
                        **schedule_params
                    )
                elif schedule_type == "Cosine Decay":
                    self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                        **schedule_params
                    )
                elif schedule_type == "Step Decay":
                    self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=opt_params["learning_rate"],
                        decay_steps=schedule_params["decay_steps"],
                        decay_rate=schedule_params["decay_rate"],
                        staircase=True
                    )

                # Update optimizer with learning rate schedule
                self.optimizer.learning_rate = self.lr_schedule

            dialog.accept()

        save_btn.clicked.connect(save_config)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def create_training_visualization_tab(self):
        """Create the training visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Training curves
        curves_group = QGroupBox("Training Curves")
        curves_layout = QVBoxLayout()

        # Create matplotlib figure for training curves
        self.training_fig = Figure(figsize=(8, 6))
        self.training_canvas = FigureCanvas(self.training_fig)
        curves_layout.addWidget(self.training_canvas)

        # Metrics selection
        metrics_layout = QHBoxLayout()
        metrics_label = QLabel("Metrics to Plot:")
        self.metrics_combo = QComboBox()
        self.metrics_combo.addItems(["loss", "accuracy", "val_loss", "val_accuracy"])
        self.metrics_combo.setCurrentText("loss")
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_combo)
        curves_layout.addLayout(metrics_layout)

        # Update plot button
        update_plot_btn = QPushButton("Update Plot")
        update_plot_btn.clicked.connect(self.update_training_plot)
        curves_layout.addWidget(update_plot_btn)

        curves_group.setLayout(curves_layout)
        layout.addWidget(curves_group)

        # Weight gradients
        gradients_group = QGroupBox("Weight Gradients")
        gradients_layout = QVBoxLayout()

        # Create matplotlib figure for weight gradients
        self.gradients_fig = Figure(figsize=(8, 6))
        self.gradients_canvas = FigureCanvas(self.gradients_fig)
        gradients_layout.addWidget(self.gradients_canvas)

        # Layer selection for gradients
        layer_layout = QHBoxLayout()
        layer_label = QLabel("Layer:")
        self.layer_combo = QComboBox()
        layer_layout.addWidget(layer_label)
        layer_layout.addWidget(self.layer_combo)
        gradients_layout.addLayout(layer_layout)

        # Update gradients button
        update_gradients_btn = QPushButton("Update Gradients")
        update_gradients_btn.clicked.connect(self.update_gradients_plot)
        gradients_layout.addWidget(update_gradients_btn)

        gradients_group.setLayout(gradients_layout)
        layout.addWidget(gradients_group)

        # Test metrics
        metrics_group = QGroupBox("Test Metrics")
        metrics_layout = QVBoxLayout()

        # Create table for test metrics
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        metrics_layout.addWidget(self.metrics_table)

        # Update metrics button
        update_metrics_btn = QPushButton("Update Metrics")

    def update_training_plot(self):
        """Update the training visualization plot with selected metrics"""
        if not hasattr(self, 'training_history'):
            self.show_error("No training history available. Please train a model first.")
            return

        try:
            # Clear the current figure
            self.training_fig.clear()

            # Get selected metric
            metric = self.metrics_combo.currentText()

            # Create plot
            ax = self.training_fig.add_subplot(111)
            
            # Plot training metric
            if metric in self.training_history.history:
                ax.plot(self.training_history.history[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in self.training_history.history:
                ax.plot(self.training_history.history[val_metric], label=f'Validation {metric}')

            # Customize plot
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'Training and Validation {metric.capitalize()}')
            ax.grid(True)
            ax.legend()

            # Update canvas
            self.training_fig.tight_layout()
            self.training_canvas.draw()

            # Update metrics table
            self.update_metrics_table()

        except Exception as e:
            self.show_error(f"Error updating training plot: {str(e)}")

    def update_metrics_table(self):
        """Update the metrics table with current training metrics"""
        if not hasattr(self, 'training_history'):
            return

        try:
            # Get the final values for each metric
            metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
            final_values = {}

            for metric in metrics:
                if metric in self.training_history.history:
                    final_values[metric] = self.training_history.history[metric][-1]

            # Update table
            self.metrics_table.setRowCount(len(final_values))
            for i, (metric, value) in enumerate(final_values.items()):
                # Add metric name
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                # Add value with 4 decimal places
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

            # Resize columns to content
            self.metrics_table.resizeColumnsToContents()

        except Exception as e:
            self.show_error(f"Error updating metrics table: {str(e)}")

    def update_gradients_plot(self):
        """Update the weight gradients visualization plot"""
        if not hasattr(self, 'model') or not hasattr(self, 'training_history'):
            self.show_error("No model or training history available. Please train a model first.")
            return

        try:
            # Clear the current figure
            self.gradients_fig.clear()

            # Get selected layer
            layer_name = self.layer_combo.currentText()
            if not layer_name:
                return

            # Find the layer in the model
            layer = None
            for l in self.model.layers:
                if l.name == layer_name:
                    layer = l
                    break

            if layer is None:
                self.show_error(f"Layer {layer_name} not found in model")
                return

            # Get layer weights and gradients
            weights = layer.get_weights()
            if not weights:
                self.show_error(f"No weights found for layer {layer_name}")
                return

            # Create subplots for each weight matrix
            n_weights = len(weights)
            fig_rows = (n_weights + 1) // 2  # Ceiling division
            fig_cols = min(2, n_weights)

            for i, weight_matrix in enumerate(weights):
                ax = self.gradients_fig.add_subplot(fig_rows, fig_cols, i + 1)
                
                # Plot weight distribution
                ax.hist(weight_matrix.flatten(), bins=50)
                ax.set_title(f'Weight Distribution - {layer_name} (Matrix {i+1})')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')
                ax.grid(True)

            # Update canvas
            self.gradients_fig.tight_layout()
            self.gradients_canvas.draw()

        except Exception as e:
            self.show_error(f"Error updating gradients plot: {str(e)}")

    def add_layer_dialog(self):
        """Open a dialog to add a new layer to the model architecture"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Layer")
        layout = QVBoxLayout(dialog)

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems([
            "Dense",
            "Conv2D",
            "MaxPooling2D",
            "Flatten",
            "Dropout"
        ])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Parameters container
        params_container = QWidget()
        params_layout = QVBoxLayout(params_container)
        layout.addWidget(params_container)

        # Parameter widgets dictionary
        param_widgets = {}

        def update_params():
            # Clear existing parameter widgets
            for widget in list(param_widgets.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            param_widgets.clear()

            layer_type = type_combo.currentText()
            
            if layer_type == "Dense":
                # Units
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(64)
                param_widgets["units"] = units_input
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)

                # Activation
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax", "linear"])
                param_widgets["activation"] = activation_combo
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Conv2D":
                # Filters
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 512)
                filters_input.setValue(32)
                param_widgets["filters"] = filters_input
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)

                # Kernel size
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QSpinBox()
                kernel_input.setRange(1, 11)
                kernel_input.setValue(3)
                param_widgets["kernel_size"] = kernel_input
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)

                # Activation
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax", "linear"])
                param_widgets["activation"] = activation_combo
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "MaxPooling2D":
                # Pool size
                pool_label = QLabel("Pool Size:")
                pool_input = QSpinBox()
                pool_input.setRange(1, 8)
                pool_input.setValue(2)
                param_widgets["pool_size"] = pool_input
                params_layout.addWidget(pool_label)
                params_layout.addWidget(pool_input)

            elif layer_type == "Dropout":
                # Rate
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 0.9)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                param_widgets["rate"] = rate_input
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        # Connect layer type change to parameter update
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial parameter setup

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}

            # Collect parameters based on layer type
            if layer_type == "Dense":
                layer_params = {
                    "units": param_widgets["units"].value(),
                    "activation": param_widgets["activation"].currentText()
                }
            elif layer_type == "Conv2D":
                layer_params = {
                    "filters": param_widgets["filters"].value(),
                    "kernel_size": param_widgets["kernel_size"].value(),
                    "activation": param_widgets["activation"].currentText()
                }
            elif layer_type == "MaxPooling2D":
                layer_params = {
                    "pool_size": param_widgets["pool_size"].value()
                }
            elif layer_type == "Dropout":
                layer_params = {
                    "rate": param_widgets["rate"].value()
                }

            # Add layer to configuration
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })

            # Update layer list display
            self.update_layer_list()
            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def load_efficient_mnist_config(self):
        """Load a compute-efficient configuration for MNIST dataset"""
        # Clear existing layer configuration
        self.layer_config = []
        
        # Add layers for a simple but effective MNIST model
        self.layer_config.extend([
            {
                "type": "Conv2D",
                "params": {
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu"
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": 2
                }
            },
            {
                "type": "Conv2D",
                "params": {
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu"
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": 2
                }
            },
            {
                "type": "Flatten",
                "params": {}
            },
            {
                "type": "Dense",
                "params": {
                    "units": 128,
                    "activation": "relu"
                }
            },
            {
                "type": "Dropout",
                "params": {
                    "rate": 0.5
                }
            },
            {
                "type": "Dense",
                "params": {
                    "units": 10,
                    "activation": "softmax"
                }
            }
        ])
        
        # Update the layer list display
        self.update_layer_list()
        
        # Set default training configuration
        self.training_config = {
            "batch_size": 64,
            "epochs": 5,
            "validation_split": 0.2,
            "early_stopping": {
                "enable": True,
                "patience": 3,
                "min_delta": 0.001
            }
        }
        
        # Create optimizer with default learning rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.status_bar.showMessage("Loaded compute-efficient MNIST configuration")

    def stop_training(self):
        """Stop the current training process"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Stop the training by setting a flag
                self.training_stopped = True
                
                # Update UI
                self.training_status.setText("Training stopped")
                self.progress_bar.setValue(0)
                self.status_bar.showMessage("Training stopped by user")
                
                # If using early stopping callback, we can also stop it
                if hasattr(self, 'early_stopping_callback'):
                    self.early_stopping_callback.stopped_epoch = self.current_epoch
                    self.early_stopping_callback.model.stop_training = True
        except Exception as e:
            self.show_error(f"Error stopping training: {str(e)}")

    def update_model_summary(self):
        """Update the model summary text with detailed information about the model architecture"""
        if not self.layer_config:
            self.model_summary_text.setText("No model architecture defined yet.")
            return

        try:
            # Create a temporary model
            model = self.create_neural_network()
            
            # Determine input shape based on dataset
            if self.X_train is not None:
                input_shape = self.X_train.shape[1:]
            else:
                # Default input shape for MNIST
                input_shape = (28, 28, 1)
            
            # Build the model with input shape
            model.build(input_shape)
            
            # Get model summary as string
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            summary_text = "\n".join(summary_list)
            
            # Add additional information
            total_params = model.count_params()
            trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            # Format the summary text
            summary_text = f"Model Architecture Summary:\n\n{summary_text}\n\n"
            summary_text += f"Total Parameters: {total_params:,}\n"
            summary_text += f"Trainable Parameters: {trainable_params:,}\n"
            summary_text += f"Non-trainable Parameters: {non_trainable_params:,}\n\n"
            
            # Add layer details
            summary_text += "Layer Details:\n"
            for i, layer in enumerate(self.layer_config, 1):
                summary_text += f"\n{i}. {layer['type']}\n"
                for param_name, param_value in layer['params'].items():
                    summary_text += f"   - {param_name}: {param_value}\n"
            
            # Add input shape information
            summary_text += f"\nInput Shape: {input_shape}\n"
            
            self.model_summary_text.setText(summary_text)
            
        except Exception as e:
            self.model_summary_text.setText(f"Error generating model summary: {str(e)}")

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
