import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# -------------------- Dataset Loading --------------------
def load_eye_images(data_dir, target_size=(64, 64)):
    """Load eye images from 'open' and 'closed' folders and normalize them."""
    images, labels = [], []
    categories = ['open', 'closed']
    
    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(idx)
    
    images = np.array(images, dtype='float32') / 255.0  # normalize
    labels = np.array(labels)
    return images, labels

# Load the dataset
X, y = load_eye_images('dataset')
y_categorical = to_categorical(y, num_classes=2)

# -------------------- CNN Model Definition --------------------
def build_cnn(input_shape=(64, 64, 3)):
    """Create and compile a simple CNN model for eye state classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Data Augmentation --------------------
augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
augmentation.fit(X)

# -------------------- Cross-Validation Training --------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_accuracies, all_aucs = [], []

for fold_num, (train_indices, test_indices) in enumerate(kf.split(X, y), start=1):
    print(f"\n--- Training Fold {fold_num} ---")
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_categorical[train_indices], y_categorical[test_indices]
    
    # Compute class weights to balance dataset
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y[train_indices]),
        y=y[train_indices]
    )
    class_weights_dict = dict(enumerate(weights))
    
    model = build_cnn()
    
    # Train model with data augmentation
    history = model.fit(
        augmentation.flow(X_train, y_train, batch_size=32),
        epochs=30,
        class_weight=class_weights_dict,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate fold
    predictions = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, predictions[:, 1])
    
    print(f"Fold {fold_num} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    all_accuracies.append(acc)
    all_aucs.append(auc)
    
    # Plot ROC for this fold
    fpr, tpr, _ = roc_curve(y_true, predictions[:, 1])
    plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {fold_num}")

# -------------------- Display ROC Curves --------------------
plt.title("ROC Curves Across Folds")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -------------------- Average Metrics --------------------
print(f"Average Accuracy: {np.mean(all_accuracies):.4f}")
print(f"Average AUC: {np.mean(all_aucs):.4f}")

# -------------------- Save Final Model --------------------
os.makedirs('models', exist_ok=True)
model.save('models/drowsiness_detector_cnn.h5')
print("Saved trained model to 'models/drowsiness_detector_cnn.h5'")
