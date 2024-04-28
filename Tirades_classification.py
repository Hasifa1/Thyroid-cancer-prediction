#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
for dirname, _, filenames in os.walk('C:/5_classes_dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_dir="C:/5_classes_dataset"


# In[3]:


Classes = ['TIRADS1','TIRADS2','TIRADS3','TIRADS4','TIRADS5']


# In[4]:


import cv2
import numpy as np
train_data = []
img_size=224
def get_training_data():
    for label in  Classes:
        path=os.path.join(train_dir, label)
        class_num = Classes.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                train_data.append([resized_arr, class_num])
            except Exception as e:
                pass


# In[5]:


get_training_data()


# In[6]:


print(len(train_data))


# In[7]:


for label in Classes:
    print(Classes.index(label))


# In[8]:


x=[]
y=[]
for i,j in train_data:
    x.append(i)
    y.append(j)
x=np.array(x).reshape(-1,img_size, img_size,3)


# In[9]:


import numpy as np
x=np.array(x)


# In[10]:


x.shape


# In[11]:


y=np.array(y)


# In[12]:


print(y.shape)
print(y)


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_temp,y_train,y_temp=train_test_split(x,y,random_state=42,test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


# In[13]:


print(x_train.shape)


# In[14]:


print(y_val)


# In[15]:


from tensorflow.keras.utils import to_categorical
y_train= to_categorical(y_train, num_classes=5)


# In[16]:


y_val= to_categorical(y_val, num_classes=5)
y_test= to_categorical(y_test, num_classes=5)


# In[17]:


print(x_train.shape,y_val.shape)


# In[18]:


get_ipython().system('pip install matplotlib')
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import utils
import os
from keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.utils import img_to_array,load_img
from keras.preprocessing.image import  ImageDataGenerator
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy
base_model = MobileNet( input_shape=(224,224,3), include_top= False )
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(units=5 , activation='softmax' )(x)
# creating our model.
model = Model(base_model.input, x)


# In[27]:


from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(units=5 , activation='softmax' )(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)


# In[ ]:


base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=output)


# In[19]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(units=5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)


# In[20]:


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


# In[21]:


model.summary()


# In[22]:


model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[23]:


model.reset_states()


# In[24]:


model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))


# In[25]:


#Evaluate the model on your test data.
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%")


# In[26]:


# Extract features using the trained CNN model
X_train_features = model.predict(x_train)
X_test_features = model.predict(x_test)


# In[27]:


# Flatten the features
train_features_flat = X_train_features.reshape((X_train_features.shape[0], -1))
test_features_flat = X_test_features.reshape((X_test_features.shape[0], -1))


# In[28]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[29]:


print(y_train[300])


# In[30]:


y_train = np.argmax(y_train, axis=1)


# In[31]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[38]:


# Train SVM classifier on extracted features
svm = SVC(kernel='linear')
svm.fit(train_features_flat, y_train)


# In[39]:


y_test = np.argmax(y_test, axis=1)


# In[40]:


# Evaluate the SVM classifier
svm_predictions = svm.predict(test_features_flat)
accuracy = accuracy_score(y_test, svm_predictions)
print(f'SVM Accuracy: {accuracy}')


# In[41]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Accuracy
accuracy = accuracy_score(y_test, svm_predictions)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, svm_predictions,average='weighted')
print("Precision:", precision)

# Recall
recall = recall_score(y_test, svm_predictions,average='weighted')
print("Recall:", recall)

# F1 Score (harmonic mean of precision and recall)
f1 = f1_score(y_test, svm_predictions,average='weighted')
print("F1 Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, svm_predictions)
print("Confusion Matrix:")
print(conf_matrix)


# In[42]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:





# In[43]:


# Train and evaluate KNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(train_features_flat, y_train)
knn_predictions = knn_classifier.predict(test_features_flat)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("Accuracy of KNN Classifier:", knn_accuracy)


# In[44]:


nb_classifier = GaussianNB()
nb_classifier.fit(train_features_flat, y_train)
nb_predictions = nb_classifier.predict(test_features_flat)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Accuracy of Naive Bayes Classifier:", nb_accuracy)


# In[45]:


# Train and evaluate Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(train_features_flat, y_train)
dt_predictions = dt_classifier.predict(test_features_flat)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Accuracy of Decision Tree Classifier:", dt_accuracy)


# In[46]:


# Train and evaluate Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_features_flat, y_train)
rf_predictions = rf_classifier.predict(test_features_flat)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Accuracy of Random Forest Classifier:", rf_accuracy)


# In[47]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Accuracy
accuracy = accuracy_score(y_test, rf_predictions)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, rf_predictions,average='weighted')
print("Precision:", precision)

# Recall
recall = recall_score(y_test, rf_predictions,average='weighted')
print("Recall:", recall)

# F1 Score (harmonic mean of precision and recall)
f1 = f1_score(y_test, rf_predictions,average='weighted')
print("F1 Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, rf_predictions)
print("Confusion Matrix:")
print(conf_matrix)


# In[49]:


classifiers = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes':GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier()
}


# In[50]:


# Perform GridSearchCV for hyperparameter tuning and cross-validation
best_accuracy = 0
best_classifier = None
for clf_name, clf in classifiers.items():
    grid_clf = GridSearchCV(clf, param_grid={}, cv=3)
    grid_clf.fit(X_train_features, y_train)
    accuracy = grid_clf.score(X_test_features, y_test)
    print(f'{clf_name} Accuracy: {accuracy}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = grid_clf.best_estimator_

# Evaluate the best classifier
svm_predictions = best_classifier.predict(X_test_features)
accuracy = accuracy_score(y_test, svm_predictions)
print(f'Best Classifier Accuracy: {accuracy}')


# In[79]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve, auc
import matplotlib.pyplot as plt


# In[54]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, svm_predictions)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(5))
plt.yticks(np.arange(5))
for i in range(5):
    for j in range(5):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if i == j else 'black')
plt.show()


# In[ ]:





# In[63]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

# Create a one-vs-rest classifier with the SVC classifier
ovr_classifier = OneVsRestClassifier(SVC(probability=True))
ovr_classifier.fit(X_train_features, y_train)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, ovr_classifier.predict_proba(X_test_features)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[77]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_features, y_train)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, rf_classifier.predict_proba(X_test_features)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[80]:


print("Classification Report:")
print(classification_report(y_test, rf_classifier.predict(X_test_features)))


# In[81]:


precision = dict()
recall = dict()
pr_auc = dict()
for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(y_test == i, rf_classifier.predict_proba(X_test_features)[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

plt.figure(figsize=(8, 6))
for i in range(5):
    plt.plot(recall[i], precision[i], label=f'PR curve (class {i}) (AUC = {pr_auc[i]:0.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()


# In[1]:


# 4. Accuracy Score
accuracy = accuracy_score(y_test, rf_classifier.predict(X_test_features))
print("Accuracy Score:", accuracy)

# 5. F1 Score
f1 = f1_score(y_test, rf_classifier.predict(X_test_features), average='weighted')
print("F1 Score:", f1)


# In[ ]:




