# %%
# IMPORT LIBRARY
from os import X_OK
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score, f1_score


data = pd.read_csv (r"D:\DOCUMENT\University\UNAIR\KULIAH\SEM 5\ANALISIS DAN VISUALISASI DATA\9 random forest\artikel + dataset\early+stage+diabetes+risk+prediction+dataset\diabetes_data_upload.csv")
print('Dataset :',data.shape)
data[:10] #read first 10 data

df = pd.DataFrame(data)
df


# %%
# change yes no to 1 0

columns_to_convert = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 
                      'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 
                      'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 
                      'Obesity']

for column in columns_to_convert:
    data[column] = data[column].map({'Yes': 1, 'No': 0})

data[:10] #read first 10 data


# %%
# change positive negative to 1 0
columns_to_convert = ['class']

for column in columns_to_convert:
    data[column] = data[column].map({'Positive': 1, 'Negative': 0})

data[:10] #read first 10 data

# %%
# change male female to 1 0
columns_to_convert = ['Gender']

for column in columns_to_convert:
    data[column] = data[column].map({'Male': 0, 'Female': 1})

data[:10] #read first 10 data

# %%
correlation_matrix = data.corr(method='pearson')
correlation_matrix

# %%
# Get the count of Neg
count_neg = data['class'].value_counts()[0]

print('Negatif diabetes : ', count_neg)

# Get the count of Pos
count_pos = data['class'].value_counts()[1]

print('Positif diabetes : ', count_pos)

y = np.array([count_neg, count_pos])
pie_label = ["Negatif", "Positif"]

plt.pie(y, labels=pie_label, autopct='%1.1f%%', startangle=90)
plt.show() 

# %%
sns.heatmap(data.corr())

# %%

# DataFrame
correlation_matrix = data.corr(method='pearson')

# Extract correlations with the 'class' column
class_correlations = correlation_matrix['class'].drop('class')  # Exclude correlation with itself

# Sort correlations in descending order
class_correlations_sorted = class_correlations.sort_values(ascending=False)

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=class_correlations_sorted.index, y=class_correlations_sorted.values, color='blue')
plt.title('Feature Correlation')
plt.xlabel('Variable')
plt.ylabel('Correlation')
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# %%
# top 5 highest correlation


correlation_matrix = data.corr(method='pearson')

# Extract correlations with the 'class' column
class_correlations = correlation_matrix['class'].drop('class')  # Exclude correlation with itself

# Sort correlations in descending order
class_correlations_sorted = class_correlations.sort_values(ascending=False)

# Select the top 5 highest correlations
top_5_correlations = class_correlations_sorted.head(5)

# Create a bar plot for the top 5 highest correlations
plt.figure(figsize=(12, 6))
sns.barplot(x=top_5_correlations.index, y=top_5_correlations.values, color='blue')
plt.title('Future Correlation (Absolute Value) vs. Characteristic')
plt.xlabel('Characteristic')
plt.ylabel('Future Correlation (Absolute Value)')
plt.xticks(rotation=0, ha='right')  # Rotate x-axis labels for better readability

# Set y-axis limits
plt.ylim(0.0, 0.8)

plt.show()


# %%
# ------------- RANDOM FOREST CLASSIFICATION (ALL VARIABLE) ---------------

# target = kolom "class"
X = data.drop('class', axis=1)
y = data['class']

# Split the data  (training=75 and testing=25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_all_75 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_all_75 = precision

f1score = f1_score(y_test, y_pred)

f1score_all_75 = f1score

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f'Accuracy: {accuracy_all_75:.4f}')
print(f'Precision: {precision_all_75:.4f}')
print(f'F1-Score: {f1score_all_75:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)

# %%
# ------------- RANDOM FOREST CLASSIFICATION (ALL VARIABLE) ---------------

# target = kolom "class"
X = data.drop('class', axis=1)
y = data['class']

# Split the data  (training=80 and testing=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_all_80 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_all_80 = precision

f1score = f1_score(y_test, y_pred)

f1score_all_80 = f1score

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f'Accuracy: {accuracy_all_80:.4f}')
print(f'Precision: {precision_all_80:.4f}')
print(f'F1-Score: {f1score_all_80:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)

# %%
# ------------- RANDOM FOREST CLASSIFICATION (ALL VARIABLE) ---------------

# target = kolom "class"
X = data.drop('class', axis=1)
y = data['class']

# Split the data  (training=70 and testing=30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_all_70 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_all_70 = precision

f1score = f1_score(y_test, y_pred)

f1score_all_70 = f1score

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f'Accuracy: {accuracy_all_70:.4f}')
print(f'Precision: {precision_all_70:.4f}')
print(f'F1-Score: {f1score_all_70:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)

# %%
# ------------- RANDOM FOREST CLASSIFICATION (ALL VARIABLE) ---------------

# target = kolom "class"
X = data.drop('class', axis=1)
y = data['class']

# Split the data  (training=60 and testing=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_all_60 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_all_60 = precision

f1score = f1_score(y_test, y_pred)

f1score_all_60 = f1score

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f'Accuracy: {accuracy_all_60:.4f}')
print(f'Precision: {precision_all_60:.4f}')
print(f'F1-Score: {f1score_all_60:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)

# %%
import matplotlib.pyplot as plt

skema = ['60/40', '70/30', '75/25', '80/20']
akurasi_skema = [accuracy_all_60, accuracy_all_70, accuracy_all_75, accuracy_all_80]
presisi_skema = [precision_all_60, precision_all_70, precision_all_75, precision_all_80]

# subplot 1
plt.bar(skema, akurasi_skema)
plt.title('Accuracy Skema Menggunakan Seluruh')
plt.xlabel('Skema')
plt.ylabel('Accuracy')

# Adding labels to the bars
for i in range(len(skema)):
    plt.text(skema[i], akurasi_skema[i], f'{akurasi_skema[i]:.3f}', ha='center', va='bottom')

# 3f nya buat naro 3 angka di belakang koma
plt.show()


# %%
# Library Import(numpy and matplotlib)
import pandas as pd 
import matplotlib.pyplot as plot 

# Make a data definition
_data=[["60/40",accuracy_all_60, precision_all_60, f1score_all_60],
      ["70/30",accuracy_all_70, precision_all_70, f1score_all_70],
      ["75/25",accuracy_all_75, precision_all_75, f1score_all_75],
      ["80/20",accuracy_all_80, precision_all_80, f1score_all_80]
     ]
 
# Colors 
graph_color =['blue', 'red', 'purple']
 
# Draw a multi-colored bar chart.
_df = pd.DataFrame(_data,columns=["Skema", "Accuracy", "Precision", "F1score"])
 
ax = _df.plot(x="Skema", y=["Accuracy", "Precision", "F1score"], kind="bar", figsize=(12,6), color=graph_color, width=0.9)

# Set y-axis range from 0.0 to 1.0
ax.set_ylim(0.0, 1.0)

# Adjust the top margin without affecting the height of the graph
plt.subplots_adjust(bottom=0.1, top=0.9)

ax.legend(["Accuracy", "Precision", "F1score"], loc='upper center', bbox_to_anchor=(0.13, 1.13), fancybox=True, shadow=True, ncol=2)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.xticks(rotation=0, ha='center')  # Rotate x-axis labels for better readability

# Display the plot
plt.show()


# %%
# DATA BARU 5 CORRELATION TERBESAR

selected_columns = ['Polyuria', 'Polydipsia', 'Gender', 'sudden weight loss', 'partial paresis', 'class']

# New dataframe
new_data = data[selected_columns].copy()

print(new_data)


# %%
# ------------- RANDOM FOREST CLASSIFICATION (WITH TOP 5 CORRELATION)---------------

# target = kolom "class"
# cuma pake polyuria, polydipsia, gender, sudden weight loss, sama partial paresis
X = new_data.drop('class', axis=1)
y = new_data['class']

# Split the data  (training=75 and testing=25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier_75 = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier_75.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier_75.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy_75 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_75 = precision

f1score = f1_score(y_test, y_pred)

f1score_75 = f1score

# Display evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1-Score: {f1score:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)


# %%
# ------------- RANDOM FOREST CLASSIFICATION (WITH TOP 5 CORRELATION)---------------

# target = kolom "class"
# cuma pake polyuria, polydipsia, gender, sudden weight loss, sama partial paresis
X = new_data.drop('class', axis=1)
y = new_data['class']

# Split the data  (training=60 and testing=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy_60 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_60 = precision

f1score = f1_score(y_test, y_pred)

f1score_60 = f1score

# Display evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1-Score: {f1score:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)


# %%
# ------------- RANDOM FOREST CLASSIFICATION (WITH TOP 5 CORRELATION)---------------

# target = kolom "class"
# cuma pake polyuria, polydipsia, gender, sudden weight loss, sama partial paresis
X = new_data.drop('class', axis=1)
y = new_data['class']

# Split the data  (training=70 and testing=30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy_70 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_70 = precision

f1score = f1_score(y_test, y_pred)

f1score_70 = f1score

# Display evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1-Score: {f1score:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)


# %%
# ------------- RANDOM FOREST CLASSIFICATION (WITH TOP 5 CORRELATION)---------------

# target = kolom "class"
# cuma pake polyuria, polydipsia, gender, sudden weight loss, sama partial paresis
X = new_data.drop('class', axis=1)
y = new_data['class']

# Split the data  (training=80 and testing=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier_80 = RandomForestClassifier(random_state=42)

# Train the model on the training data
rf_classifier_80.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier_80.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy_80 = accuracy

precision = average_precision_score(y_test, y_pred)

precision_80 = precision

f1score = f1_score(y_test, y_pred)

f1score_80 = f1score

# Display evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1-Score: {f1score:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)


# %%
import matplotlib.pyplot as plt

skema = ['60/40', '70/30', '75/25', '80/20']
akurasi_skema = [accuracy_60, accuracy_70, accuracy_75, accuracy_80]
presisi_skema = [precision_60, precision_70, precision_75, precision_80]

# subplot 1
plt.bar(skema, akurasi_skema)
plt.title('Accuracy Skema Menggunakan 5 Atribut')
plt.xlabel('Skema')
plt.ylabel('Accuracy')

# Adding labels to the bars
for i in range(len(skema)):
    plt.text(skema[i], akurasi_skema[i], f'{akurasi_skema[i]:.3f}', ha='center', va='bottom')

# 3f nya buat naro 3 angka di belakang koma
plt.show()


# %%
import matplotlib.pyplot as plt

skema = ['60/40', '70/30', '75/25', '80/20']
presisi_skema = [precision_60, precision_70, precision_75, precision_80]

plt.bar(skema, presisi_skema)
plt.title('Precision Skema Menggunakan 5 Atribut')
plt.xlabel('Skema')
plt.ylabel('Precision')

# Adding labels to the bars
for i in range(len(skema)):
    plt.text(skema[i], presisi_skema[i], f'{presisi_skema[i]:.3f}', ha='center', va='bottom')

# 3f nya buat naro 3 angka di belakang koma
plt.show()


# %%
# Library Import(numpy and matplotlib)
import pandas as pd 
import matplotlib.pyplot as plot 

# Make a data definition
_data=[["60/40",accuracy_60, precision_60, f1score_60],
      ["70/30",accuracy_70, precision_70, f1score_70],
      ["75/25",accuracy_75, precision_75, f1score_75],
      ["80/20",accuracy_80, precision_80, f1score_80]
     ]
 
# Colors 
graph_color =['blue', 'red', 'purple']
 
# Draw a multi-colored bar chart.
_df = pd.DataFrame(_data,columns=["Skema", "Accuracy", "Precision", "F1score"])
 
ax = _df.plot(x="Skema", y=["Accuracy", "Precision", "F1score"], kind="bar", figsize=(12,6), color=graph_color, width=0.9)

# Set y-axis range from 0.0 to 1.0
ax.set_ylim(0.0, 1.0)

# Adjust the top margin without affecting the height of the graph
plt.subplots_adjust(bottom=0.1, top=0.9)

ax.legend(["Accuracy", "Precision", "F1score"], loc='upper center', bbox_to_anchor=(0.13, 1.13), fancybox=True, shadow=True, ncol=2)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.xticks(rotation=0, ha='center')  # Rotate x-axis labels for better readability

# Display the plot
plt.show()


# %%
# ----------------- PREDICTION -------------------
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)


#INPUT
print("Isi 1 jika mengalami Polyuria, 0 jika tidak")
polyuria = int(input("Polyuria = "))
print(polyuria)

print("Isi 1 jika mengalami Polydipsia, 0 jika tidak")
polydipsia = int(input("Polydipsia = "))
print(polydipsia)

print("Isi 1 jika perempuan, 0 jika tidak")
gender = int(input("Gender (Male/Female) = "))
print(gender)

print("Isi 1 jika mengalami Sudden Weight Loss, 0 jika tidak")
swl = int(input("Sudden Weight Loss = "))
print(swl)

print("Isi 1 jika mengalami Partial Paresis, 0 jika tidak")
parpar = int(input("Partial Paresis = "))
print(parpar)

# Create a DataFrame with the user input
user_data = pd.DataFrame({
    'Polyuria': [polyuria],
    'Polydipsia': [polydipsia],
    'Gender': [gender],
    'Sudden Weight Loss': [swl],
    'Partial Paresis': [parpar]
})

# Print the user data DataFrame
print('User Data:\n', user_data)
print('User Data Shape:', user_data.shape)
print('User Data Columns:', user_data.columns)

Train = [polyuria, polydipsia, gender, swl, parpar]
print(Train)

test = pd.DataFrame(Train).T

predtest = rf_classifier_80.predict(test)

print("Classification =", predtest)

if predtest==1:
    print("Pasien terkena diabetes")
else:
    print("Pasien tidak terkena diabetes")



