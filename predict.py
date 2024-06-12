
from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

drive.mount('/content/drive')

file_path = "recetas.csv"

recipes_df = pd.read_csv(file_path, delimiter='|')

tfidf_matrix = tfidf_vectorizer.fit_transform(recipes_df['Nombre']
labels = recipes_df['Categoria']  

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
return classification_report(y_test, y_pred))
