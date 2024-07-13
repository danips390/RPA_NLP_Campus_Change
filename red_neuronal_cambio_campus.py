import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import re
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

# Cargar el dataset
file_path = '/Users/Daniel/Documents/ITESM/Intern/RPAs/campus_changes.csv'
df = pd.read_csv(file_path)

# Preprocesar los datos
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('spanish'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['texto'] = df['texto'].apply(preprocess_text)
texts = df['texto'].tolist()
campuses = df['nuevo_campus'].tolist()

# Tokenización y creación de secuencias
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Codificación de etiquetas
label_encoder = LabelEncoder()
campuses_encoded = label_encoder.fit_transform(campuses)

# Padding de las secuencias
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = to_categorical(campuses_encoded, num_classes=len(label_encoder.classes_))

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Construcción del modelo
embedding_dim = 100
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo con conjunto de validación
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val))

# Evaluación del modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Guardar el modelo entrenado
model.save('/Users/Daniel/Documents/ITESM/Intern/RPAs/campus_ner_model_improved.keras')

# Guardar el tokenizer
tokenizer_json = tokenizer.to_json()
with open('/Users/Daniel/Documents/ITESM/Intern/RPAs/tokenizer.json', 'w') as f:
    f.write(json.dumps(tokenizer_json))

# Guardar el label encoder
np.save('/Users/Daniel/Documents/ITESM/Intern/RPAs/label_encoder.npy', label_encoder.classes_)

# Función para predecir el campus
def predict_campus(new_text):
    # Cargar el modelo
    model = tf.keras.models.load_model('/Users/Daniel/Documents/ITESM/Intern/RPAs/campus_ner_model_improved.keras')
    
    # Cargar el tokenizer
    with open('/Users/Daniel/Documents/ITESM/Intern/RPAs/tokenizer.json') as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    
    # Cargar el label encoder
    label_encoder_classes = np.load('/Users/Daniel/Documents/ITESM/Intern/RPAs/label_encoder.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
    
    # Preprocesar el texto
    preprocessed_text = preprocess_text(new_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Predecir el campus
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_campus = label_encoder.inverse_transform(predicted_label)
    
    return predicted_campus[0]
