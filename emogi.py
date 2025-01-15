from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy

data = [
    ("😊", "feliz"),
    ("😃", "feliz"),
    ("😂", "feliz"),
    ("😢", "triste"),
    ("😭", "triste"),
    ("😠", "enojado"),
    ("😡", "enojado"),
    ("❤️", "amor"),
    ("😍", "amor"),
    ("😴", "cansado"),
    ("🤔", "pensativo")
]


def extract_features(emoji):
    return {f"emoji_{emoji}": True}  # Usa el emoji como una característica



# Preparar los datos
features = [(extract_features(emoji), label) for (emoji, label) in data]

# Dividir los datos en entrenamiento y prueba
train_data = features[:8]
test_data = features[8:]

# Entrenar el modelo
classifier = NaiveBayesClassifier.train(train_data)

# Evaluar el modelo
accuracy_score = accuracy(classifier, test_data)
print(f"Accuracy: {accuracy_score * 100:.2f}%")


# Función para clasificar emojis
def classify_emoji(emoji, model):
    features = extract_features(emoji)
    return model.classify(features)

# Prueba con un emoji
emoji_input = input("Introduce un emoji: ")
result = classify_emoji(emoji_input, classifier)
print(f"El emoji '{emoji_input}' se clasifica como: {result}")


