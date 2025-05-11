# -------- Step 1: Read and shuffle the data --------

def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = parts[0]
            text = ' '.join(parts[1:])
            data.append((text, label))
    return data

# File paths
train_file = './data/menu_train.txt'
dev_file = './data/menu_dev.txt'
test_file = './data/menu_test.txt'

# Read and shuffle data
train_data = read_data(train_file)
dev_data = read_data(dev_file)
test_data = read_data(test_file)


# Combine train and dev
full_train_data = train_data + dev_data

# -------- Step 2: Build vocabulary from training data --------
vocab = {}
for text, _ in full_train_data:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

# -------- Step 3: Convert text to Bag-of-Words vector --------
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text.lower().split():
        if word in vocab:
            index = vocab[word]
            vector[index] += 1
    return vector

# -------- Step 4: Prepare training and test sets --------
# Step 4.1: Map string labels to integers
label_to_index = {}
current_index = 0
for _, label in full_train_data + test_data:
    if label not in label_to_index:
        label_to_index[label] = current_index
        current_index += 1

# Step 4.2: Create feature vectors and integer labels
X_train = [text_to_vector(text) for text, label in full_train_data]
y_train = [label_to_index[label] for _, label in full_train_data]

X_test = [text_to_vector(text) for text, label in test_data]
y_test = [label_to_index[label] for _, label in test_data]


# -------- Step 5: Define Perceptron class --------
class Perceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = [0.0] * (len(X[0]) + 1)  # +1 for bias
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                activation = self.predict_raw(xi)
                error = target - (1 if activation >= 0 else -1)
                for i in range(len(xi)):
                    self.weights[i + 1] += self.eta * error * xi[i]
                self.weights[0] += self.eta * error  # bias update

    def predict_raw(self, xi):
        return sum(w * x for w, x in zip(self.weights[1:], xi)) + self.weights[0]

    def predict(self, X):
        return [1 if self.predict_raw(xi) >= 0 else -1 for xi in X]

# -------- Step 6: Train and evaluate Perceptron --------
perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)

# -------- Step 7: Evaluate Accuracy --------
correct = sum(1 for yp, yt in zip(y_pred, y_test) if yp == yt)
accuracy = correct / len(y_test) * 100
print("Accuracy:", round(accuracy, 2), "%")
