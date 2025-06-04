import argparse
import csv
import json
import math
import os

DATA_FILE = 'training_data.csv'
MODEL_FILE = 'model.json'


def load_data():
    data = []
    labels = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                *features, label = row
                data.append([float(x) for x in features])
                labels.append(int(label))
    return data, labels


def save_example(features, label):
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(features) + [label])


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def train(lr=0.1, epochs=1000):
    data, labels = load_data()
    if not data:
        print('No training data found.')
        return
    n_features = len(data[0])
    weights = [0.0] * n_features
    bias = 0.0
    for _ in range(epochs):
        for x, y in zip(data, labels):
            z = sum(w * xi for w, xi in zip(weights, x)) + bias
            pred = logistic(z)
            error = pred - y
            for i in range(n_features):
                weights[i] -= lr * error * x[i]
            bias -= lr * error
    with open(MODEL_FILE, 'w') as f:
        json.dump({'weights': weights, 'bias': bias}, f)
    print('Model trained and saved to', MODEL_FILE)


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError('Model not found. Train first.')
    with open(MODEL_FILE, 'r') as f:
        model = json.load(f)
    return model['weights'], model['bias']


def predict(features):
    weights, bias = load_model()
    z = sum(w * x for w, x in zip(weights, features)) + bias
    p = logistic(z)
    return 1 if p >= 0.5 else 0, p


def main():
    parser = argparse.ArgumentParser(description='Simple AI Trainer')
    sub = parser.add_subparsers(dest='command')

    add_cmd = sub.add_parser('add', help='Add training example')
    add_cmd.add_argument('x1', type=float, help='First feature')
    add_cmd.add_argument('x2', type=float, help='Second feature')
    add_cmd.add_argument('label', type=int, help='Expected label (0 or 1)')

    sub.add_parser('train', help='Train the model')

    test_cmd = sub.add_parser('test', help='Test the model with features')
    test_cmd.add_argument('x1', type=float, help='First feature')
    test_cmd.add_argument('x2', type=float, help='Second feature')

    args = parser.parse_args()

    if args.command == 'add':
        save_example([args.x1, args.x2], args.label)
        print('Example added.')
    elif args.command == 'train':
        train()
    elif args.command == 'test':
        label, prob = predict([args.x1, args.x2])
        print(f'Predicted label: {label} (prob={prob:.2f})')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
