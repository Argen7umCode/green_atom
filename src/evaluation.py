from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch


class Evaluator:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self):
        self.model.eval()
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)

                true_labels.extend(labels.tolist())
                predicted_labels.extend(predicted.tolist())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion
        }

    def visualize_confusion_matrix(self, confusion_matrix):
        # Здесь можно написать код для визуализации матрицы ошибок
        pass

    # Другие методы для оценки и визуализации метрик
