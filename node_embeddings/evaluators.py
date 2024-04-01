from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

class SingleLabelEvaluator:
    def evaluate(self, embedding, labels, labeled_portion):
        X = embedding.detach().numpy()
        y = labels
        
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=(1-labeled_portion))

        classifier = LogisticRegression(random_state=0, multi_class='ovr')
        classifier = classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        
        microf1 = f1_score(y_test, y_pred, average='micro')
        macrof1 = f1_score(y_test, y_pred, average='macro')
        return y_pred, microf1, macrof1


def threshold_probabilities(number_labels, y_probs):
    y_pred = np.zeros_like(y_probs)
    sort_index = np.flip(np.argsort(y_probs, axis=1), 1)
    for i in range(y_probs.shape[0]):
        for j in range(number_labels[i]):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred

class MultiLabelEvaluator:
    def evaluate(self, embedding, labels, labeled_portion):
        X = embedding.detach().numpy()
        y = labels
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-labeled_portion))

        classifier = OneVsRestClassifier(
            estimator=LogisticRegression(random_state=0)
        )
        classifier = classifier.fit(X_train, y_train)
        
        y_hat = classifier.predict_proba(X_test)

        # This is the same trick implemented by  "Graph Embedding on Biomedical Networks: Methods, Applications, and Evaluations" 
        # We assume the number of labels desired is known and use it to define the threshold
        number_labels = y_test.sum(axis=1).astype(int)
        y_pred = threshold_probabilities(number_labels, y_hat)
        
        microf1 = f1_score(y_test, y_pred, average='micro')
        macrof1 = f1_score(y_test, y_pred, average='macro')
        return y_pred, microf1, macrof1