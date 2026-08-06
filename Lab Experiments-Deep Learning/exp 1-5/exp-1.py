import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual and predicted labels
actual = np.array([
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Dog',
    'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

predicted = np.array([
    'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog',
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Create confusion matrix
conf_matrix = confusion_matrix(actual, predicted)

# Plot confusion matrix
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='g',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog'],
    cmap='RdPu'
)

plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.title("Confusion Matrix", fontsize=18)
plt.show()
