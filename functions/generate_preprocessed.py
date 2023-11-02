import pandas as pd

def create_attributes(X, y, protected_cols):
    dataset = pd.concat([X, y], axis=1)
    column_labels = pd.DataFrame(0, columns=dataset.columns, index=dataset.index)
    column_labels[protected_cols] = 1
    column_labels['completed'] = 2
    column_labels = column_labels.iloc[:1]
    return dataset, column_labels  