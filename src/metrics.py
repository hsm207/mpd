from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def compute_eval_metrics(df):
    def compute_eval_metric_helper(y_true, y_pred):
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        cf = confusion_matrix(y_true=y_true, y_pred=y_pred)

        return {
            'accuracy': acc,
            'f1-macro': f1_macro,
            'confusion_matrix': cf
        }

    malay_publications = ['bh', 'hm']
    df['lang'] = 'bm'
    df['lang'] = df['lang'].where(df['pub'].isin(malay_publications), 'en')

    bm_target, bm_pred = df.query('lang == "bm"')[['target', 'pred']].T.values
    en_target, en_pred = df.query('lang == "en"')[['target', 'pred']].T.values

    metrics = {
        'overall': compute_eval_metric_helper(df['target'], df['pred']),
        'bm': compute_eval_metric_helper(y_true=bm_target, y_pred=bm_pred),
        'en': compute_eval_metric_helper(y_true=en_target, y_pred=en_pred)
    }

    return metrics


def print_metrics(metrics):
    print(f'Overall accuracy:{metrics["overall"]["accuracy"]:>11.4f}')
    print(f'Overall f1-macro:{metrics["overall"]["f1-macro"]:>11.4f}\n')

    print(f'BM accuracy:{metrics["bm"]["accuracy"]:>16.4f}')
    print(f'BM f1-macro:{metrics["bm"]["f1-macro"]:>16.4f}\n')

    print(f'EN accuracy:{metrics["en"]["accuracy"]:>16.4f}')
    print(f'EN f1-macro:{metrics["en"]["f1-macro"]:>16.4f}\n')

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()