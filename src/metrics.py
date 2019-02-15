from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_eval_metrics(df):
    def compute_eval_metric_helper(y_true, y_pred):
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        cf = confusion_matrix(y_true=y_true, y_pred=y_pred)

        return {
            'accuracy': acc,
            'f1-macro': f1_macro,
            'confusian_matrix': cf
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
