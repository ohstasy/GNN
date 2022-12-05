import chemprop
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


test_path = 'tests/data/Halicin_test.csv'
preds_path = 'classification_checkpoints/param_self_attention/exp1/'

train_arguments = [
    '--data_path', 'tests/data/Halicin_train.csv',
    '--seed','42',
    '--dataset_type', 'classification',
    '--save_dir', preds_path,
    '--ffn_num_layers', '2',
    '--num_folds', '3',
    '--extra_metrics','accuracy',
    '--aggregation','param_self_attention'

]



train_args = chemprop.args.TrainArgs().parse_args(train_arguments)

mean_score, std_score = chemprop.train.cross_validate(args=train_args, train_func=chemprop.train.run_training)


arguments = [
    '--test_path', test_path,
    '--preds_path', preds_path + 'Halicin_predictions.csv',
    '--checkpoint_dir', preds_path,
    '--num_workers','0'
]


args = chemprop.args.PredictArgs().parse_args(arguments)
preds = chemprop.train.make_predictions(args=args)




y_true = pd.read_csv(test_path)
y_preds = pd.read_csv(preds_path +'Halicin_predictions.csv')
fpr, tpr, thresholds = roc_curve(y_true['Active'], y_preds['Active'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
df = pd.read_csv(preds_path + 'test_scores.csv')
df.head()
df['Test_AUC'] = roc_auc
df.to_csv(preds_path + 'test_scores_full.csv', mode='w', index=False)
plt.plot(fpr, tpr, color='darkorange', label="ROC curve")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic / AUC")
plt.legend(loc="lower right")
plt.savefig(preds_path + 'exp1.png')


