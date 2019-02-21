'''
Scoring Auxilary Function
Class to help provide organize method to keep track of scores and report them.
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import numpy as np
import seaborn as sns
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import metrics as met

class ScoreReport:
    def __init__(self, run_id, model_name, output_dir='./'):
        # Generate Directory for Score Report
        self.output_dir = output_dir + '/' + run_id.replace(' ', '_').lower()
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        self.rmse_list = []     # RMSE List
        self.logloss_list = []  # Log Loss List
        self.acc_list = []      # Accuracy List
        self.tpr_list = []      # True Positive Rate List
        self.fpr_list = []      # False Positive Rate List
        self.f1_list = []       # F-Score (F1 Metric) List
        self.auc_list = []      # AUC/ROC List
        self.auc_score = []     # AUC/ROC Score List
        self.conf_list = []     # Confusion Matrix List
        self.avg_conf = np.array([[0,0],[0,0]]) # Average Confusion Matrix

        self.pre_list = []      # Average Precision List
        self.rec_list = []      # Average Recall List
        self.kappa_list = []    # Average Cohen-Kappa Score

        self.mean_conf = np.array([[0,0],[0,0]])    # Mean Confusion Matrix
        self.mean_tpr = 0.0                         # Mean True Positive Rate List
        self.mean_fpr = np.linspace(0, 1, 100)

        self.report_list = []           # Score Report List
        self.model_name = model_name    # Model Name

    def append_result(self, y_true, y_pred, y_prob):
        fpr, tpr, thresholds = met.roc_curve(y_true, y_prob, pos_label=1)

        self.tpr_list.append(tpr)
        self.fpr_list.append(fpr)

        self.pre_list.append(met.precision_score(y_true, y_pred))
        self.rec_list.append(met.recall_score(y_true, y_pred))
        self.kappa_list.append(met.cohen_kappa_score(y_true, y_pred))

        self.rmse_list.append(met.mean_squared_error(y_true, y_pred))
        self.logloss_list.append(met.log_loss(y_true, y_pred))
        self.acc_list.append(met.accuracy_score(y_true, y_pred))
        self.f1_list.append(met.f1_score(y_true, y_pred, pos_label=1))
        self.avg_conf += np.array(met.confusion_matrix(y_true, y_pred))
        self.auc_list.append(met.auc(fpr, tpr))
        self.auc_score.append(met.roc_auc_score(y_true, y_pred))
        self.mean_tpr += interp(self.mean_fpr, fpr, tpr)
        self.mean_tpr[0] = 0.0

        self.report_list.append(met.classification_report(y_true, y_pred))

    def generate_report(self):
        # Generates a directory with a complete summary and plot.
        self.generate_rocplot()
        self.generate_score_report()

    def generate_score_report(self):
        output = open(self.output_dir + '/' + self.model_name.replace(' ', '_').lower() + '_score-report.md', 'w')
        output.write("# " + self.model_name + '\n')
        output.write('**Model Performance Score Report**\n\n')

        output.write('### K-Fold Classification Report\n')
        output.write('| K | RMSE | Log Loss | Accuracy | Precision | Recall | F-Measure | AUC | Kappa |\n')
        output.write('| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for i in range(len(self.acc_list)):
            output.write('| ' + str(i+1) + ' | '
                        + str(self.rmse_list[i]) + ' | '
                        + str(self.logloss_list[i]) + ' | '
                        + str(self.acc_list[i]) + ' | '
                        + str(self.pre_list[i]) + ' | '
                        + str(self.rec_list[i]) + ' | '
                        + str(self.f1_list[i]) + ' | '
                        + str(self.auc_score[i]) + ' | '
                        + str(self.kappa_list[i]) + ' |\n')

        output.write('\n### Average Confusion Matrix\n')
        conf = np.true_divide(np.array(self.avg_conf), len(self.acc_list))
        output.write('| | Pred POS | Pred NEG |\n')
        output.write('| --- | --- | --- |\n')
        output.write('| **True POS** | ' + str(conf[1, 1])  + ' | '+ str(conf[1, 0]) + ' |\n')
        output.write('| **True NEG** | ' + str(conf[0, 1])  + ' | '+ str(conf[0, 0]) + ' |\n')

        output.write('\n### Average Model Performance Metrics\n')
        output.write('| RMSE | LOGLOSS | ACC | PRE | REC | F1 | AUC | KAPP |\n')
        output.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        output.write('| ' + str(np.mean(self.rmse_list)) +
                    ' | ' + str(np.mean(self.logloss_list)) +
                    ' | ' + str(np.mean(self.acc_list)) +
                    ' | ' + str(np.mean(self.pre_list)) +
                    ' | ' + str(np.mean(self.rec_list)) +
                    ' | ' + str(np.mean(self.f1_list)) +
                    ' | ' + str(np.mean(self.auc_score)) +
                    ' | ' + str(np.mean(self.kappa_list)) + ' |\n')
        output.write('\n### AUC/ROC Plot\n')
        output.write('![ROC Plot]('+self.model_name.replace(' ', '_').lower() + '_auc-plot.png'+')\n')
        output.close()

    def generate_rocplot(self):
        # Plot Each K-Fold & Output CSV for Each Fold
        for i in range(len(self.acc_list)):
            self.output_roc_csv(i+1, self.tpr_list[i], self.fpr_list[i])
            plt.plot(self.fpr_list[i], self.tpr_list[i], lw=1, label='ROC Fold %d (area = %0.2f)' % (i+1, self.auc_score[i]))

        # Plot Random AUC Line
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')

        # Plot Mean AUC
        self.mean_tpr /= len(self.acc_list)
        self.mean_tpr[-1] = 1.0
        mean_auc = np.mean(self.auc_score)
        plt.plot(self.mean_fpr, self.mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        self.output_roc_csv(len(self.acc_list), self.mean_tpr, self.mean_fpr, True)

        # Plot Axis and Normalize Graph
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.model_name+' '+str(len(self.acc_list))+'-Fold ROC')
        plt.legend(loc="lower right")
        plt.savefig(self.output_dir + '/' + self.model_name.replace(' ', '_').lower() + '_auc-plot.png')

    def output_roc_csv(self, k, tpr, fpr, average=False):
        if average:
            output = open(self.output_dir + '/' + self.model_name.replace(' ', '_').lower() + '_AVG_KF-'+str(k)+'.csv', 'w')
        else:
            output = open(self.output_dir + '/' + self.model_name.replace(' ', '_').lower() + '_KF-'+str(k)+'.csv', 'w')
        for t, f in zip(tpr, fpr): output.write(str(t)+','+str(f)+'\n')
        output.close()

# Unit Testing
if __name__ == '__main__':
    report = ScoreReport('Sample Model')
    y_pred = [1, 0, 1]
    y_true = [1, 0, 1]
    y_prob = [0.65, 0.78, 0.95]
    report.append_result(y_true, y_pred, y_prob)

    y_pred = [1, 0, 1]
    y_true = [1, 0, 1]
    y_prob = [0.65, 0.78, 0.95]
    report.append_result(y_true, y_pred, y_prob)

    y_pred = [0, 1, 0]
    y_true = [1, 0, 1]
    y_prob = [0.65, 0.78, 0.95]
    report.append_result(y_true, y_pred, y_prob)

    report.generate_rocplot()
    report.generate_score_report()
