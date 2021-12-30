import numpy as np

conf_mat =\
[[457.,  0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.],
 [  0.,262.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
 [  0.,  1., 148.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
 [  0.,  0.,   0., 225.,   0.,   0.,   0.,   0.,   1.,   0.],
 [  0.,  0.,   0.,   0., 323.,   0.,   0.,   0.,   1.,   1.],
 [  0.,  0.,   0.,   0.,   0., 153.,   0.,   0.,   1.,   0.],
 [  1.,  2.,   0.,   0.,   0.,   0., 195.,   2.,   0.,   0.],
 [  1.,  0.,   0.,   0.,   0.,   0.,   0., 186.,   0.,   0.],
 [  0.,  0.,   0.,   3.,   0.,   0.,   0.,   1., 462.,   1.],
 [  0.,  0.,   0.,   0.,   0.,   1.,   0.,   0.,   0., 187.]]


conf_mat = np.asarray(conf_mat)

num_class = len(conf_mat)

TP = conf_mat.diagonal()  # True positive
AP = np.sum(conf_mat, axis=0)  # TP + FP
AT = np.sum(conf_mat, axis=1)  # TP + FN

precision = TP / AP
recall = TP / AT
F1 = (2 * precision * recall) / (precision + recall)
accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", F1)
