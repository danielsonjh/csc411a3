import pickle
import csv

clf = pickle.load(open('model_final'))
y_test_pred = clf.predict(pickle.load(open('test_features')))

print(y_test_pred)

with open('submission.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['Id', 'Prediction'])
    id = 1
    while id <= len(y_test_pred):
        csvwriter.writerow([id, y_test_pred[id-1]])
        id += 1
