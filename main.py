# Import custom modules
import data_info   # datasets information such as attacks classes or column names
import data_handler  # data loading, gathering and preprocessing (manipulation)
import evaluation_metrics  # evaluation metrics such as PCC, confusion matrix or score
import model_trainer  # model training (using classification algorithms e.g. Random Forest & Perceptron)


print("Loading data...")  #
traindata = data_handler.load("data10")  # Load training set and assign column names
testdata = data_handler.load("test10")   # Load testing  set and assign column names


print("ATTACK CLASSES IN KDD99 DATASET")
print("Number of different attacks in train:  ", traindata['attack_type'].unique().size)
print("Number of different attacks in test :  ", testdata['attack_type'].unique().size)


print("GATHERING (before classification) and PREPROCESSING data...")
attack_classes = data_info.map_classes()  # Get attacks 5 classes dictionary
data_handler.gathering(traindata, 'attack_type', attack_classes)  # apply gathering to training data
data_handler.gathering(testdata, 'attack_type', attack_classes)   # apply gathering to testing data

traindata = data_handler.preprocess(traindata)
testdata = data_handler.preprocess(testdata)


X_train = traindata.drop(columns=['attack_type'])
y_train = traindata['attack_type']
X_test = testdata.drop(columns=['attack_type'])
y_test = testdata['attack_type']


trained_model = model_trainer.train(X_train, y_train)  # Train model (on training data)


training_score = model_trainer.evaluate_model(trained_model, X_train, y_train)  # Performance on training data
testing_score = model_trainer.evaluate_model(trained_model, X_test, y_test)  # Performance on testing data

evaluation_metrics.print_score(training_score, testing_score)  # Print performance scores (accuracy) in tabulate form

y_pred = trained_model.predict(X_test)  # Predictions on testing data
pcc = evaluation_metrics.calculate_pcc(y_test, y_pred)  # Calculate PCC (Percent of Correct Classification)

evaluation_metrics.print_confusion_matrix(y_test, y_pred)  # Print confusion matrix in tabulate form and as heatmap
evaluation_metrics.print_pcc(pcc)  # Print PCC
