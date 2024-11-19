import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report

from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model
from architectures.helpers.wandb_handler import initialize_wandb
from architectures.helpers.custom_callbacks import CustomCallback

from architectures.helpers.model_handler import get_model

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from wandb.keras import WandbCallback

hyperparameters = hyperparameters[selected_model]
t = time.time()
epoch_counter = 1

''' Dataset Preperation
'''


def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for etf in etf_list:
        x_train.extend(
            np.load(f"ETF/TrainData/x_{etf}.npy"))
        y_train.extend(
            np.load(f"ETF/TrainData/y_{etf}.npy"))
        x_test.extend(
            np.load(f"ETF/TestData/x_{etf}.npy"))
        y_test.extend(
            np.load(f"ETF/TestData/y_{etf}.npy"))

    #x_train_new = []
    #y_train_new = []
    #for x_t, y_t in zip(x_train, y_train):
    #    if y_t != 1:
    #        x_train_new.append(x_t)
    #        y_train_new.append(y_t)
    #        x_train_new.append(x_t)
    #        y_train_new.append(y_t)

    #x_train.extend(x_train_new)
    #y_train.extend(y_train_new)
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique, counts)).T)
    return x_train, y_train, x_test, y_test


def prepare_dataset(x_train, y_train):
    val_split = 0.01

    val_indices = int(len(x_train) * (1-val_split))
    #new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
    #x_val, y_val = x_train[:val_indices], y_train[:val_indices]

    new_x_train, new_y_train = x_train[:val_indices], y_train[:val_indices]
    x_val, y_val = x_train[val_indices:], y_train[val_indices:]

    print(f"Training data samples: {len(new_x_train)}")
    print(f"Validation data samples: {len(x_val)}")
    print(f"Test data samples: {len(x_test)}")

    return new_x_train, new_y_train, x_val, y_val


def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    #if is_train:
    #    dataset = dataset.shuffle(hyperparameters["batch_size"] * 10)
    dataset = dataset.batch(hyperparameters["batch_size"])
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test):
    train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
    val_dataset = make_datasets(x_val, y_val)
    test_dataset = make_datasets(x_test, y_test)
    return train_dataset, val_dataset, test_dataset


#def run_experiment(model, test_dataset, x_test, y_test):
def run_experiment(model, train_dataset, val_dataset, x_train, y_train):
    early_stopping = tf.keras.callbacks.EarlyStopping(
         monitor="val_loss", patience=5, restore_best_weights=True
     )
    #callback_list = [CustomCallback(
    #    test_dataset, epoch_counter, t, y_test), WandbCallback()]
    #callback_list = [CustomCallback(
    #    test_dataset, epoch_counter, t, y_test)]
    #callback_list = [early_stopping]
    
    #if hyperparameters["learning_rate_type"] != "WarmUpCosine" and hyperparameters["learning_rate_type"] != "Not found":
    #    callback_list.append(hyperparameters["learning_rate_scheduler"])

    #print(train_dataset)
    #import sys
    #sys.exit(1)

    #callback_list = [early_stopping]
    callback_list = []
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=hyperparameters["num_epochs"],
        callbacks=callback_list,
    )

    filepath = "my_model_10epoch.weights.h5"
    model.save_weights(filepath)
    model2 = get_model()
    model2.load_weights(filepath)
    #import sys
    #sys.exit(1)
    
    #filepath = "my_model.keras"
    #model.save(filepath)
    #from keras.models import load_model
    #model = load_model(filepath)
    
    preds = []
    for ind in range(len(x_train)):
        image = tf.convert_to_tensor(x_train[ind].reshape(1,65,65))
        prediction = model2.predict(image)
        preds.append(prediction)
    print(preds)
    import sys
    sys.exit(1)
    
    loss, accuracy, *anything_else = model2.evaluate(train_dataset)
    print(loss)
    print(accuracy)
    print(anything_else)
    import sys
    sys.exit(1)
    
    print(
        f"Test accuracy: {round(accuracy * 100, 2)}%, Test loss: {round(loss, 4)}")
    print(f"Anything else: {anything_else}")
    return history, model


if __name__ == "__main__":
    #initialize_wandb()
    x_train, y_train, x_test, y_test = load_dataset()

    #d = {}
    #for item2 in y_test:
    #    item = item2[0] 
    #    if item not in d:
    #        d[item] = 0
    #    d[item] += 1    
    #print(d)
    #import sys
    #sys.exit(1)
    
    new_x_train, new_y_train, x_val, y_val = prepare_dataset(
        x_train, y_train)
    train_dataset, val_dataset, test_dataset = get_finalized_datasets(
        new_x_train, new_y_train, x_val, y_val, x_test, y_test)
    model = get_model()
    history, trained_model = run_experiment(model, train_dataset, val_dataset, x_train, y_train)

    predictions = trained_model.predict(test_dataset)
    classes = np.argmax(predictions, axis=1)
    cf = confusion_matrix(y_test, classes)
    print(f"\n{cf}")
    cr = classification_report(y_test, classes)
    print(f"\n{cr}")
    f1 = f1_score(y_test, classes, average='micro')
    print(f"\nF1 score: {f1}")
