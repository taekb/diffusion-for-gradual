import pickle
import pandas as pd
import numpy as np
from PIL import Image
import os
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split


data_dir = './10707_final_project' # Revise this as needed
data_path = osp.join(data_dir, 'age_gender.csv')

# Load data
with open(osp.join(data_dir, 'data.pkl'), 'rb') as fh:
    data_dict = pickle.load(fh)

# Visualize age distribution in the data
plt.figure(figsize=(10,6))
sns.histplot(data_dict['ages'])
plt.xlabel('Age')
plt.title('Age Distribution in Data')
plt.show()

#np.where(data_dict['ages'] == 92)

# Split according to 30% and 70% quantiles
def split_age_groups(data_dict):
    sorted_ages = np.sort(data_dict['ages'])
    source_age_cutoff = np.quantile(sorted_ages, 0.3)
    inter_age_cutoff = np.quantile(sorted_ages, 0.7)
    
    print(f'Source age range: 0-{int(np.floor(source_age_cutoff))}')
    print(f'Intermediate age range: {int(np.ceil(source_age_cutoff))}-{int(np.floor(inter_age_cutoff))}')
    print(f'Target age range: {int(np.ceil(inter_age_cutoff))}-{int(np.max(sorted_ages))}\n')

    # Sort the data according to age
    source_idx = np.where(data_dict['ages'] <= source_age_cutoff)
    inter_idx = np.where((data_dict['ages'] > source_age_cutoff) & 
                        (data_dict['ages'] <= inter_age_cutoff))
    target_idx = np.where(data_dict['ages'] > inter_age_cutoff)

    # Source data
    X_source = data_dict['X'][source_idx]
    X_source = X_source[...,None] # Add channel axis
    Y_source = data_dict['Y_gender'][source_idx]
    N_source = X_source.shape[0]

    # Intermediate data
    X_inter = data_dict['X'][inter_idx]
    X_inter = X_inter[...,None]
    Y_inter = data_dict['Y_gender'][inter_idx] # Will not be used
    N_inter = X_inter.shape[0]

    # Target data
    X_target = data_dict['X'][target_idx]
    X_target = X_target[...,None]
    Y_target = data_dict['Y_gender'][target_idx]
    N_target = X_target.shape[0]

    return [(X_source, Y_source), (X_inter, Y_inter), (X_target, Y_target)]

dataset = split_age_groups(data_dict)

generator_kwargs = dict(batch_size=256)

def get_model():
    return Sequential([
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same', input_shape=(48,48,1)),
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same'),
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same'),
        Dropout(0.5),
        BatchNormalization(),
        Flatten(),
        Dense(2, activation=softmax)
    ])
    
# Baseline: Train only on source and evaluate on target without any self-training
def exp1(source_data, inter_data, target_data):
    print(f'\nExperiment 1\n')

    print('Training source model...')
    model = get_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    # Train on source data
    X_source, Y_source = source_data
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2)

    datagen = ImageDataGenerator(rescale=1./255)
    datagen.fit(X_source_train)

    train_generator = datagen.flow(X_source_train, Y_source_train, **generator_kwargs)
    val_generator = datagen.flow(X_source_val, Y_source_val, **generator_kwargs)

    history = model.fit(
                  train_generator, 
                  validation_data=val_generator,
                  epochs=100,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
              )
    
    # Evaluate on the held-out target data
    X_target, Y_target = target_data
    N_target = X_target.shape[0]

    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)
    
    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test)
    target_test_generator = datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)

    final_loss, final_acc = model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    return model, final_acc

# Train on source, self-train on target, and evaluate on target
def exp2(source_data, inter_data, target_data):
    print(f'\nExperiment 2\n')
    
    print('Training source model...')
    model = get_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    # Train on source data
    X_source, Y_source = source_data
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2)

    datagen = ImageDataGenerator(rescale=1./255)
    datagen.fit(X_source_train)

    train_generator = datagen.flow(X_source_train, Y_source_train, **generator_kwargs)
    val_generator = datagen.flow(X_source_val, Y_source_val, **generator_kwargs)

    history = model.fit(
                  train_generator, 
                  validation_data=val_generator,
                  epochs=100,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
              )
    
    # Get train-val-test split on target data
    X_target, Y_target = target_data
    N_target = X_target.shape[0]
    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)

    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test)
    X_target_train, X_target_val, Y_target_train, Y_target_val = train_test_split(X_target_train, Y_target_train, test_size=N_target_val)

    # Generate pseudolabels
    target_train_generator = datagen.flow(X_target_train, shuffle=False, **generator_kwargs)
    target_val_generator = datagen.flow(X_target_val, shuffle=False, **generator_kwargs)

    Y_target_train_pseudo = tf.keras.utils.to_categorical(np.argmax(model.predict(target_train_generator), axis=1), num_classes=2)
    Y_target_val_pseudo = tf.keras.utils.to_categorical(np.argmax(model.predict(target_val_generator), axis=1), num_classes=2)

    # Update generators with pseudolabels
    new_datagen = ImageDataGenerator(rescale=1./255)
    new_datagen.fit(X_target_train)
    target_train_generator = new_datagen.flow(X_target_train, Y_target_train_pseudo, **generator_kwargs)
    target_val_generator = new_datagen.flow(X_target_val, Y_target_val_pseudo, **generator_kwargs)

    # Train new model on pseudolabeled target data
    # TODO: Check label sharpening?
    print('\nSelf-training with pseudolabeled target data...\n')
    new_model = get_model()
    new_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    new_history = new_model.fit(
                      target_train_generator, 
                      validation_data=target_val_generator,
                      epochs=100,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                  )
    
    # Evaluate on target test data using ground-truth labels
    target_test_generator = new_datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)
    final_loss, final_acc = new_model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    return new_model, final_acc

# Train on source, self-train on intermediate, self-train on target, and evaluate on target data 
# (Without retraining on data from previous distribution at each self-training step)
def exp3(source_data, inter_data, target_data):
    print(f'\nExperiment 3\n')

    # Source model
    print('Training source model...')
    source_model = get_model()
    source_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    # Train on source data
    X_source, Y_source = source_data
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2)

    source_datagen = ImageDataGenerator(rescale=1./255)
    source_datagen.fit(X_source_train)

    source_train_generator = source_datagen.flow(X_source_train, Y_source_train, **generator_kwargs)
    source_val_generator = source_datagen.flow(X_source_val, Y_source_val, **generator_kwargs)

    source_history = source_model.fit(
                         source_train_generator, 
                         validation_data=source_val_generator,
                         epochs=100,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                     )
    
    # Get train-val split on intermediate data
    X_inter, Y_inter = inter_data
    X_inter_train, X_inter_val, Y_inter_train, Y_inter_val = train_test_split(X_inter, Y_inter, test_size=0.2)

    # Generate pseudolabels for intermediate data using source model
    inter_train_generator = source_datagen.flow(X_inter_train, shuffle=False, **generator_kwargs)
    inter_val_generator = source_datagen.flow(X_inter_val, shuffle=False, **generator_kwargs)

    Y_inter_train_pseudo = to_categorical(np.argmax(source_model.predict(inter_train_generator), axis=1), num_classes=2)
    Y_inter_val_pseudo = to_categorical(np.argmax(source_model.predict(inter_val_generator), axis=1), num_classes=2)

    # Update generators with pseudolabels
    inter_datagen = ImageDataGenerator(rescale=1./255)
    inter_datagen.fit(X_inter_train)
    inter_train_generator = inter_datagen.flow(X_inter_train, Y_inter_train_pseudo, **generator_kwargs)
    inter_val_generator = inter_datagen.flow(X_inter_val, Y_inter_val_pseudo, **generator_kwargs)

    # Train new model on pseudolabeled intermediate data
    print('\nSelf-training with pseudolabeled intermediate data...\n')
    inter_model = get_model()
    inter_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    inter_history = inter_model.fit(
                        inter_train_generator, 
                        validation_data=inter_val_generator,
                        epochs=100,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                    )
    
    # Get train-val-test split on target data
    X_target, Y_target = target_data
    N_target = X_target.shape[0]
    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)

    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test)
    X_target_train, X_target_val, Y_target_train, Y_target_val = train_test_split(X_target_train, Y_target_train, test_size=N_target_val)

    # Generate pseudolabels
    target_train_generator = inter_datagen.flow(X_target_train, shuffle=False, **generator_kwargs)
    target_val_generator = inter_datagen.flow(X_target_val, shuffle=False, **generator_kwargs)

    Y_target_train_pseudo = to_categorical(np.argmax(inter_model.predict(target_train_generator), axis=1), num_classes=2)
    Y_target_val_pseudo = to_categorical(np.argmax(inter_model.predict(target_val_generator), axis=1), num_classes=2)

    # Update generators with pseudolabels
    target_datagen = ImageDataGenerator(rescale=1./255)
    target_datagen.fit(X_target_train)
    target_train_generator = target_datagen.flow(X_target_train, Y_target_train_pseudo, **generator_kwargs)
    target_val_generator = target_datagen.flow(X_target_val, Y_target_val_pseudo, **generator_kwargs)

    # Train new model on pseudolabeled target data
    print('\nSelf-training with pseudolabeled target data...\n')
    target_model = get_model()
    target_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    target_history = target_model.fit(
                      target_train_generator, 
                      validation_data=target_val_generator,
                      epochs=100,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                  )
    
    # Evaluate on target test data using ground-truth labels
    target_test_generator = target_datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)
    final_loss, final_acc = target_model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    return target_model, final_acc

# Train on source, self-train on intermediate, self-train on target, and evaluate on target data 
# (With retraining on data from previous distribution at each self-training step)
def exp4(source_data, inter_data, target_data):
    print(f'\nExperiment 4\n')

    # Source model
    print('Training source model...')
    source_model = get_model()
    source_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    # Train on source data
    X_source, Y_source = source_data
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2)

    source_datagen = ImageDataGenerator(rescale=1./255)
    source_datagen.fit(X_source_train)

    source_train_generator = source_datagen.flow(X_source_train, Y_source_train, **generator_kwargs)
    source_val_generator = source_datagen.flow(X_source_val, Y_source_val, **generator_kwargs)

    source_history = source_model.fit(
                         source_train_generator, 
                         validation_data=source_val_generator,
                         epochs=100,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                     )
    
    # Get train-val split on intermediate data
    X_inter, Y_inter = inter_data
    X_inter_train, X_inter_val, Y_inter_train, Y_inter_val = train_test_split(X_inter, Y_inter, test_size=0.2)

    # Generate pseudolabels for intermediate data using source model
    inter_train_generator = source_datagen.flow(X_inter_train, shuffle=False, **generator_kwargs)
    inter_val_generator = source_datagen.flow(X_inter_val, shuffle=False, **generator_kwargs)

    Y_inter_train_pseudo = to_categorical(np.argmax(source_model.predict(inter_train_generator), axis=1), num_classes=2)
    Y_inter_val_pseudo = to_categorical(np.argmax(source_model.predict(inter_val_generator), axis=1), num_classes=2)
    Y_inter_pseudo = np.concatenate([Y_inter_train_pseudo, Y_inter_val_pseudo])

    # Concatenate source and pseudolabeled data and update generators
    inter_datagen = ImageDataGenerator(rescale=1./255)
    inter_datagen.fit(np.concatenate([X_source, X_inter_train]))
    inter_train_generator = inter_datagen.flow(
                                np.concatenate([X_source, X_inter_train]), 
                                np.concatenate([Y_source, Y_inter_train_pseudo]), 
                                **generator_kwargs
                            )
    inter_val_generator = inter_datagen.flow(X_inter_val, Y_inter_val_pseudo, **generator_kwargs)

    # Train new model on pseudolabeled intermediate data
    print('\nSelf-training with source and pseudolabeled intermediate data...\n')
    inter_model = get_model()
    inter_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    inter_history = inter_model.fit(
                        inter_train_generator, 
                        validation_data=inter_val_generator,
                        epochs=100,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                    )
    
    # Get train-val-test split on target data
    X_target, Y_target = target_data
    N_target = X_target.shape[0]
    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)

    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test)
    X_target_train, X_target_val, Y_target_train, Y_target_val = train_test_split(X_target_train, Y_target_train, test_size=N_target_val)

    # Generate pseudolabels
    target_train_generator = inter_datagen.flow(X_target_train, shuffle=False, **generator_kwargs)
    target_val_generator = inter_datagen.flow(X_target_val, shuffle=False, **generator_kwargs)

    Y_target_train_pseudo = to_categorical(np.argmax(inter_model.predict(target_train_generator), axis=1), num_classes=2)
    Y_target_val_pseudo = to_categorical(np.argmax(inter_model.predict(target_val_generator), axis=1), num_classes=2)

    # Concatenate source and pseudolabeled data and update generators
    target_datagen = ImageDataGenerator(rescale=1./255)
    target_datagen.fit(np.concatenate([X_source, X_inter, X_target_train]))
    target_train_generator = target_datagen.flow(
                                np.concatenate([X_source, X_inter, X_target_train]), 
                                np.concatenate([Y_source, Y_inter_pseudo, Y_target_train_pseudo]), 
                                **generator_kwargs
                            )
    target_val_generator = target_datagen.flow(X_target_val, Y_target_val_pseudo, **generator_kwargs)

    # Train new model on pseudolabeled target data
    print('\nSelf-training with source and pseudolabeled intermediate + target data...\n')
    target_model = get_model()
    target_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    target_history = target_model.fit(
                      target_train_generator, 
                      validation_data=target_val_generator,
                      epochs=100,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
                  )
    
    # Evaluate on target test data using ground-truth labels
    target_test_generator = target_datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)
    final_loss, final_acc = target_model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    return target_model, final_acc

# Run all of the above experiments for given dataset
def run_exps(data, save_model=False):
    # Experiment 1
    exp1_model, exp1_acc = exp1(*data)

    # Experiment 2
    exp2_model, exp2_acc = exp2(*data)

    # Experiment 3
    exp3_model, exp3_acc = exp3(*data)

    # Experiment 4
    exp4_model, exp4_acc = exp4(*data)

    if save_model:
        results = dict(exp1={'model': exp1_model, 'acc': exp1_acc},
                       exp2={'model': exp2_model, 'acc': exp2_acc},
                       exp3={'model': exp3_model, 'acc': exp3_acc},
                       exp4={'model': exp4_model, 'acc': exp4_acc})
    else:
        del exp1_model
        del exp2_model
        del exp3_model
        del exp4_model
        del exp5_model
        del exp6_model

        results = dict(exp1={'acc': exp1_acc},
                       exp2={'acc': exp2_acc},
                       exp3={'acc': exp3_acc},
                       exp4={'acc': exp4_acc})

    return results