import pickle
import os
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, MaxPool2D, ReLU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# CONSTANTS
DATA_PATH = './dataset.pkl'
DIFF_PATH_1 = './class0_compressed_12288x64x64.npz'
DIFF_PATH_2 = './class1_compressed_12288x64x64.npz'
DEFAULT_SEED = 1
GENERATOR_CONFIG = dict(
    batch_size=256
)

'''
    Experimental Settings:
    1. No Adaptation (Train on Source, Evaluate on Target without Any Adaptation on Unlabeled Target Data)
    2. Direct Adaptation (Train on Source, Self-Train on Unlabeled Target, and Evaluate on Target)
    3. Gradual Adaptation (With Varying # of Intermediate Domain Samples)
    4. Oracle Model (Train on Target, Evaluate on Target)
'''

def get_model(dropout=0.5, seed=DEFAULT_SEED):
    '''Returns a 3-layer CNN used for all experiments.'''

    return Sequential([
        # First CNN block
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same', input_shape=(64,64,1),
               kernel_initializer=GlorotUniform(seed=seed)), # [None,64,64,32]
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same',
               kernel_initializer=GlorotUniform(seed=seed)), # [None,64,64,32]
        BatchNormalization(),
        ReLU(),
        MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'), # [None,32,32,32]
        Dropout(dropout),

        # Second CNN block
        Conv2D(64, (5,5), (2,2), activation=relu, padding='same',
               kernel_initializer=GlorotUniform(seed=seed)), # [None,32,32,64]
        Conv2D(64, (5,5), (2,2), activation=relu, padding='same',
               kernel_initializer=GlorotUniform(seed=seed)), # [None,32,32,64]
        BatchNormalization(),
        ReLU(),
        Dropout(dropout),

        # Dense layer
        Flatten(),
        Dense(2, activation=softmax, kernel_initializer=GlorotUniform(seed=seed))
    ])

def get_model_orig(dropout=0.5, seed=DEFAULT_SEED):
    '''Original CNN model.'''

    return Sequential([
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same', input_shape=(64,64,1)),
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same'),
        Conv2D(32, (5,5), (2,2), activation=relu, padding='same'),
        Dropout(0.5),
        BatchNormalization(),
        Flatten(),
        Dense(2, activation=softmax)
    ])

def load_data():
    '''Loads the source, intermediate, and target domain datasets and the fine-tuned diffusion-model samples.'''

    # Load original dataset
    with open(DATA_PATH, 'rb') as fh:
        dataset = pickle.load(fh)

    print(f'Source: {dataset["source"][0].shape}, {dataset["source"][1].shape}')
    print(f'Intermediate 1: {dataset["inter_1"][0].shape}, {dataset["inter_1"][1].shape}')
    print(f'Intermediate 2: {dataset["inter_2"][0].shape}, {dataset["inter_2"][1].shape}')
    print(f'Target: {dataset["target"][0].shape}, {dataset["target"][1].shape}\n')

    # Load the sampled data from fine-tuned diffusion models
    diff_data_1 = np.load(DIFF_PATH_1)['array'][...,None]
    diff_data_2 = np.load(DIFF_PATH_2)['array'][...,None]
    
    diff_dataset = dict(
        inter_1=diff_data_1,
        inter_2=diff_data_2
    )
    
    return dataset, diff_dataset

def exp_no_adapt(dataset, seed=DEFAULT_SEED):
    '''No adaptation baseline.'''

    # Initialize model
    model = get_model(seed=seed)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    
    # Train on source data
    X_source, Y_source = dataset['source']
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2, random_state=seed)

    datagen = ImageDataGenerator(rescale=1./255)
    datagen.fit(X_source_train)

    train_generator = datagen.flow(X_source_train, Y_source_train, **GENERATOR_CONFIG)
    val_generator = datagen.flow(X_source_val, Y_source_val, **GENERATOR_CONFIG)

    print('Training on the source data...')
    model_file_name = f'./models/no_adapt_{seed}.h5'
    history = model.fit(
                  train_generator, 
                  validation_data=val_generator,
                  epochs=150,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                             ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
              )
    model.load_weights(model_file_name) # Load best model
    
    # Evaluate on the held-out target data
    X_target, Y_target = dataset['target']
    N_target = X_target.shape[0]

    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)
    
    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test,
                                                                                    random_state=0) # NOTE: This should be fixed for all exps.
    target_test_generator = datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)

    print('Evaluating on the target test data...')
    final_loss, final_acc = model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    # Save the trained model
    model.save(f'./models/no_adapt_{seed}.h5')

    return model, final_acc

def exp_direct(dataset, seed=DEFAULT_SEED):
    '''Direct adaptation baseline.'''

    # Initialize model
    model = get_model(seed=seed)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    # Train on source data
    X_source, Y_source = dataset['source']
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2, random_state=seed)

    datagen = ImageDataGenerator(rescale=1./255)
    datagen.fit(X_source_train)

    train_generator = datagen.flow(X_source_train, Y_source_train, **GENERATOR_CONFIG)
    val_generator = datagen.flow(X_source_val, Y_source_val, **GENERATOR_CONFIG)

    print('Training on the source data...')
    model_file_name = f'./models/direct_{seed}.h5'
    history = model.fit(
                  train_generator, 
                  validation_data=val_generator,
                  epochs=150,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                             ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
              )
    model.load_weights(model_file_name)
    
    # Evaluate on the held-out target data
    X_target, Y_target = dataset['target']
    N_target = X_target.shape[0]

    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)
    
    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test,
                                                                                    random_state=0) # NOTE: This should be fixed for all exps.
    X_target_train, X_target_val, Y_target_train, Y_target_val = train_test_split(X_target_train, Y_target_train, test_size=N_target_val,
                                                                                  random_state=seed)
    
    # Generate pseudolabels for target training and validation data
    target_train_generator = datagen.flow(X_target_train, shuffle=False, **GENERATOR_CONFIG)
    target_val_generator = datagen.flow(X_target_val, shuffle=False, **GENERATOR_CONFIG)

    Y_target_train_pseudo = tf.keras.utils.to_categorical(np.argmax(model.predict(target_train_generator), axis=1), num_classes=2)
    Y_target_val_pseudo = tf.keras.utils.to_categorical(np.argmax(model.predict(target_val_generator), axis=1), num_classes=2)
    
    # Update generators with pseudolabels
    new_datagen = ImageDataGenerator(rescale=1./255)
    new_datagen.fit(X_target_train)
    target_train_generator = new_datagen.flow(X_target_train, Y_target_train_pseudo, **GENERATOR_CONFIG)
    target_val_generator = new_datagen.flow(X_target_val, Y_target_val_pseudo, **GENERATOR_CONFIG)

    # Train new model on pseudolabeled target data
    print('\nSelf-training with pseudolabeled target data...\n')
    new_model = get_model(seed=seed)
    new_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    model_file_name = f'./models/direct_{seed}.h5'
    new_history = new_model.fit(
                      target_train_generator, 
                      validation_data=target_val_generator,
                      epochs=150,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                                 ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
                  )
    new_model.load_weights(model_file_name)

    # Evaluate on target test data using ground-truth labels
    print('Evaluating on the target test data...')
    target_test_generator = datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)
    final_loss, final_acc = new_model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    # Save the trained model
    new_model.save(f'./models/direct_{seed}.h5')

    return new_model, final_acc

def exp_gradual(dataset, diff_dataset, n_samples=3000, seed=DEFAULT_SEED):
    '''Gradual adaptation approach, with two unlabeled intermediate domain data.'''

    # NOTE: n_samples is the number of randomly sampled data from each intermediate domain

    # Initialize model
    model = get_model(seed=seed)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    # Train on source data
    X_source, Y_source = dataset['source']
    X_source_train, X_source_val, Y_source_train, Y_source_val = train_test_split(X_source, Y_source, test_size=0.2, random_state=seed)

    datagen = ImageDataGenerator(rescale=1./255)
    datagen.fit(X_source_train)

    train_generator = datagen.flow(X_source_train, Y_source_train, **GENERATOR_CONFIG)
    val_generator = datagen.flow(X_source_val, Y_source_val, **GENERATOR_CONFIG)

    print('Training on the source data...')
    model_file_name = f'./models/gradual_{n_samples}_{seed}.h5'
    history = model.fit(
                  train_generator, 
                  validation_data=val_generator,
                  epochs=150,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                             ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
              )
    model.load_weights(model_file_name)
    
    # Randomly subsample a set number of diffusion model samples
    X_inter_1, Y_inter_1 = dataset['inter_1']

    if n_samples > 0:
        X2_inter_1 = diff_dataset['inter_1'] #  Diffusion model samples for first intermediate domain
        idxs = np.arange(0,X2_inter_1.shape[0])
        sampled_idxs, _ = train_test_split(idxs, train_size=n_samples, random_state=seed)
        X2_inter_1 = X2_inter_1[sampled_idxs] # Subsampled diffusion model samples

        # Append diffusion model samples to original data with dummy target labels
        X_inter_1 = np.concatenate([X_inter_1, X2_inter_1], axis=0)
        Y_inter_1 = np.concatenate([Y_inter_1, np.zeros((n_samples,2))], axis=0)
        print(f'Appended {n_samples} diffusion samples to first intermediate domain data.')
    
    print(f'X: {X_inter_1.shape}')
    print(f'Y: {Y_inter_1.shape}')

    # Get train-val split on first intermediate data
    X_inter_1_train, X_inter_1_val, Y_inter_1_train, Y_inter_1_val = train_test_split(X_inter_1, Y_inter_1, test_size=0.2, random_state=seed)

    # Generate pseudolabels for first intermediate data using source model
    inter_1_train_generator = datagen.flow(X_inter_1_train, shuffle=False, **GENERATOR_CONFIG)
    inter_1_val_generator = datagen.flow(X_inter_1_val, shuffle=False, **GENERATOR_CONFIG)

    Y_inter_1_train_pseudo = to_categorical(np.argmax(model.predict(inter_1_train_generator), axis=1), num_classes=2)
    Y_inter_1_val_pseudo = to_categorical(np.argmax(model.predict(inter_1_val_generator), axis=1), num_classes=2)

    # Update generators with pseudolabels
    inter_1_datagen = ImageDataGenerator(rescale=1./255)
    inter_1_datagen.fit(X_inter_1_train)
    inter_1_train_generator = inter_1_datagen.flow(X_inter_1_train, Y_inter_1_train_pseudo, **GENERATOR_CONFIG)
    inter_1_val_generator = inter_1_datagen.flow(X_inter_1_val, Y_inter_1_val_pseudo, **GENERATOR_CONFIG)

    # Train new model on pseudolabeled first intermediate data
    print('\nSelf-training with pseudolabeled first intermediate data...\n')
    inter_1_model = get_model()
    inter_1_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    model_file_name = f'./models/gradual_{n_samples}_{seed}.h5'
    inter_1_history = inter_1_model.fit(
                          inter_1_train_generator, 
                          validation_data=inter_1_val_generator,
                          epochs=150,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                                     ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
                      )
    inter_1_model.load_weights(model_file_name)
    
    # Randomly subsample a set number of diffusion model samples
    X_inter_2, Y_inter_2 = dataset['inter_2']

    if n_samples > 0:
        X2_inter_2 = diff_dataset['inter_2'] #  Diffusion model samples for first intermediate domain
        idxs = np.arange(0,X2_inter_2.shape[0])
        sampled_idxs, _ = train_test_split(idxs, train_size=n_samples, random_state=seed)
        X2_inter_2 = X2_inter_2[sampled_idxs] # Subsampled diffusion model samples

        # Append diffusion model samples to original data with dummy target labels
        X_inter_2 = np.concatenate([X_inter_2, X2_inter_2], axis=0)
        Y_inter_2 = np.concatenate([Y_inter_2, np.zeros((n_samples,2))], axis=0)
        print(f'Appended {n_samples} diffusion samples to second intermediate domain data.')
    
    print(f'X: {X_inter_2.shape}')
    print(f'Y: {Y_inter_2.shape}')

    # Get train-val split on second intermediate data
    X_inter_2_train, X_inter_2_val, Y_inter_2_train, Y_inter_2_val = train_test_split(X_inter_2, Y_inter_2, test_size=0.2, random_state=seed)

    # Generate pseudolabels for second intermediate data using first intermediate model
    inter_2_train_generator = datagen.flow(X_inter_2_train, shuffle=False, **GENERATOR_CONFIG)
    inter_2_val_generator = datagen.flow(X_inter_2_val, shuffle=False, **GENERATOR_CONFIG)

    Y_inter_2_train_pseudo = to_categorical(np.argmax(inter_1_model.predict(inter_2_train_generator), axis=1), num_classes=2)
    Y_inter_2_val_pseudo = to_categorical(np.argmax(inter_1_model.predict(inter_2_val_generator), axis=1), num_classes=2)

    # Update generators with pseudolabels
    inter_2_datagen = ImageDataGenerator(rescale=1./255)
    inter_2_datagen.fit(X_inter_2_train)
    inter_2_train_generator = inter_2_datagen.flow(X_inter_2_train, Y_inter_2_train_pseudo, **GENERATOR_CONFIG)
    inter_2_val_generator = inter_2_datagen.flow(X_inter_2_val, Y_inter_2_val_pseudo, **GENERATOR_CONFIG)

    # Train new model on pseudolabeled first intermediate data
    print('\nSelf-training with pseudolabeled second intermediate data...\n')
    inter_2_model = get_model()
    inter_2_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    model_file_name = f'./models/gradual_{n_samples}_{seed}.h5'
    inter_2_history = inter_2_model.fit(
                          inter_2_train_generator, 
                          validation_data=inter_2_val_generator,
                          epochs=150,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                                     ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
                      )
    inter_2_model.load_weights(model_file_name)
    
    # Get train-val-test split on target data
    X_target, Y_target = dataset['target']
    N_target = X_target.shape[0]
    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)

    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test,
                                                                                    random_state=0) # NOTE: Always use 0.
    X_target_train, X_target_val, Y_target_train, Y_target_val = train_test_split(X_target_train, Y_target_train, test_size=N_target_val)

    # Generate pseudolabels
    target_train_generator = inter_2_datagen.flow(X_target_train, shuffle=False, **GENERATOR_CONFIG)
    target_val_generator = inter_2_datagen.flow(X_target_val, shuffle=False, **GENERATOR_CONFIG)

    Y_target_train_pseudo = to_categorical(np.argmax(inter_2_model.predict(target_train_generator), axis=1), num_classes=2)
    Y_target_val_pseudo = to_categorical(np.argmax(inter_2_model.predict(target_val_generator), axis=1), num_classes=2)

    # Update generators with pseudolabels
    target_datagen = ImageDataGenerator(rescale=1./255)
    target_datagen.fit(X_target_train)
    target_train_generator = target_datagen.flow(X_target_train, Y_target_train_pseudo, **GENERATOR_CONFIG)
    target_val_generator = target_datagen.flow(X_target_val, Y_target_val_pseudo, **GENERATOR_CONFIG)

    # Train new model on pseudolabeled target data
    print('\nSelf-training with pseudolabeled target data...\n')
    target_model = get_model()
    target_model.compile(
        optimizer=Adam(learning_rate=1e-3), 
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy()]
    )

    model_file_name = f'./models/gradual_{n_samples}_{seed}.h5'
    target_history = target_model.fit(
                         target_train_generator, 
                         validation_data=target_val_generator,
                         epochs=150,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                                    ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
                     )
    target_model.load_weights(model_file_name)
    
    # Evaluate on target test data using ground-truth labels
    print('Evaluating on the target test data...')
    target_test_generator = target_datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)
    final_loss, final_acc = target_model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    # Save the trained model
    target_model.save(f'./models/gradual_{n_samples}_{seed}.h5')

    return target_model, final_acc

def exp_oracle(dataset, seed=DEFAULT_SEED):
    '''No adaptation baseline.'''

    # Initialize model
    model = get_model(seed=seed)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    
    # Train on target data
    X_target, Y_target = dataset['target']
    N_target = X_target.shape[0]

    N_target_train = int(0.6 * N_target)
    N_target_val = int(0.2 * N_target)
    N_target_test = int(0.2 * N_target)

    X_target_train, X_target_test, Y_target_train, Y_target_test = train_test_split(X_target, Y_target, test_size=N_target_test,
                                                                                    random_state=0) # NOTE: This should be fixed for all exps.
    X_target_train, X_target_val, Y_target_train, Y_target_val = train_test_split(X_target_train, Y_target_train, test_size=N_target_val,
                                                                                  random_state=seed)
    datagen = ImageDataGenerator(rescale=1./255)
    datagen.fit(X_target_train)

    train_generator = datagen.flow(X_target_train, Y_target_train, **GENERATOR_CONFIG)
    val_generator = datagen.flow(X_target_val, Y_target_val, **GENERATOR_CONFIG)
    target_test_generator = datagen.flow(X_target_test, Y_target_test, batch_size=N_target_test, shuffle=False)

    print('Training on the target data...')
    model_file_name = f'./oracle_{seed}.h5'
    history = model.fit(
                  train_generator, 
                  validation_data=val_generator,
                  epochs=150,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                             ModelCheckpoint(model_file_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]
              )
    model.load_weights(model_file_name)

    print('Evaluating on the target test data...')
    final_loss, final_acc = model.evaluate(target_test_generator)
    print(f'\nAccuracy on Target Test Data: {final_acc}')

    # Save the trained model
    model.save(f'./models/oracle_{seed}.h5')

    return model, final_acc

def main(**kwargs):
    #exp = kwargs['exp']
    seeds = kwargs['seeds']

    if not osp.exists('./models'):
        os.mkdir('./models')

    # Check if GPUs are detected
    print(tf.config.list_physical_devices('GPU'))

    # Load the dataset
    dataset, diff_dataset = load_data()

    model_names = [
        'no-adapt',
        'direct',
        'gradual_0',
        'gradual_3000',
        'gradual_6000',
        'gradual_9000',
        'gradual_12000',
        'oracle'
    ]
    acc_dict = defaultdict(list)
    start_time = datetime.now()

    for model_name in model_names:
        model = model_name.split('_')[0]

        if model == 'no-adapt':
            exp_f = exp_no_adapt
            print('Training the "No Adaptation" baseline.')

        elif model == 'direct':
            exp_f = exp_direct
            print('Training the "Direct Adaptation" baseline.')

        elif model == 'gradual':
            exp_f = exp_gradual
            print('Training the "Gradual Adaptation" model.')

        else:
            exp_f = exp_oracle
            print('Training the "Oracle" model.')

        for i, seed in enumerate(seeds):
            print(f'[{i+1}] Manual Seed = {seed}')
            if model == 'gradual':
                n_samples = int(model_name.split('_')[1])
                loss, acc = exp_f(dataset, diff_dataset, n_samples=n_samples, seed=seed)
            else:
                loss, acc = exp_f(dataset, seed=seed)

            acc_dict[model_name].append(acc)

    with open('./adapt_results.pkl', 'wb') as fh:
        pickle.dump(acc_dict, fh)

    for model_name in model_names:
        print(f'[{model_name}] Mean={np.mean(acc_dict[model_name])}, Std={np.std(acc_dict[model_name])}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--exp', help='Choice of experiment to run', default='gradual', type=str)
    parser.add_argument('--seeds', help='Random seeds', default=[DEFAULT_SEED], nargs='*', type=int)
    args = parser.parse_args()

    main(**vars(args))