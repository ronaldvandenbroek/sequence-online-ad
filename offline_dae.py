import os
import time
import pandas as pd
import numpy as np

from keras.layers import Dense, Dropout, Input, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam

from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from tqdm import tqdm
from datetime import datetime

np.set_printoptions(suppress=True)

# Parameters for for the autoencoder
# NOTE: DAE is configured according to implementation of BPAD
hidden_layers=2
hidden_size_factor=0.1 #0.5
noise=None
dropout=0.5

beta_2=0.99
learning_rate=0.0001

# Parameters for the training
epochs = 100
batch_size = 128

# Parameters for Word2Vec encoding
vector_size = 100
window = 5
min_count = 1

# dir_datasets = 'data/'
dir_datasets = 'data-hospital/'
dir_results = 'results/dae'

def model_fn(nr_features, hidden_layers=2, hidden_size_factor=0.5, noise=None, dropout=0.5, learning_rate=0.0001, beta_2=0.99):
    '''
    Create the DAE model
    '''
    input_ = Input(shape=(nr_features,), name='input')
    x = input_

    if noise is not None:
        x = GaussianNoise(noise)(x) 

    for i in range(hidden_layers):
        if isinstance(hidden_size_factor, list):
            factor = hidden_size_factor[i]
        else:
            factor = hidden_size_factor
        x = Dense(int(nr_features * factor), activation='relu', name=f'hid{i + 1}')(x)
        x = Dropout(dropout)(x)

    output = Dense(nr_features, activation='tanh', name='output')(x)

    model = Model(inputs=input_, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, beta_2=beta_2),
        loss='mean_squared_error',
    )

    return model

def encoding_fn(cases, size, window, min_count):
    model = Word2Vec(
                vector_size=size,
                window=window,
                min_count=min_count)
    sentences = []
    for group in cases:
        group_sentences = []
        for row in group:
            row_sentences = [str(item) for item in row]
            group_sentences.extend(row_sentences)
        sentences.append(group_sentences)
    
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=10)
    return model

def embed_prefix(events, model:Word2Vec, vector_size, input_length_model):
    vectors = []
    for event in events:
        case_vector = []
        for token in event:
            # Do not embed padding
            if token != 0: 
                try:
                    case_vector.append(model.wv[token])
                except KeyError as e:
                    print("Token not found:", e)
                    pass
        if len(case_vector) != 0:
            embedded_event = np.array(case_vector).mean(axis=0)
            vectors.append(embedded_event)

    embedded_sequence = np.array(vectors)
    embedded_sequence = np.reshape(embedded_sequence, (vector_size * len(vectors)))

    padded_vector = np.pad(embedded_sequence, (0, input_length_model - embedded_sequence.shape[0]), 'constant', constant_values=0)
    encoded_sequence = np.expand_dims(padded_vector, axis=0)
    return encoded_sequence

def cal_best_PRF(y_true,probas_pred):
    '''
    计算在任何阈值下，最好的precision，recall。f1
    :param y_true:
    :param probas_pred:
    :return:
    '''
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, probas_pred)

    epsilon = 1e-10  # Small constant to avoid division by zero
    f1s=(2*precisions*recalls)/(precisions+recalls+epsilon)
    f1s[np.isnan(f1s)] = 0

    best_index=np.argmax(f1s)

    aupr = average_precision_score(y_true, probas_pred)

    # Alternative F1-scoring
    # Find best F1 index, ensuring recall is at least 50%
    alt_best_index = np.argmax(f1s * (recalls >= 0.5))  

    # Compute best binary F1-score using the threshold
    best_threshold = thresholds[alt_best_index]
    y_pred = (probas_pred >= best_threshold).astype(int)
    binary_f1 = f1_score(y_true, y_pred)

    return precisions[best_index],recalls[best_index],f1s[best_index],aupr,binary_f1

datasets = [dir_datasets + '/' + dataset for dataset in os.listdir(dir_datasets)]
for dataset in datasets:
    # Load the dataset
    file_name = os.path.basename(dataset).removesuffix('.csv')
    print(f"Processing {file_name}")
    df_dataset = pd.read_csv(dataset)
    df_dataset.sort_values('timestamp', inplace=True)

    print(f'Containing {len(df_dataset)} events')
    # NOTE: Timestamps are not used in this implementation as no time-based limited memory is used
    # Fix the timestamp for real datasets
    # try:
    #     df_dataset['timestamp'] = pd.to_datetime(df_dataset['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # except:
    #     # NOTE: Real world datasets do not seem to include a date, seems that this is done incorrectly in OAE as timestamps cant be ordered
    #     print('Invalid timestamp format')
    #     reference_date = pd.Timestamp("2025-01-01")
    #     df_dataset['timestamp'] = reference_date + pd.to_timedelta('00:' + df_dataset['timestamp'])
    #     df_dataset['timestamp'] = pd.to_datetime(df_dataset['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the data
    max_length = df_dataset.groupby('case_id').size().max()
    input_dim = vector_size * max_length
    cases = df_dataset.groupby(df_dataset.columns[0]).apply(lambda x: x.iloc[:, ~df_dataset.columns.isin(['case_id', 'event_position', 'timestamp', 'isAnomaly', 'anomaly'])].values.tolist())
    cases_result = df_dataset.groupby(df_dataset.columns[0]).apply(lambda x: x.iloc[:, df_dataset.columns.isin(['isAnomaly'])].values.tolist())
    print(f'Containing {len(cases)} cases with a maximum length of {max_length}')

    # Build the models
    word2vec = encoding_fn(cases, vector_size, window, min_count)
    autoencoder = model_fn(input_dim, hidden_layers, hidden_size_factor, noise, dropout, learning_rate, beta_2)

    # Encode the prefixes
    start_time_encoding = time.time()
    prefixes = []
    prefixes_target = []
    prefixes_mask = []
    for sequence, outcome in tqdm(zip(cases, cases_result), total=len(cases), desc=f"Encoding prefixes"):
        for i in range(1, len(sequence) + 1):
            # Embed the prefix
            prefix = sequence[:i]
            encoded_prefix = embed_prefix(prefix, word2vec, vector_size, input_dim)

            # Get the ground truth anomaly for the prefix
            prefix_target = np.max(outcome[:i])

            # Create the mask for the prefix
            prefix_mask = np.zeros(input_dim)
            prefix_mask[:vector_size * i] = 1

            # Save the encoded prefix
            prefixes.append(encoded_prefix)
            prefixes_target.append(prefix_target)
            prefixes_mask.append(prefix_mask)

            end_time_encoding = time.time()

    prefixes = np.vstack(prefixes)
    prefixes_target = np.vstack(prefixes_target)
    prefixes_mask = np.vstack(prefixes_mask)

    # Train and predict
    # NOTE: Offline DAE model is able to look into the future because it is trained on the full selection of prefixes
    # NOTE: Offline DAE model is able to train far faster than the online OAE model as it is able to use batches (in theory also possible to use with OAE however this is seen as future work)
    # NOTE: Offline DAE model is able to train over multiple epochs, which is the main reason for the slowdown in total runtime, however also greatly improves performance
    # NOTE: Offline DAE model is not restricted to the memory of the online OAE model
    # NOTE: All these reasonings also hold for the encoding of the prefixes
    start_time_label = time.time()
    autoencoder.fit(prefixes, prefixes, epochs=epochs, batch_size=batch_size, shuffle=True)
    start_time_scoring = time.time()
    predictions = autoencoder.predict(prefixes)

    # Calculate the reconstruction error
    errors = np.power(prefixes - predictions, 2)

    # Mask the errors to not count padding
    errors_masked = errors * prefixes_mask
    non_zero_counts = np.count_nonzero(errors_masked, axis=1)
    non_zero_counts[non_zero_counts == 0] = 1

    # Calculate the prefix erros
    error_sums = np.sum(errors_masked, axis=1)
    reconstruction_errors = error_sums / non_zero_counts
    # print(prefixes.shape, prefixes_target.shape, prefixes_mask.shape, predictions.shape, reconstruction_errors.shape)
    
    # Calculate the best F1-score
    prefix_p, prefix_r, prefix_f1, _, prefix_binary_f1 = cal_best_PRF(prefixes_target, reconstruction_errors)
    print(f'F1-score: {prefix_f1:2f}, Binary F1-score: {prefix_binary_f1:2f}, (precision: {prefix_p:2f}, recall: {prefix_r:2f})')

    end_time_label = time.time()
    encoding_duration = start_time_label - start_time_encoding
    prediction_duration = start_time_scoring - start_time_label
    scoring_duration = end_time_label - start_time_scoring
    total_duration = end_time_label - start_time_encoding
    num_prefixes = len(prefixes)
    encoding_duration_per_event = encoding_duration / num_prefixes
    prediction_duration_per_event = prediction_duration / num_prefixes
    scoring_duration_per_event = scoring_duration / num_prefixes
    total_duration_per_event = total_duration / num_prefixes
    print(f"Encoding: {encoding_duration:.2f}s, Prediction: {prediction_duration:.2f}s, Scoring: {scoring_duration:.2f}s, Total: {total_duration:.2f}s")
    print(f"Encoding Event: {encoding_duration_per_event:.8f}s, Prediction Event: {prediction_duration_per_event:.8f}s, Scoring Event: {scoring_duration_per_event:.8f}s, Total Event: {total_duration_per_event:.8f}s")

    # Saving the results
    saving_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_filename = f'{dir_results}/results_dae_{file_name}.csv'
    raw_results_filename = f'{dir_results}/raw/results_dae_{file_name}_{saving_timestamp}.csv'
    with open(results_filename, "a+") as csvfile:
        csvfile.write(f"{prefix_f1:.5f},{prefix_binary_f1:.5f},{encoding_duration:.5f},{prediction_duration:.5f},{scoring_duration:.5f},{total_duration:.5f},{encoding_duration_per_event:.5f},{prediction_duration_per_event:.5f},{scoring_duration_per_event:.5f},{total_duration_per_event:.5f},\n")    
    with open(raw_results_filename, "a+") as csvfile:
        for i in range(len(reconstruction_errors)):
            csvfile.write(f"{reconstruction_errors[i]:.5f},{prefixes_target[i][0]}\n")
    