from keras.models import load_model
import csv
import numpy as np
import time
from datetime import datetime, timedelta

def encode(sentence, maxlen):
    num_features = len(chars) + 1  # We only need the characters
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c == char:
                X[0, t + leftpad, char_indices[c]] = 1
    return X

def get_symbol(predictions):
    maxPrediction = 0
    symbol = ''
    for i, prediction in enumerate(predictions):
        if prediction >= maxPrediction:
            maxPrediction = prediction
            symbol = target_indices_char[i]
    return symbol

def make_predictions(model, lines, maxlen, chars, char_indices, indices_char, target_indices_char):
    with open('output_files/results/next_activity_%s' % eventlog, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["Actual Label", "Predicted Label"])
        
        for line in lines:
            line = line + '!'  # Add a termination symbol
            actual_label = line[-1]  # The last character is the actual label
            cropped_line = line[:-1]  # Remove the termination symbol for prediction
            
            for i in range(predict_size):
                enc = encode(cropped_line, maxlen)
                y = model.predict(enc, verbose=0)
                y_char = y[0][0]
                prediction = get_symbol(y_char)
                cropped_line += prediction
                
                if prediction == '!':
                    print('! predicted, end case')
                    break
                
                if i == 0:
                    predicted_label = prediction
                    spamwriter.writerow([actual_label, predicted_label])

# Load model
model = load_model('old files/output_files\models\model_26-1.29.h5')

# Read the data
eventlog = 'event_data_aggregated_alex2_reworked.csv'
csvfile = open('data/%s' % eventlog, 'r')
datareader = csv.reader(csvfile, delimiter=',', quotechar='|')

# Helper variables
lines = []

next(datareader, None)  # skip the headers
for row in datareader:
    lines.append(str(row[1]))  # Add the activity label to the lines list

# Unique characters
chars = list(set().union(*lines))
chars.sort()

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_indices_char = dict((i, c) for i, c in enumerate(chars))

# Set parameters
maxlen = max(map(lambda x: len(x), lines))
predict_size = 1

# Make predictions
make_predictions(model, lines, maxlen, chars, char_indices, indices_char, target_indices_char)
