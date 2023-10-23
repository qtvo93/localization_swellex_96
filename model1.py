# !pip install transformers
# !pip install torchlibrosa
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import librosa
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import librosa.display

from torchlibrosa.augmentation import SpecAugmentation
import torchaudio.transforms as T

mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=63,
            mel_scale="htk",)

spec_augmenter = SpecAugmentation(
                    time_drop_width=2,
                    time_stripes_num=2,
                    freq_drop_width=2,
                    freq_stripes_num=2)

data_array = np.loadtxt('data.txt', delimiter='\t')

# Load SproulToVLA.txt data
sproul_data = pd.read_csv('SproulToVLA.S5.txt', sep='\t')

# Rename the columns to remove any leading or trailing spaces
sproul_data.columns = sproul_data.columns.str.strip()

print(sproul_data.columns)

# Split the "Jday Time Duration Range(km)" column into separate columns
sproul_data[['Jday', 'Time', 'Duration', 'Range(km)']] = sproul_data['Jday Time Duration Range(km)'].str.split(expand=True)

# Convert 'Duration' and 'Range(km)' columns to appropriate data types
sproul_data['Duration'] = sproul_data['Duration'].astype(int)
sproul_data['Range(km)'] = sproul_data['Range(km)'].astype(float)

# Drop the original combined column
sproul_data.drop(columns=['Jday Time Duration Range(km)'], inplace=True)

# Display the first few rows of the DataFrame
print(sproul_data.head())

# Calculate spectrogram duration in seconds
spectrogram_duration = 1.0

# Calculate the number of spectrograms
num_spectrograms = int(len(data_array) / (1500 * spectrogram_duration))

# Create a new metadata DataFrame
# metadata = pd.DataFrame(columns=['filename', 'range_km'])
metadata2 = pd.DataFrame(columns=['filename', 'fold',	'target',	'range_km',	'esc',	'src_file',	'take'])


# Function to determine the target class based on range
def determine_target_class(range_value):
    if range_value < 2.0:
        return 0
    elif range_value < 4.0:
        return 1
    elif range_value < 6.0:
        return 2
    elif range_value < 8.0:
        return 3
    else:
        return 4

# Calculate the fold for each data point
fold_values = np.repeat(np.arange(1, 7), np.ceil(num_spectrograms / 6))[:num_spectrograms]

target_dict = {}
target_class_int = 0
# Iterate through each spectrogram
for i in range(num_spectrograms):
    # Calculate the timestamp for the spectrogram
    timestamp = i * spectrogram_duration

    # Convert the Duration column to seconds for comparison
    sproul_data['Duration_seconds'] = sproul_data['Duration'] * 60

    # Adjust the column name by adding a space before 'Duration'
    closest_idx = np.argmin(np.abs(sproul_data['Duration_seconds'] - timestamp))

    # Get the corresponding Range(km) value
    range_km = sproul_data.loc[closest_idx, 'Range(km)']

    filename = f'file_{i+1}.wav'


    ####
    fold = fold_values[i]
    # try:
    #     target = target_dict[float(range_km)]
    # except:
    #     target_dict[float(range_km)] = target_class_int
    #     target = target_dict[float(range_km)]
    #     target_class_int += 1
    target = determine_target_class(float(range_km))
    # category = f'class_{target}'  # Assign class category based on target
    esc = 'FALSE'
    src_file = filename
    take = i+1

    
    # metadata = metadata.append({'filename': filename, 'range_km': range_km}, ignore_index=True)
    metadata2 = metadata2.append({'filename': filename, 'fold': fold,'target': target,'range_km': range_km,'esc':esc,'src_file': src_file,'take': take}, ignore_index=True)

print("metadata generated")
print(target_dict)
print(len(target_dict))







scaler = StandardScaler()

# spectrogram
# Parameters for the mel spectrogram
n_fft = 1024  # FFT window size
hop_length = 72  # Hop size for spectrogram frames
n_mels = 223  # Number of mel bands
sr = 16000  # Sampling rate
num_channels = 21


from sklearn.utils import shuffle
# metadata2 = shuffle(metadata2)
# Create a dictionary from xdata and labels
data_dict = {}
num_slices = len(metadata2)
for i in range(num_slices):
    filename = metadata2.iloc[i]['filename']
    range_km = metadata2.iloc[i]['range_km']
    start_idx = i * 1500
    end_idx = min((i + 1) * 1500, data_array.shape[0])  # Adjust for the last slice
    data_dict[filename] = {'data': data_array[start_idx:end_idx], 'target': range_km}

metadata2 = shuffle(metadata2)

output_dict = [[] for _ in range(6)]
audio_list = set(metadata2['filename'])
for index, row in metadata2.iterrows():
    name = row['filename']
    fold = row['fold']
    target = row['target']
    if name in audio_list:
        signal, target_value = data_dict[name]['data'], data_dict[name]['target']
        # Resample the signal to 16000 Hz
        resampled_signal = librosa.resample(signal.T, orig_sr=1500, target_sr=sr)
        # Initialize an empty array to store mel spectrograms for all channels
        mel_spectrograms = []

        # Iterate through each channel
        # for channel_data_np in signal.T:  # Transpose the data to iterate through channels
        i = 0
        for channel_data_np in resampled_signal:  

            # Compute mel spectrogram for the current channel
            # mel_spec = librosa.feature.melspectrogram(y=channel_data, sr=sr, n_fft=n_fft , n_mels=n_mels, hop_length=hop_length)
            # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
            channel_data = torch.tensor(channel_data_np, dtype=torch.float32)
            # S = torch.log10(mel_spectrogram(channel_data + 1e-10))
            # S = librosa.power_to_db(S, ref=np.max)
            # Compute the FFT to get the spectrum
            spectrum = torch.fft.fft(channel_data)
            S = torch.abs(spectrum)
            
            if index == 0:
                # Plot the spectrum
                plt.figure(figsize=(10, 4))
                plt.imshow(S.numpy()[None, :], cmap='viridis', aspect='auto')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrum - Channel {i+1}')
                plt.xlabel('Frequency Bin')
                plt.ylabel('Magnitude')
                plt.tight_layout()

                # plt.figure(figsize=(10, 4))
                # plt.imshow(S, cmap='viridis', origin='lower', aspect='auto')
                # plt.colorbar(format='%+2.0f dB')
                # # librosa.display.specshow(S.squeeze(), x_axis='time', y_axis='mel', sr=16000, hop_length=256)
                # # plt.colorbar(format='%+2.0f dB')
                # plt.title(f'Spectrogram - Channel {i+1}')
                # plt.xlabel('Time')
                # plt.ylabel('Mel Frequency')
                # plt.tight_layout()
                
                # Save the plot as an image
                plt.savefig(f'spec_{i+1}.png')
                i += 1

                # Optional: Show the plot (uncomment the line below if you want to display the plots)
                # plt.show()

                # Close the current plot to avoid overlapping in the next iteration
                plt.close()

            mel_spectrograms.append(S)

        # Stack the mel spectrograms along the first axis to get a (21, n_mels, width) array
        mel_spectrograms = np.stack(mel_spectrograms, axis=0)



        output_dict[int(fold) - 1].append(
            {
                "name": name,
                "target": target,
                "waveform": np.float32(mel_spectrograms)
            }
        )
        if index == 0:
            print(f'Processing {name}, Fold: {fold}, Target: {target_value}')
            print("waveform: ", np.float32(mel_spectrograms))
            print("shape:", mel_spectrograms.shape)
            print("target: ", target)
        # for quick 10 samples testing
        # if index == 10:
        #   break

print(len(output_dict[0])+len(output_dict[1])+len(output_dict[2])+len(output_dict[3])+len(output_dict[4])+len(output_dict[5]))
# np.save('swellex-data.npy', output_dict)

print("-------------Success-------------")

# Split data into training (folds 1 to 5) and testing (fold 6)
# train_data = []
# test_data = []
# for fold in range(5):
#     train_data.extend(output_dict[fold])
# test_data.extend(output_dict[5])

# if quick test

# datata = []
# for fold in range(6):
#     try:
#       datata.extend(output_dict[fold])
#     except:
#       pass

# train_data = datata[:-2]
# test_data = datata[-2:]

# endif quick test

ori_data = []
for fold in range(6):
    ori_data.extend(output_dict[fold])

ori_data = shuffle(ori_data)
split_ratio = 0.8  
split_index = int(len(ori_data) * split_ratio)

train_data = ori_data[:split_index]
test_data = ori_data[split_index:]
# Convert to numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)

# train_metadata = metadata2[metadata2['fold'] < 6]
# test_metadata = metadata2[metadata2['fold'] == 6]


class CustomDataset(Dataset):
    def __init__(self, data):
        # self.metadata = metadata
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data = self.data[idx] # Load preprocessed data from swellex-data.npy
        # target = self.metadata.iloc[idx]['target']
        data = self.data[idx]["waveform"]
        target = self.data[idx]["target"]
        return data, target

train_dataset = CustomDataset(data=train_data)
test_dataset = CustomDataset( data=test_data)
print(len(train_dataset), len(test_dataset))
print(train_dataset[0], test_dataset[0])
# train_dataset = TensorDataset(train_dataset)
# test_dataset = TensorDataset(test_dataset)


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from transformers import ViTConfig, ViTModel
# from torch.autograd import Variable

# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig(num_channels=21, image_size=63,num_labels=5)

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)

# Accessing the model configuration
print(model)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 1200
num_classes = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch, data in enumerate(train_loader):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        # print("****================================***")

        # inputs = spec_augmenter(inputs)

        optimizer.zero_grad()
        base_output = model(inputs)
        dropout = nn.Dropout(0.1)
        classifier = nn.Linear(768, num_classes).to(device)

        # print("****================================***")

        # print(pooler_output)
        # print(base_output.last_hidden_state[:,0])
        # print("****================================***")
        # print(base_output.last_hidden_state)
        pooler_output = dropout(base_output.last_hidden_state[:,0])
        logits = classifier(pooler_output)
        # print("****================================***")
        # print(logits.shape)
        # print(logits)
        # print("****================================***")

        # loss = criterion(logits.view(-1, num_classes), labels.view(-1))
        loss = criterion(logits, labels)
        # pooler_output = pooler_output.detach().cpu().numpy()  # Convert to a NumPy array

        # print(torch.tensor(pooler_output, dtype=torch.float32))
        # # Apply the linear layer

        # logits = classifier(torch.tensor(pooler_output, dtype=torch.float32))  # logits shape: (batch_size, 21, num_classes)

        # # Apply softmax to get class probabilities
        # probs = F.softmax(logits, dim=2)  # probs shape: (batch_size, 21, num_classes)


        # predicted_classes = torch.argmax(probs, dim=2)  # predicted_classes shape: (batch_size, 21)

        # loss = criterion(predicted_classes.clone().detach().requires_grad_(True), labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

# Set the model in evaluation mode
model.eval()

# Lists to store predictions and targets
all_predictions = []
all_targets = []

# Iterate over batches in the test dataset
with torch.no_grad():
    for batch, data in enumerate(test_loader):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass through the model
        base_output = model(inputs)
        classifier = nn.Linear(768, num_classes).to(device)

        # print(pooler_output)
        pooler_output = base_output.last_hidden_state[:,0]
        # pooler_output = pooler_output.detach().cpu().numpy()  # Convert to a NumPy array
        logits = classifier(pooler_output)
        # print(logits.shape)
        # print(torch.tensor(pooler_output, dtype=torch.float32))
        # # Apply the linear layer
        # logits = classifier(torch.tensor(pooler_output, dtype=torch.float32))  # logits shape: (batch_size, 21, num_classes)

        # Apply softmax to get class probabilities
        # probs = F.softmax(logits, dim=2)  # probs shape: (batch_size, 21, num_classes)
        # print(probs)
        # If get the most likely class for each channel,
        predicted_labels = torch.argmax(logits, dim=1)
        # print(predicted_labels.shape)
        # print(predicted_labels)

        # # Convert logits to predicted class labels
        # _, predicted_labels = torch.max(outputs, dim=1)

        # Convert tensors to numpy arrays
        predicted_labels = predicted_labels.cpu().numpy()
        batch_targets = labels.cpu().numpy()

        # Append batch predictions and targets to the lists
        all_predictions.extend(predicted_labels)
        all_targets.extend(batch_targets)

# Convert lists to numpy arrays for easier manipulation
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
print(all_predictions[:10], all_targets[:10])
# tolerance = 0.05
# accurate_predictions = np.abs(all_predictions - all_targets) < tolerance
# accuracy = np.mean(accurate_predictions)
# Calculate accuracy
accuracy = (all_predictions == all_targets).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")



# Assuming true_labels and predicted_labels as numpy arrays with shape (4500, )
# Each element corresponds to the label for a spectrogram

# Create an array of indices for spectrograms (datapoint numbers)
indices = np.arange(1, len(all_targets) + 1)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(20, 6))

# Plot the true labels in blue
ax.scatter(indices, all_targets, c='b', label='True Labels')

# Plot the predicted labels in red (assuming predicted_labels)
ax.scatter(indices, all_predictions, c='r', label='Predicted Labels')

# Add labels and legend
ax.set_xlabel('Spectrogram Index')
ax.set_ylabel('Labels')
ax.set_title('True vs. Predicted Labels for Spectrograms')
ax.legend()

plt.savefig('testing_ranges.png')
plt.show()






"""
from torchlibrosa.augmentation import SpecAugmentation
import torchaudio.transforms as T

mel_spectrogram = T.MelSpectrogram(
            sample_rate=44100,
            n_fft=4096,
            win_length=4096,
            hop_length=1024,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=256,
            mel_scale="htk",)

spec_augmenter = SpecAugmentation(
                    time_drop_width=2,
                    time_stripes_num=2,
                    freq_drop_width=2,
                    freq_stripes_num=2)



inputs = spec_augmenter(inputs)
"""