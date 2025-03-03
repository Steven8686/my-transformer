from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformer import TransformerDecoderModel
from my_transformer import Decoder
import torch.nn as nn
import torch.optim as optim
import torch
import jieba
import json
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Hyper parameters
num_decoder_layer = 6
num_attn_head = 8
batch_size = 64
learning_rate = 0.0002
embed_size = 128
max_token = 50
num_epochs = 10

# Lucky Number
torch.manual_seed(1020)

class TextDataset(Dataset):
    """
        Rewrite Dataset to load and preprocess the vocabulary list, turing them into dicts and tensors.
    """
    def __init__(self, filepath):
        if len(filepath) == 1:
            # Generate all the information based on a file with texts.
            print("Generating from original texts...")
            filepath = filepath[0]
            words = []

            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    # Use jieba to split the tokens.
                    words.extend(list(jieba.cut(line.strip())))

            # Deduplicate and save word list to cache.
            self.vocab = list(set(words))
            with open('data/vocab.json', 'w', encoding="utf-8") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=4)
            with open('data/words.json', 'w', encoding="utf-8") as f:
                json.dump(words, f, ensure_ascii=False, indent=4)
            self.vocab_size = len(self.vocab)

            # Generate mapping between tokens and integer. Save to cache.
            self.word_to_int = {word: i for i, word in enumerate(self.vocab)}
            self.int_to_word = {i: word for i, word in enumerate(self.vocab)}
            with open('data/word_to_int.json', 'w', encoding="utf-8") as f:
                json.dump(self.word_to_int, f, ensure_ascii=False, indent=4)
            with open('data/int_to_word.json', 'w', encoding="utf-8") as f:
                json.dump(self.int_to_word, f, ensure_ascii=False, indent=4)
            self.data = [self.word_to_int[word] for word in words]
            print("Generate complete. Caches saved to ./data.")

        if len(filepath) == 4:
            # Load the vocabs and mapping from caches.
            print("Loading words from cache...")
            (w2i_path, i2w_path, vocab_path, words_path) = filepath
            with open(w2i_path, 'r', encoding='utf-8') as f:
                self.word_to_int = json.load(f)
            with open(i2w_path, 'r', encoding='utf-8') as f:
                self.int_to_word = json.load(f)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            with open(words_path, 'r', encoding='utf-8') as f:
                words = json.load(f)
            self.vocab_size = len(self.vocab)
            self.data = [self.word_to_int[word] for word in words]
            print("Loading words succeeded")

    def __len__(self):
        # Returns length-1: Need to predict next word.
        return len(self.data) - 1

    def __getitem__(self, idx):
        # Extract [idx-max_token:idx] as input, [idx] as output
        input_seq = torch.tensor(self.data[max(0, idx - max_token):idx], dtype=torch.long)
        target = torch.tensor(self.data[idx], dtype=torch.long)
        return input_seq, target


def collate_fn(batch):
    """
         Generate padding masks when data is loaded rather than generate them in model
         Args:
            batch: Tensor. The original input tensor.
         Return:
            (padded_sequences, padded_masks, labels_tensor): Tuple. The masked sequence and masks, label tensor. Should
            be unpacked in every batch.
    """

    sequences, labels = zip(*batch)
    masks = []
    for seq in sequences:
        mask = torch.ones(seq.shape[0], dtype=torch.bool) if seq.shape[0] > 0 else torch.zeros(1, dtype=torch.bool)
        masks.append(mask)

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=False)
    labels_tensor = torch.tensor(labels)

    return padded_sequences, padded_masks, labels_tensor


# filepath = "./texts/sentence.txt"
filepath = ("./data/word_to_int.json", "./data/int_to_word.json", "./data/vocab.json", "./data/words.json")
dataset = TextDataset(filepath)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
# The model definition below uses the standard decoder by pytorch
# model = TransformerDecoderModel(vocab_size=dataset.vocab_size, embed_size=128, num_heads=8, hidden_dim=1024, num_layers=6)
# The below one is mine.
model = Decoder(vocab_size=dataset.vocab_size, embed_size=embed_size, num_layers=num_decoder_layer, num_heads=num_attn_head, seq_length=max_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("Start training...")
# Here, used auto precision to save resources.
scaler = GradScaler()

# Start training
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    last_time = time.time()
    for i, (inputs, mask, targets) in enumerate(dataloader):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs, mask)
            loss = criterion(outputs[:, -1, :], targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if i % 100 == 0 and i != 0:
            usetime = time.time()-last_time
            last_time = time.time()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item()}, time: {usetime}')

    print("Epoch", epoch, ":\n", time.time()-epoch_start_time)

# Save model
model_path = "./model/model"
torch.save(model, model_path)
print('Model saved to ', model_path)


