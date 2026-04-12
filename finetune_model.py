import json
import numpy as np
import pandas as pd
from custom_modules import dataparser
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, roc_auc_score, log_loss, confusion_matrix, classification_report
from momentfm import MOMENTPipeline
from tqdm import tqdm

# HYPERPARAMETERS
batch_size = 32
max_epochs = 10
learning_rate = 1e-4
train_split = 0.8
val_split = 0.1

# LOAD GLOBALS
with open("env.json", "r") as file:
    globalVars = json.load(file)
yearNow, instrument, granularity, forecastHorizon = globalVars.values()

# SETUP
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LOAD DATA
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-04-01.json")
df = dataparser.addTarget(df, forecastHorizon, 10)

# SELECT FEATURES
featureList = [
    "open_return", "high_return", "low_return", "close_return", "vol_return", "smooth_return", "dist_smooth",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "oc_spread", "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "dist_ema100", "ema_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum", "adx_direction",
    "dist_high", "dist_low"
]

# SPLIT DATA
X = df[featureList]
y = df["target"]
trainEndIdx = int(train_split * len(X))
valEndIdx = int((train_split + val_split) * len(X))

X_train = X[:trainEndIdx]
X_val = X[trainEndIdx:valEndIdx]
X_test = X[valEndIdx:]
y_train = y[:trainEndIdx].values
y_val = y[trainEndIdx:valEndIdx].values
y_test = y[valEndIdx:].values

# SCALE DATA (becomes numpy array after scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# DATASET CLASS
class ForexWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = features
        self.y = labels
 
    def __len__(self):
        return len(self.X) - 511 # last valid idx = len(X) - 512
 
    def __getitem__(self, idx):
        # Window end (inclusive) and start
        x_win = self.X[idx : idx + 512] # [512, n_features]
        label = self.y[idx + 511] # label at last element in x_win
 
        # convert to tensors
        x_tensor = torch.tensor(x_win.T, dtype=torch.float32) # .T swaps dimensions into [n_features, 512] as MOMENT expects
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor

# BUILD DATASETS AND LOADERS
trainData = ForexWindowDataset(X_train, y_train)
valData = ForexWindowDataset(X_val, y_val)
testData = ForexWindowDataset(X_test, y_test)

trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
valLoader = DataLoader(valData, batch_size=batch_size, shuffle=False)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=False)

# LOAD MOMENT MODEL
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "classification",
        "n_channels": X.shape[1],
        "num_class": 3,
        "freeze_encoder": True, # freeze embedding layer
        "freeze_embedder": True, # freeze transformer blocks
        "freeze_head": False # train classification head
    }
)
model.init()
model = model.to(device)

# DISPLAY TRAINABLE PARAMETERS
totalParams = sum(p.numel() for p in model.parameters())
trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {totalParams}")
print(f"Trainable parameters: {trainableParams}")

# CLASS WEIGHTING
counts = np.bincount(y_train, minlength=3).astype(np.float32)
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum() * 3 # normalise
weightsTensor = torch.tensor(weights, dtype=torch.float32).to(device)

# LOSS, OPTIMISER, SCHEDULER
criterion = torch.nn.CrossEntropyLoss(weight=weightsTensor)
optimiser = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)
totalSteps = len(trainLoader) * max_epochs
scheduler = OneCycleLR(optimiser, max_lr=learning_rate, total_steps=totalSteps, pct_start=0.3)
ampScaler = torch.amp.GradScaler("cuda")

# TRAINING LOOP
bestValF1 = 0.0

print("Fine-tuning model...")
for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}")

    model.train()
    for X_batch, y_batch in tqdm(trainLoader, desc="    train"):
        # cast to gpu
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # train
        optimiser.zero_grad() # clear gradients
        output = model(x_enc=X_batch)
        loss = criterion(output.logits, y_batch)
        # backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()