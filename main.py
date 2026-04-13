import json
import numpy as np
import pandas as pd
from custom_modules import dataparser
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from momentfm import MOMENTPipeline
from tqdm import tqdm
from pathlib import Path
import warnings

# shut up
warnings.filterwarnings("ignore", message=".*Only reconstruction head is pre-trained.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")

# HYPERPARAMETERS
batch_size = 32
max_epochs = 10
learning_rate = 1e-5
numBlocksTrain = 3
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
featureList = [
    "adx_direction", "macd_hist", "dist_high", "dist_low",
    "close_return", "high_return", "low_return", "vol_return",
    "atr_14", "hl_spread",
    "rsi_14", "dist_ema15", "vol_momentum"
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
    "AutonLab/MOMENT-1-small",
    model_kwargs={
        "task_name": "classification",
        "n_channels": X_train.shape[1],
        "num_class": 3,
        "freeze_encoder": True, # freeze embedding layer
        "freeze_embedder": False, # train (some) transformer blocks
        "freeze_head": False # train classification head
    }
)
model.init()
model = model.to(device)

# SELECTIVELY REFREEZE TRANSFORMER
for i, block in enumerate(model.encoder.block):
    freeze = i < (len(model.encoder.block) - numBlocksTrain)
    for param in block.parameters():
        param.requires_grad = not freeze

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

# TRAINING LOOP
bestValF1 = 0.0

for epoch in range(max_epochs):
    # epoch loop
    trainLosses, trainPreds = [], []
    model.train()
    for X_batch, y_batch in tqdm(trainLoader, desc="TRAIN"):
        # cast to gpu
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # train
        optimiser.zero_grad() # clear gradients
        output = model(x_enc=X_batch)
        logits = model.head(output.embeddings)
        loss = criterion(logits, y_batch)
        # backpropagation
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        scheduler.step()
        # record training metrics
        trainLosses.append(loss.item())
        trainPreds.extend(logits.argmax(dim=1).cpu().numpy())
    
    # compile epoch training metrics
    avgTrainLoss = np.mean(trainLosses)
    trainF1 = f1_score(y_train[511:], trainPreds, average="macro", zero_division=0)
    
    # validation
    model.eval()
    valLosses, valPreds = [], []
    for X_batch_val, y_batch_val in tqdm(valLoader, desc="VAL"):
        # cast to gpu
        X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
        # infer
        with torch.no_grad():
            valOutput = model(x_enc=X_batch_val)
            valLogits = model.head(valOutput.embeddings)
            valLoss = criterion(valLogits, y_batch_val)
        # record
        valLosses.append(valLoss.item())
        valPreds.extend(valLogits.argmax(dim=1).cpu().numpy())
    
    # compile epoch validation metrics
    avgValLoss = np.mean(valLosses)
    valF1 = f1_score(y_val[511:], valPreds, average="macro", zero_division=0)
    print(f"Epoch {epoch + 1} | Loss: {avgValLoss:.4f} | Train loss: {avgTrainLoss:.4f} | F1: {valF1:.4f} | Train F1: {trainF1:.4f}")

    # save checkpoint
    if valF1 > bestValF1:
        bestValF1 = valF1
        torch.save(model.state_dict(), Path(f"models/MOMENT_{instrument}_{granularity}_{yearNow}.pt"))

# EVALUATION
bestState = torch.load(Path(f"models/MOMENT_{instrument}_{granularity}_{yearNow}.pt"), map_location=device)
model.load_state_dict(bestState)

model.eval()
testLosses, testPreds, testProbs = [], [], []
for X_batch_test, y_batch_test in tqdm(testLoader, desc="TEST"):
    X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)
    with torch.no_grad():
        testOutput = model(x_enc=X_batch_test)
        testLogits = model.head(testOutput.embeddings)
        testLoss = criterion(testLogits, y_batch_test)
    testLosses.append(testLoss.item())
    testPreds.extend(testLogits.argmax(dim=1).cpu().numpy())
    testProbs.extend(testLogits.softmax(dim=1).cpu().numpy())

avgTestLoss = np.mean(testLosses)
testF1 = f1_score(y_test[511:], testPreds, average="macro", zero_division=0)
rocAucScore = roc_auc_score(y_test[511:], testProbs, multi_class="ovr", average="macro")
cmatrix = confusion_matrix(y_test[511:], testPreds)
cmatrixDf = pd.DataFrame(cmatrix, index=["Real -", "Real ~", "Real +"], columns=["Pred -", "Pred ~", "Pred +"])
cmatrixDf["Count"] = cmatrixDf.sum(axis=1)
cmatrixDf.loc["Count"] = cmatrixDf.sum(axis=0)
testReport = classification_report(
    y_test[511:], testPreds,
    target_names=["DOWN", "FLAT", "UP"],
    zero_division=0
)

print(f"\nF1: {testF1:.4f}")
print(f"ROC-AUC score: {rocAucScore:.4f}")
print(f"Confusion matrix:\n{cmatrixDf}")
print(f"Classification report:\n{testReport}")