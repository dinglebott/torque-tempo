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
featureList = ["open_return", "high_return", "low_return", "close_return", "vol_return"]

# SPLIT DATA
X = df[featureList]
y = df["target"]
trainEndIdx = int(train_split * len(X))
valEndIdx = int((train_split + val_split) * len(X))

X_train = X[:trainEndIdx]
X_test = X[valEndIdx:]
y_train = y[:trainEndIdx].values
y_test = y[valEndIdx:].values

# SCALE DATA (becomes numpy array after scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
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
testData = ForexWindowDataset(X_test, y_test)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=False)

# LOAD MOMENT MODEL
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "classification",
        "n_channels": X_train.shape[1],
        "num_class": 3,
        "freeze_encoder": True, # freeze all
        "freeze_embedder": True,
        "freeze_head": True
    }
)
model.init()
model = model.to(device)

# CLASS WEIGHTING
counts = np.bincount(y_train, minlength=3).astype(np.float32)
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum() * 3 # normalise
weightsTensor = torch.tensor(weights, dtype=torch.float32).to(device)

# LOSS
criterion = torch.nn.CrossEntropyLoss(weight=weightsTensor)

# EVALUATION
bestState = torch.load(Path(f"models/MOMENT_{instrument}_{granularity}_{yearNow}.pt"), map_location=device)
model.load_state_dict(bestState)

model.eval()
testLosses, testPreds, testProbs = [], [], []
for X_batch_test, y_batch_test in tqdm(testLoader, desc="TEST"):
    X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)
    with torch.no_grad():
        testOutput = model(x_enc=X_batch_test)
        testLoss = criterion(testOutput.logits, y_batch_test)
    testLosses.append(testLoss.item())
    testPreds.extend(testOutput.logits.argmax(dim=1).cpu().numpy())
    testProbs.extend(testOutput.logits.softmax(dim=1).cpu().numpy())

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

print(f"\n F1: {testF1:.4f}")
print(f"ROC-AUC score: {rocAucScore:.4f}")
print(f"Confusion matrix:\n{cmatrixDf}")
print(f"Classification report:\n{testReport}")