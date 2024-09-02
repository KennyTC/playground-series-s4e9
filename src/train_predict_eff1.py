import os
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from matplotlib import transforms
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
import os
import random
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import io
import gc

PRO_DIR = r"/home/kenny/Projects/kaggle/isic2024"
os.chdir(PRO_DIR)
print("project_directory:", PRO_DIR)

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
SEED = 42
seed_everything(SEED)


class ISICDataset(Dataset):
    def __init__(self, df_meta , transform=None):   
        df_meta = df_meta.reset_index(drop=True)  # Reset the index
        file_list = {}
        for _, row in df_meta.iterrows():        
            if row["set"]=="org":
                file_list[row["isic_id"]] = f"{PRO_DIR}/input/train-image/image/{row['isic_id']}.jpg"
            else:
                file_list[row["isic_id"]] = f"{PRO_DIR}/data/external/{row['isic_id']}.jpg"
        self.file_list = file_list
        self.df = df_meta
        # print(f"filelist {len(self.file_list)}, df {self.df.shape}")
        assert len(self.file_list.keys()) == self.df.shape[0]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = self.df.loc[idx,"isic_id"]
        target = self.df.loc[idx,"target"]        
        img_path = self.file_list[isic_id]
        # print(f"isic_id {isic_id}. target {target}, path {img_path}")
        try:
            img = Image.open(img_path).convert("RGB")            
            if self.transform:
                img = self.transform(img)
        except Exception as ex:
            raise ex

        return img, img_path, target
class ISICDatasetTest(Dataset):
    def __init__(self, df_meta, file_list, transform=None):        
        self.file_list = file_list
        self.df = df_meta
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        isic_id = self.df.loc[idx,"isic_id"]
        img_path = self.file_list[isic_id]
        try:
            img = Image.open(img_path).convert("RGB")            
            if self.transform:
                img = self.transform(img)
        except Exception as ex:
            raise ex

        return img

def read_images_from_hdf5_and_save(file_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(file_path, 'r') as file:
        ids_list = list(file.keys())        
        ids_paths = {}
        for img_id in tqdm(ids_list):
            image_data = file[img_id][()]
            image_path = os.path.join(output_dir, f"{img_id}.png")  # Define how you want to save the file
            if os.path.exists(image_path):
                ids_paths[img_id] = image_path
                continue
            # Save the image data to a file
            with Image.open(io.BytesIO(image_data)) as image:
                image.save(image_path)

            # Store the path instead of the image data
            ids_paths[img_id] = image_path
    return ids_paths

## 2. DEFINE MODEL -----------------------------------------------------------
IMG_SIZE=192
BATCH_SIZE=24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name="efficientnet_v2_m"):
    if model_name == "mobilenet_v3_small":
    # mobile net
        model = models.mobilenet_v3_small()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    if model_name == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)        
    

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = BCEWithLogitsLoss()
    return model, optimizer, criterion

## 3. DEFINE METRICS -------------------------------------------------------

from sklearn.metrics import auc, roc_curve
def compute_pauc(y_true, y_scores, tpr_threshold=0.8):
    """
    Compute the partial AUC above a given TPR threshold.

    Parameters:
    y_true (np.array): True binary labels.
    y_scores (np.array): Target scores.
    tpr_threshold (float): TPR threshold above which to compute the pAUC.
    Returns:
    float: The partial AUC above the given TPR threshold.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the indices where the TPR is above the threshold
    tpr_above_threshold_indices = np.where(tpr >= tpr_threshold)[0]
    if len(tpr_above_threshold_indices) == 0:
        return 0.0

    # Extract the indices for the ROC segment above the threshold
    start_index = tpr_above_threshold_indices[0] 
    fpr_above_threshold = fpr[start_index:]
    tpr_above_threshold = tpr[start_index:] - tpr_threshold
    partial_auc = auc(fpr_above_threshold, tpr_above_threshold)    
    return partial_auc

## 4. LOAD DATASET -------------------------------------------------------
class PATHS:
    train_images_h5_path = f"{PRO_DIR}/input/train-image.hdf5"
    test_images_h5_path = f"{PRO_DIR}/input/test-image.hdf5"    
    train_metadata_path = f"{PRO_DIR}/input/train-metadata.csv"
    test_metadata_path = f"{PRO_DIR}/input/test-metadata.csv"    
    submission_path = f"{PRO_DIR}/input/sample_submission.csv"
    train_metadata_path_ext = f"{PRO_DIR}/data/external/metadata.csv"

# # get more 2000 malignant data from external dataset
df_meta = pd.read_csv(PATHS.train_metadata_path)[["isic_id","patient_id","target"]]
df_meta["set"] = "org"

df_meta_ext = pd.read_csv(PATHS.train_metadata_path_ext)
df_meta_ext = df_meta_ext.loc[df_meta_ext["benign_malignant"]=="malignant",["isic_id","patient_id","benign_malignant"]]
df_meta_ext = df_meta_ext.loc[(df_meta_ext["benign_malignant"]=="malignant")&(~df_meta_ext["patient_id"].isna())]
df_meta_ext["benign_malignant"] = 1
df_meta_ext.rename(columns={"benign_malignant":"target"}, inplace=True)
df_meta_ext["set"]="ext"
df_meta_ext = df_meta_ext.sample(frac=1).sample(n=2000)

meta_train = pd.concat([df_meta, df_meta_ext])
meta_train["target"]=meta_train["target"].astype("int")

df_meta_pos = meta_train.loc[meta_train["target"]==1]
df_meta_neg = meta_train.loc[meta_train["target"]==0].sample(frac=1).sample(n=len(df_meta_pos)*1, random_state=SEED)
meta_train = pd.concat([df_meta_pos, df_meta_neg]).reset_index()
# meta_train = meta_train.set_index("index")

## 5. TRAIN MODEL ----------------------------------------------
def train(model, train_loader, optimizer, criterion):
    total_loss = 0
    all_targets = []
    all_probs = []        

    model.train()
    for input,_, targets in train_loader:        
        input = input.to(DEVICE)
        targets = targets.to(DEVICE)

        targets = targets.unsqueeze(1) # make the target [batch, 1]
        targets = targets.float() # BCEWithLogitsLoss requires targets as float()
        # print(f"input shape {input.shape}")
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, targets)
        total_loss += loss.item()
        
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(output).cpu().detach().numpy()
        # predictions = (probs > 0.5)

        all_targets.extend(targets.cpu().detach().numpy().flatten())
        all_probs.extend(probs.flatten())

        loss.backward()
        optimizer.step()
    
    pauc = compute_pauc(np.array(all_targets), np.array(all_probs))
    return total_loss, pauc

def val(model, val_loader, criterion):
    total_loss= 0
    all_targets = []
    all_probs = []        
    model.eval()
    with torch.no_grad():
        for input, _, targets in val_loader:
            input = input.to(DEVICE)
            targets = targets.to(DEVICE)

            targets = targets.unsqueeze(1) # make the target [batch, 1]
            targets = targets.float() # BCEWithLogitsLoss requires targets as float()

            output = model(input)
            val_loss = criterion(output, targets)
            total_loss +=  val_loss.item()

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(output).cpu().detach().numpy().flatten()
            # predictions = (probs > 0.5)
            
            all_targets.extend(targets.cpu().detach().numpy().flatten())
            all_probs.extend(probs)           
    
    # pauc = compute_pauc(all_targets, all_predictions)
    # print(f"all_targets {len(all_targets)}, all_probs {len(all_probs)}")
    pauc = compute_pauc(np.array(all_targets), np.array(all_probs))
    return total_loss, pauc, all_probs, all_targets

def get_mean_std(df):
    trn_dataset = ISICDataset(df,
                                transform=transforms.Compose([
                                    transforms.Resize((IMG_SIZE, IMG_SIZE)),            
                                    transforms.ToTensor(),
                                ])
                            ) 
    train_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    mean = 0.0
    for images, _,_ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)        
        images = images.view(batch_samples, images.size(1), -1)  # print(images.shape) # will be (64, 3, 224x224)
        mean += images.mean(2).sum(0)  
    mean = mean / len(train_loader.dataset)

    var = 0.0
    for images, _,_ in train_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(train_loader.dataset)*IMG_SIZE*IMG_SIZE))
    return mean, std

EXP_ID    = 2
MODEL_NAME = "efficientnet_v2_m"
NUM_EPOCHS = 30
# BATCH_SIZE = 32
NOTE="with_2k_external_db_split_patient_id"
EXP_NAME = "{:03}_{}_{}_{}_{}".format(EXP_ID, MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, NOTE)  # you can name your experiment whatever you like
# SAVE_PATH = "/kaggle/working"
SAVE_PATH = "models"

logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(EXP_NAME))
def report_gpu(): 
    print(torch.cuda.list_gpu_processes()) 
    gc.collect() 
    torch.cuda.empty_cache()

TRAIN_CV = True
if TRAIN_CV:
    logging.info(f"TRAIN CV ---------------------------------")
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=5)

    meta_train["fold"] = -1
    for idx, (train_idx, val_idx) in enumerate(cv.split(meta_train, meta_train["target"], groups=meta_train["patient_id"])):
        meta_train.loc[val_idx, "fold"] = idx

    # Add summary
    fold_summary = meta_train.groupby("fold")["patient_id"].nunique().to_dict()
    total_patients = meta_train["patient_id"].nunique()

    print(f"Fold Summary (patients per fold):")
    for fold, count in fold_summary.items():
        if fold != -1:  # Exclude the initialization value
            logging.info(f"Fold {fold}: {count} patients")
    logging.info(f"Total patients: {total_patients}")

    train_data_stat = {}

    for i, (i_trn, i_val) in enumerate():   
        mean, std = get_mean_std(meta_train.loc[i_trn])        
        train_data_stat[i] = {"mean": mean, "std": std}
        logging.info(f"Fold {i}, train {meta_train.loc[i_trn].shape}, val {meta_train.loc[i_val].shape}")
        logging.info(f"fold {i}, train_data_mean {mean}, train_data_std {std}")

    # train_data_stat ={
    #     0: {"mean": [0.6908, 0.5208, 0.4539], "std":[0.1609, 0.1558, 0.1664]},
    #     1: {"mean": [0.6903, 0.5210, 0.4524], "std":[0.1599, 0.1543, 0.1652]},
    #     2: {"mean": [0.6929, 0.5238, 0.4557], "std":[0.1619, 0.1567, 0.1676]},
    #     3: {"mean": [0.6937, 0.5233, 0.4546], "std":[0.1597, 0.1549, 0.1646]},
    #     4: {"mean": [0.6932, 0.5236, 0.4543], "std":[0.1611, 0.1559, 0.1659]},
    # }
        
    for i, (i_trn, i_val) in enumerate(cv.split(meta_train.drop("target", axis=1), meta_train["target"], groups=meta_train["patient_id"])):
        train_data_mean = train_data_stat[i]["mean"]
        train_data_std = train_data_stat[i]["std"]
        train_trans = transforms.Compose([    
            transforms.Resize((IMG_SIZE, IMG_SIZE)),       
            transforms.ToTensor(),
            transforms.Normalize(mean=train_data_mean, std=train_data_std),
        ])
        val_trans =  transforms.Compose([    
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=train_data_mean, std=train_data_std),
        ])

        trn_dataset = ISICDataset(
            meta_train.loc[i_trn],
            transform=train_trans
        )
        val_dataset = ISICDataset(
            meta_train.loc[i_val],
            transform=val_trans
        )
        
        # Now, you can create separate data loaders for each split:
        train_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        logging.info(f"Fold {i}")
        model, optimizer, criterion = load_model("efficientnet_v2_m")

        best_val_loss, best_val_pauc = 100, 0        
        for epoch in range(30):
            gc.collect()
            
            train_loss, train_pauc = train(model, train_loader, optimizer, criterion)
            val_loss, val_pauc, _,_ = val(model, val_loader, criterion)        
            if val_pauc > best_val_pauc:
                best_val_pauc = val_pauc
                os.makedirs(f"{SAVE_PATH}/{EXP_NAME}", exist_ok=True)            
                torch.save(model.state_dict(),f"{SAVE_PATH}/{EXP_NAME}/best_{i}.pth")
                logging.info(f"Epoch {epoch}, train_loss {train_loss:.4f}, train_pauc {train_pauc:.2f}, val_loss {val_loss:.4f}, val_pauc {val_pauc:.2f} --> Best val_pauc {val_pauc:.2f} at epoch {epoch}")    
            else:        
                logging.info(f"Epoch {epoch}, train_loss {train_loss:.4f}, train_pauc {train_pauc:.2f}, val_loss {val_loss:.4f}, val_pauc {val_pauc:.2f}") 
        
        report_gpu()
        del model

RETRAIN = False
if RETRAIN:

    logging.info(f"TRAIN ALL---------------------------------")
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # Perform the split
    train_idx, val_idx = next(gss.split(meta_train, groups=meta_train['patient_id']))
    train_data = meta_train.loc[train_idx]
    val_data = meta_train.loc[val_idx]

    train_data_mean, train_data_std = get_mean_std(train_data)
    # train_data_mean=[0.6835, 0.5216, 0.4443]
    # train_data_std = [0.1682, 0.1579, 0.1619]
    logging.info(f"train {len(train_data)}, val {len(val_data)}, train's mean {train_data_mean}, train's std {train_data_std}")
    train_trans = transforms.Compose([    
        transforms.Resize((IMG_SIZE, IMG_SIZE)),       
        transforms.ToTensor(),
        transforms.Normalize(mean=train_data_mean, std=train_data_std),
    ])
    val_trans =  transforms.Compose([    
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=train_data_mean, std=train_data_std),
    ])

    trn_dataset = ISICDataset(
        train_data,
        transform=train_trans
    )
    val_dataset = ISICDataset(
        val_data,
        transform=val_trans
    )
    
    # Now, you can create separate data loaders for each split:
    train_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, optimizer, criterion = load_model("efficientnet_v2_m")

    best_val_loss, best_val_pauc = 100, 0    
    for epoch in range(NUM_EPOCHS):
        gc.collect()

        train_loss, train_pauc = train(model, train_loader, optimizer, criterion)
        val_loss, val_pauc, _,_ = val(model, val_loader, criterion)        
        if val_pauc > best_val_pauc:
            best_val_pauc = val_pauc
            os.makedirs(f"{SAVE_PATH}/{EXP_NAME}", exist_ok=True)            
            torch.save(model.state_dict(),f"{SAVE_PATH}/{EXP_NAME}/best_all.pth")
            logging.info(f"Epoch {epoch}, train_loss {train_loss:.4f}, train_pauc {train_pauc:.2f}, val_loss {val_loss:.4f}, val_pauc {val_pauc:.2f} --> Best val_pauc {val_pauc:.2f} at epoch {epoch}")    
        else:        
            logging.info(f"Epoch {epoch}, train_loss {train_loss:.4f}, train_pauc {train_pauc:.2f}, val_loss {val_loss:.4f}, val_pauc {val_pauc:.2f}") 
    
    report_gpu()
    del model

INFERENCE=True
if INFERENCE:
    pass