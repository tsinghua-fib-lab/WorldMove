import numpy as np
import torch
from dataset.feature import TrajFeatureDataset
from model import RegionDiff
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = TrajFeatureDataset(
    "data/traj/user_day_senegal_feature_rank_idx.npy",  # path to feature embedding of city
    norm=False,
)
loader = DataLoader(dataset, batch_size=2048, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegionDiff.load_from_checkpoint(
    "checkpoints/epoch=199-step=21600.ckpt",  # path to trained model checkpoint
    map_location=device,
)
model.eval()
for param in model.parameters():
    param.requires_grad = False

traj_gen = []
for batch in tqdm(loader):
    batch = batch.to(device)
    real_idx = model._reconstruct_idx(model._get_training_target(batch))
    sample = model.sampling(batch, num_steps=model.sample_steps, show_progress=False)
    gen_idx = model._reconstruct_idx(sample)
    traj_gen.append(gen_idx.cpu().numpy())
traj_gen = np.concatenate(traj_gen)
np.save("saves/generated_trajs/gen_idx.npy", traj_gen)
