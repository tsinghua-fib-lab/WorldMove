import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from modules.utils import weight_init
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class AE(nn.Module):
    def __init__(self, feature_dim=37, latent_dim=8, rank_label_num=7):
        super(AE, self).__init__()
        self.rank_embedding = nn.Embedding(rank_label_num, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim - 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder_poi_dist = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 34),
            nn.Softmax(dim=-1),
        )
        self.decoder_poi_num = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        self.decoder_pop = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        self.decoder_rank = nn.Linear(latent_dim, rank_label_num)
        self.apply(weight_init)

    def forward(self, x):
        emb = self.encode(x)
        x_hat, rank_logits = self.decode(emb)
        loss = self.loss(x, x_hat, rank_logits)
        return x_hat, loss

    def encode(self, x):
        value_feature = x[:, :36]
        rank_label = x[:, 36].long()
        rank_embedding = self.rank_embedding(rank_label)
        return self.encoder(value_feature) + rank_embedding

    def decode(self, x_hat):
        # value
        poi_dist = self.decoder_poi_dist(x_hat)
        poi_num = self.decoder_poi_num(x_hat)
        pop = self.decoder_pop(x_hat)
        # rank
        rank_logits = self.decoder_rank(x_hat)
        rank_pred = torch.argmax(rank_logits, dim=-1).unsqueeze(-1)
        return (
            torch.cat([poi_dist, poi_num, pop, rank_pred], dim=-1),
            rank_logits,
        )

    def loss(self, x, x_hat, rank_logits):
        poi_dist, poi_num, pop, rank = (
            x[:, :34],
            x[:, 34],
            x[:, 35],
            x[:, 36],
        )
        poi_dist_hat, poi_num_hat, pop_hat = (
            x_hat[:, :34],
            x_hat[:, 34],
            x_hat[:, 35],
        )
        # kl for poi_dist
        kl_poi_dist = F.kl_div(
            torch.log(poi_dist_hat + 1e-6),
            poi_dist,
            reduction="batchmean",
        )
        # mse for poi_num
        mse_poi_num = F.mse_loss(poi_num_hat, poi_num)
        # mse for pop
        mse_pop = F.mse_loss(pop_hat, pop)
        # cross entropy for rank
        rank = rank.long()
        rank_loss = F.cross_entropy(rank_logits, rank)

        print(
            f"kl_poi_dist: {kl_poi_dist}, mse_poi_num: {mse_poi_num}, mse_pop: {mse_pop}, rank_loss: {rank_loss}"
        )
        kl_poi_dist = kl_poi_dist * 1e-2
        return kl_poi_dist + mse_poi_num + mse_pop + rank_loss


def get_grid_feature(city: str):
    poi_grid_count = np.load(f"data/poi/poi_grid_count_{city}_osm.npy")
    pop_gird_count = np.load(f"data/population/pop_grid_count_{city}.npy")
    rank_grid_label = np.load(f"data/rank/rank_grid_label_{city}.npy")

    poi_grid_dist = (poi_grid_count + 1e-6) / np.sum(
        poi_grid_count + 1e-6, axis=-1, keepdims=True
    )
    poi_grid_num = np.sum(poi_grid_count, axis=-1)
    poi_grid_num_norm = poi_grid_num / poi_grid_num.max()
    pop_gird_count_norm = pop_gird_count / pop_gird_count.max()
    gx, gy = pop_gird_count_norm.shape[:2]
    # grid_index = np.stack(
    #     np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij"),
    #     axis=-1,
    #     dtype=np.float32,
    # )
    # grid_index = grid_index / np.array([gx, gy], dtype=np.float32)
    grid_feature = np.concatenate(
        [
            poi_grid_dist,
            poi_grid_num_norm[..., None],
            pop_gird_count_norm[..., None],
            # grid_index,
            rank_grid_label[..., None],
        ],
        axis=-1,
    )

    grid_feature = grid_feature.reshape(-1, grid_feature.shape[-1])
    return grid_feature


def main():
    torch.manual_seed(42)
    # load data
    grid_feature_shanghai = get_grid_feature("shanghai")
    grid_feature_nanchang = get_grid_feature("nanchang")
    grid_feature = np.concatenate(
        [grid_feature_shanghai, grid_feature_nanchang], axis=0
    )
    print(f"grid_feature: {grid_feature.shape}")

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE()
    data = torch.tensor(grid_feature, dtype=torch.float32)

    # train
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    loss_list = []
    epoch = 15000
    bar = tqdm(range(epoch), dynamic_ncols=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)
    for _ in range(epoch):
        model.train()
        optimizer.zero_grad()
        _, loss = model(data)
        # check nan
        if torch.isnan(loss).any():
            print("nan loss, break")
            break
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        bar.set_description(f"loss: {loss.item()}, lr: {scheduler.get_last_lr()}")
        bar.update()

    # plot and save
    fig, ax = plt.subplots()
    ax.plot(loss_list)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    plt.savefig("saves/region-vae/loss.png")
    torch.save(model.state_dict(), "saves/region-vae/model_rank.pth")


if __name__ == "__main__":
    main()
