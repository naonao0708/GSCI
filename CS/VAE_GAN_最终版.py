# 融合版 VAE-GAN 主程序
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ---------------------- 配置设备 ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 模型定义 ----------------------
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Softplus()
        )
        self.residual = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Linear(1024, output_dim)
        )

    def forward(self, z):
        return self.decoder(z) + 0.3 * self.residual(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class VAE_GAN(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, latent_dim)
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.generator(z)
        return recon_x, mu, logvar


# ---------------------- 损失函数 ----------------------
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def vae_loss(self, recon_x, x, mu, logvar, zero_mask):
        weights = torch.where(zero_mask, 1.0, 3.0)
        recon_loss = (F.mse_loss(recon_x, x, reduction='none') * weights).sum()
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

    def adversarial_loss(self, d_real, d_fake):
        real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
        fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        return (real_loss + fake_loss) / 2

    def forward(self, recon_x, x, mu, logvar, d_real, d_fake, zero_mask):
        return self.alpha * self.vae_loss(recon_x, x, mu, logvar, zero_mask) + \
               self.beta * self.adversarial_loss(d_real, d_fake)


# ---------------------- 训练引擎 ----------------------
class TrainingEngine:
    def __init__(self, model, lr=1e-4):
        self.model = model.to(device)
        self.optim_G = optim.Adam(list(model.encoder.parameters()) + list(model.generator.parameters()), lr=lr, weight_decay=1e-5)
        self.optim_D = optim.Adam(model.discriminator.parameters(), lr=lr * 0.5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim_G, 'min', factor=0.5, patience=5)
        self.loss_fn = HybridLoss()
        self.best_loss = float('inf')
        self.patience = 10
        self.no_improve = 0
        self.loss_history = []  # 记录每轮 Generator 总损失
        self.d_loss_history = []  # 记录每轮 Discriminator 损失

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_d_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            zero_mask = (batch == 0)

            self.optim_D.zero_grad()
            with torch.no_grad():
                recon_x, _, _ = self.model(batch)
            real_pred = self.model.discriminator(batch)
            fake_pred = self.model.discriminator(recon_x.detach())
            d_loss = self.loss_fn.adversarial_loss(real_pred, fake_pred)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 1.0)
            self.optim_D.step()

            self.optim_G.zero_grad()


            recon_x, mu, logvar = self.model(batch)
            real_pred = self.model.discriminator(batch)
            fake_pred = self.model.discriminator(recon_x)
            loss = self.loss_fn(recon_x, batch, mu, logvar, real_pred, fake_pred, zero_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 1.0)
            self.optim_G.step()

            total_loss += loss.item()
            total_d_loss += d_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)
        self.scheduler.step(avg_loss)
        self.loss_history.append(avg_loss)
        self.d_loss_history.append(avg_d_loss)
        return avg_loss

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(dataloader)
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
            if self.no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.4f}")

##################### 新增损失函数 ######################
def plot_training_loss(engine, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(engine.loss_history, label='Generator Loss', linewidth=2)
    plt.plot(engine.d_loss_history, label='Discriminator Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve (VAE-GAN)', fontsize=14)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"训练损失图已保存至：{save_path}")
    plt.show()

# ---------------------- 评估函数 ----------------------
# ---------------------- 评估模块 ----------------------
def safe_pearsonr(x, y):
    try:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        return float(pearsonr(x, y)[0])
    except:
        return 0.0

def safe_cosine(a, b):
    try:
        a = np.asarray(a).reshape(1, -1)
        b = np.asarray(b).reshape(1, -1)
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return float(cosine_similarity(a, b)[0][0])
    except:
            return 0.0

# def evaluate_gene(original, recovered, idx, name):
#     try:
#         o = original[:, idx]
#         r = recovered[:, idx]
#         return {
#             'Gene': name,
#             'MSE': mean_squared_error(o, r),
#             'MAE': mean_absolute_error(o, r),
#             'PCC': safe_pearsonr(o, r),
#             'CS': safe_cosine(o, r)
#         }
#     except:
#         return None
def evaluate_gene(original, recovered, idx, name):
    try:
        o = original[:, idx]
        r = recovered[:, idx]
        mse = mean_squared_error(o, r)
        mae = mean_absolute_error(o, r)
        pcc = safe_pearsonr(o, r)
        cs = safe_cosine(o, r)
        print(f"[✓] {name}: MSE={mse:.4f}, MAE={mae:.4f}, PCC={pcc:.4f}, CS={cs:.4f}")
        return {
            'Gene': name,
            'MSE': mse,
            'MAE': mae,
            'PCC': pcc,
            'CS': cs
        }
    except Exception as e:
        print(f"[X] Gene {name} 评估失败: {e}")
        return None


def evaluate_all(original, recovered, zero_mask, gene_info=None):
    result = {
        'Global_MSE': mean_squared_error(original, recovered),
        'Global_MAE': mean_absolute_error(original, recovered),
        'Global_PCC': safe_pearsonr(original.flatten(), recovered.flatten()),
        'Global_CS': safe_cosine(original.flatten(), recovered.flatten())
    }

    # 分层评估
    mask_flat = zero_mask.flatten()
    orig_flat = original.flatten()
    rec_flat = recovered.flatten()

    zero_idx = np.where(mask_flat)[0]
    non_zero_idx = np.where(~mask_flat)[0]

    if len(zero_idx) > 0:
        result['Zero_MSE'] = mean_squared_error(orig_flat[zero_idx], rec_flat[zero_idx])
        result['Zero_MAE'] = mean_absolute_error(orig_flat[zero_idx], rec_flat[zero_idx])
    if len(non_zero_idx) > 0:
        result['NonZero_MSE'] = mean_squared_error(orig_flat[non_zero_idx], rec_flat[non_zero_idx])
        result['NonZero_MAE'] = mean_absolute_error(orig_flat[non_zero_idx], rec_flat[non_zero_idx])
        result['NonZero_PCC'] = safe_pearsonr(orig_flat[non_zero_idx], rec_flat[non_zero_idx])
        result['NonZero_CS'] = safe_cosine(orig_flat[non_zero_idx], rec_flat[non_zero_idx])

    # 基因评估
    gene_metrics = []
    if gene_info:
        for idx, name in gene_info:
            res = evaluate_gene(original, recovered, idx, name)
            if res:
                gene_metrics.append(res)
        result['Gene_Metrics'] = gene_metrics

    return result

# ---------------------- 数据加载与主运行 ----------------------
if __name__ == '__main__':
    # 选择数据集路径（默认 GSE124989）
    # log_path = './Data/GSE124989/GSE124989_log.npy'
    # raw_path = './Data/GSE124989/GSE124989_raw.npy'
    # gene_path = './Data/GSE124989/gene.txt'
    # output_dir = './result/GSE124989/results/'

    # GSE123358（可切换）
    log_path = './Data/GSE123358/GSE123358_log.npy'
    raw_path = './Data/GSE123358/GSE123358_raw.npy'
    gene_path = './Data/GSE123358/gene.txt'
    output_dir = './result/GSE123358/results/'

    # GSE147326（可切换）
    # log_path = './Data/GSE147326/GSE147326_log.npy'
    # raw_path = './Data/GSE147326/GSE147326.npy'
    # gene_path = './Data/GSE147326/gene.txt'
    # output_dir = './result/GSE147326/results/'

    os.makedirs(output_dir, exist_ok=True)

    log_data = np.load(log_path)
    raw_data = np.load(raw_path)
    log_data = log_data.T
    raw_data = raw_data.T
    print(log_data.shape)
    print(raw_data.shape)
    gene_list = pd.read_csv(gene_path, header=None)[0].tolist()
    if log_data.shape[1] != len(gene_list):
        raise ValueError("log_data 和 gene 列表长度不一致")

    zero_mask = (raw_data == 0)
    tensor_data = torch.FloatTensor(log_data).to(device)
    dataloader = torch.utils.data.DataLoader(tensor_data, batch_size=128, shuffle=True)

    # 模型与训练
    model = VAE_GAN(input_dim=log_data.shape[1], latent_dim=32)
    engine = TrainingEngine(model)
    engine.train(dataloader, epochs=200)

    # 重建与评估
    with torch.no_grad():
        model.eval()
        recovered = model(tensor_data)[0].cpu().numpy()

    # 匹配关键基因
    target_genes = [
        'BRCA1', 'BRCA2', 'BARD1', 'BRIP1', 'PALB2', 'RAD51', 'RAD54L', 'XRCC3',
        'ERBB2', 'ESR1', 'PGR', 'PIK3CA', 'TP53', 'PPM1D', 'RB1CC1', 'HMMR', 'NQO2',
        'SLC22A18', 'PTEN', 'EGFR', 'KIT', 'NOTCH1', 'FZD7', 'LRP6', 'FGFR1', 'CCND1'
    ]
    gene_info = [(i, g) for i, g in enumerate(gene_list) if g in target_genes]
    print(f"gene_info: {gene_info[:5]} ... 共 {len(gene_info)} 个")

    results = evaluate_all(log_data, recovered, zero_mask, gene_info)
    print("评估结果：")
    for k, v in results.items():
        if not isinstance(v, list):
            print(f"{k}: {round(v, 4)}")

    # 保存结果（分两个表）
    global_df = pd.DataFrame({
        'Global_MSE': [results['Global_MSE']],
        'Global_MAE': [results['Global_MAE']],
        'Global_PCC': [results['Global_PCC']],
        'Global_CS': [results['Global_CS']],
        'Zero_MSE': [results.get('Zero_MSE', np.nan)],
        'Zero_MAE': [results.get('Zero_MAE', np.nan)],
        'NonZero_MSE': [results.get('NonZero_MSE', np.nan)],
        'NonZero_MAE': [results.get('NonZero_MAE', np.nan)],
        'NonZero_PCC': [results.get('NonZero_PCC', np.nan)],
        'NonZero_CS': [results.get('NonZero_CS', np.nan)]
    })
    plot_training_loss(engine, save_path=os.path.join(output_dir, 'vae_gan_loss_curve.png'))
    # global_df.to_csv(os.path.join(output_dir, 'GSE147326_VAE-GAN-global_metrics.csv'), index=False)
    #
    # if 'Gene_Metrics' in results:
    #     gene_df = pd.DataFrame(results['Gene_Metrics'])
    #     gene_df.to_csv(os.path.join(output_dir, 'GSE147326_VAE-GAN-gene_metrics.csv'), index=False)
    #     print(f"基因指标已保存")
    #
    # np.save(os.path.join(output_dir, 'GSE147326_VAE-GAN.npy'), recovered)
    # np.save(os.path.join(output_dir, 'GSE147326_VAE-GAN.csv'), recovered)

    # global_df.to_csv(os.path.join(output_dir, 'GSE124989_VAE-GAN-global_metrics.csv'), index=False)
    #
    # if 'Gene_Metrics' in results:
    #     gene_df = pd.DataFrame(results['Gene_Metrics'])
    #     gene_df.to_csv(os.path.join(output_dir, 'GSE124989_VAE-GAN-gene_metrics.csv'), index=False)
    #
    # np.save(os.path.join(output_dir, 'GSE124989_VAE-GAN.npy'), recovered)
    # np.save(os.path.join(output_dir, 'GSE124989_VAE-GAN.csv'), recovered)

    # global_df.to_csv(os.path.join(output_dir, 'GSE123358_VAE-GAN-global_metrics.csv'), index=False)
    #
    # if 'Gene_Metrics' in results:
    #     gene_df = pd.DataFrame(results['Gene_Metrics'])
    #     gene_df.to_csv(os.path.join(output_dir, 'GSE123358_VAE-GAN-gene_metrics.csv'), index=False)
    #
    # np.save(os.path.join(output_dir, 'GSE123358_VAE-GAN.npy'), recovered)
    # np.save(os.path.join(output_dir, 'GSE123358_VAE-GAN.csv'), recovered)
    # print(" 结果保存完成 ✅")

