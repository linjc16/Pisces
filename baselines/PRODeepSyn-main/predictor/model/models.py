import torch
import torch.nn as nn
import torch.nn.functional as F


class PPI2Cell(nn.Module):

    def __init__(self, n_cell: int, ppi_emb: torch.Tensor, bias=True):
        super(PPI2Cell, self).__init__()
        self.n_cell = n_cell
        self.cell_emb = nn.Embedding(n_cell, ppi_emb.shape[1], max_norm=1.0, norm_type=2.0)
        if bias:
            self.bias = nn.Parameter(torch.randn((1, ppi_emb.shape[0])), requires_grad=True)
        else:
            self.bias = 0
        self.ppi_emb = ppi_emb.permute(1, 0)

    def forward(self, x: torch.Tensor):
        x = x.squeeze(dim=1)
        emb = self.cell_emb(x)
        y = emb.mm(self.ppi_emb)
        y += self.bias
        return y


class PPI2CellV2(nn.Module):

    def __init__(self, n_cell: int, ppi_emb: torch.Tensor, hidden_dim: int, bias=True):
        super(PPI2CellV2, self).__init__()
        self.n_cell = n_cell
        self.projector = nn.Sequential(
            nn.Linear(ppi_emb.shape[1], hidden_dim, bias=bias),
            nn.LeakyReLU()
        )
        self.cell_emb = nn.Embedding(n_cell, hidden_dim, max_norm=1.0, norm_type=2.0)
        self.ppi_emb = ppi_emb

    def forward(self, x: torch.Tensor):
        x = x.squeeze(dim=1)
        proj = self.projector(self.ppi_emb).permute(1, 0)
        emb = self.cell_emb(x)
        y = emb.mm(proj)
        return y


class SynEmb(nn.Module):

    def __init__(self, n_drug: int, drug_dim: int, n_cell: int, cell_dim: int, hidden_dim: int):
        super(SynEmb, self).__init__()
        self.drug_emb = nn.Embedding(n_drug, drug_dim, max_norm=1)
        self.cell_emb = nn.Embedding(n_cell, cell_dim, max_norm=1)
        self.network = DNN(2 * drug_dim + cell_dim, hidden_dim)

    def forward(self, drug1, drug2, cell):
        d1 = self.drug_emb(drug1).squeeze(1)
        d2 = self.drug_emb(drug2).squeeze(1)
        c = self.cell_emb(cell).squeeze(1)
        return self.network(d1, d2, c)


class AutoEncoder(nn.Module):

    def __init__(self, input_size: int, latent_size: int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size)
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class GeneExpressionAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int):
        super(GeneExpressionAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, latent_size),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, input_size),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class DrugFeatAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int):
        super(DrugFeatAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class DSDNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DSDNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, feat: torch.Tensor):
        out = self.network(feat)
        return out


class DNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out


class BottleneckLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(BottleneckLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class PatchySan(nn.Module):

    def __init__(self, drug_size: int, cell_size: int, hidden_size: int, field_size: int):
        super(PatchySan, self).__init__()
        # self.drug_proj = nn.Linear(drug_size, hidden_size, bias=False)
        # self.cell_proj = nn.Linear(cell_size, hidden_size, bias=False)
        self.conv = nn.Sequential(
            BottleneckLayer(field_size, 16),
            BottleneckLayer(16, 32),
            BottleneckLayer(32, 16),
            BottleneckLayer(16, 1),
        )
        self.network = nn.Sequential(
            nn.Linear(2 * drug_size + cell_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        cell_feat = cell_feat.permute(0, 2, 1)
        cell_feat = self.conv(cell_feat).squeeze(1)
        # drug1_feat = self.drug_proj(drug1_feat)
        # drug2_feat = self.drug_proj(drug2_feat)
        # express = self.cell_proj(cell_feat)
        # feat = torch.cat([drug1_feat, drug2_feat, express], 1)
        # drug_feat = (drug1_feat + drug2_feat) / 2
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out


class SynSyn(nn.Module):

    def __init__(self, drug_size: int, cell_size: int, hidden_size: int):
        super(SynSyn, self).__init__()

        self.drug_proj = nn.Linear(drug_size, drug_size)
        self.cell_proj = nn.Linear(cell_size, cell_size)
        self.network = nn.Sequential(
            nn.Linear(drug_size + cell_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        d1 = self.drug_proj(drug1_feat)
        d2 = self.drug_proj(drug2_feat)
        d = d1.mul(d2)
        c = self.cell_proj(cell_feat)
        feat = torch.cat([d, c], 1)
        out = self.network(feat)
        return out


class PPIDNN(nn.Module):

    def __init__(self, drug_size: int, cell_size: int, hidden_size: int, emb_size: int):
        super(PPIDNN, self).__init__()
        self.conv = nn.Sequential(
            BottleneckLayer(emb_size, 64),
            BottleneckLayer(64, 128),
            BottleneckLayer(128, 64),
            BottleneckLayer(64, 1),
        )
        self.network = nn.Sequential(
            nn.Linear(2 * drug_size + cell_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        cell_feat = cell_feat.permute(0, 2, 1)
        cell_feat = self.conv(cell_feat).squeeze(1)
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out


class StackLinearDNN(nn.Module):

    def __init__(self, input_size: int, stack_size: int, hidden_size: int):
        super(StackLinearDNN, self).__init__()

        self.compress = nn.Parameter(torch.zeros(size=(1, stack_size)))
        nn.init.xavier_uniform_(self.compress.data, gain=1.414)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        cell_feat = torch.matmul(self.compress, cell_feat).squeeze(1)
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out


class InteractionNet(nn.Module):

    def __init__(self, drug_size: int, cell_size: int, hidden_size: int):
        super(InteractionNet, self).__init__()

        # self.compress = nn.Parameter(torch.ones(size=(1, stack_size)))
        # self.drug_proj = nn.Sequential(
        #     nn.Linear(drug_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(hidden_size)
        # )
        self.inter_net = nn.Sequential(
            nn.Linear(drug_size + cell_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size)
        )

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        # cell_feat = torch.mat
        # mul(self.compress, cell_feat).squeeze(1)
        # d1 = self.drug_proj(drug1_feat)
        # d2 = self.drug_proj(drug2_feat)
        dc1 = torch.cat([drug1_feat, cell_feat], 1)
        dc2 = torch.cat([drug2_feat, cell_feat], 1)
        inter1 = self.inter_net(dc1)
        inter2 = self.inter_net(dc2)
        inter3 = inter1 + inter2
        out = self.network(inter3)
        return out


class StackProjDNN(nn.Module):

    def __init__(self, drug_size: int, cell_size: int, stack_size: int, hidden_size: int):
        super(StackProjDNN, self).__init__()

        self.projectors = nn.Parameter(torch.zeros(size=(stack_size, cell_size, cell_size)))
        nn.init.xavier_uniform_(self.projectors.data, gain=1.414)

        self.network = nn.Sequential(
            nn.Linear(2 * drug_size + cell_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        cell_feat = cell_feat.unsqueeze(-1)
        cell_feats = torch.matmul(self.projectors, cell_feat).squeeze(-1)
        cell_feat = torch.sum(cell_feats, 1)
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out
