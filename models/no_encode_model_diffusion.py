import math
import torch
import torch.nn as nn
from torch.nn import Module
from data.dataloader_ngsim import outputActivation

from models.layers import PositionalEncoding, ConcatSquashLinear


class TransformerDenoisingModel(Module):
    def __init__(self, context_dim=96, tf_layer=2):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        # self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim)
        self.layer = nn.TransformerEncoderLayer(d_model=2 * context_dim, nhead=2, dim_feedforward=2 * context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)
        self.enc_lstm = torch.nn.LSTM(32, 64, 1)
        # self.dec_lstm = torch.nn.LSTM(96, 256)
        self.dec_lstm = torch.nn.LSTM(96, 256)
        self.op = torch.nn.Linear(256, 5)
        self.dyn_emb = torch.nn.Linear(64, 32)
        self.ip_emb = torch.nn.Linear(2, 32)
        self.linear1 = torch.nn.Linear(32+64, 1)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        # self.dec_lstm = torch.nn.LSTM(512, 128)

    def forward(self, x, beta, context, nbrs, mask):
        # beta = beta.view(beta.size(0), 1, 1)
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(context)))
        hist_enc_one = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        # Masked scatter
        soc_enc = torch.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], -1, soc_enc.shape[3])

        # Mask for attention
        mask_enc = mask.contiguous().view(mask.shape[0], -1, mask.shape[3])
        mask_enc = mask_enc[:, :, 0].bool()

        # Attention
        hist_enc = hist_enc_one.unsqueeze(1).tile(1, soc_enc.size(1), 1)
        new_hs = torch.cat((hist_enc, soc_enc), dim=2)
        e = self.linear1(self.tanh(new_hs)).squeeze(2)
        e = e.masked_fill(mask_enc == 0, -1e9)
        # e_array = e.detach().cpu().numpy()
        attn = self.softmax(e)
        # attn_array = attn.detach().cpu().numpy()
        enc = torch.bmm(attn.unsqueeze(1), soc_enc)
        # b, 96
        context = torch.cat((hist_enc_one, enc.squeeze(1)), dim=1)
        #### beta , gussian over
        return self.decode(context)
        # return self.beta_decode(final_emb)
        # trans = None
        # if len(final_emb.shape) == 3:
        #     trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 24, 512)
        # elif len(final_emb.shape) == 4:
        #     trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 24, 24, 512)
        # trans = self.concat3(ctx_emb, trans)
        # trans = self.concat4(ctx_emb, trans)
        # return self.linear(ctx_emb, trans)

    def decode(self, enc):
        # 128, 96
        enc = enc.repeat(24, 1, 1)
        enc = enc.float()
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred.permute(1, 0, 2)

    def beta_decode(self, enc):
        # 24, 128, 192
        # enc = enc.repeat(24, 1, 1)
        enc = enc.float()
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred.permute(1, 0, 2)

    def generate_accelerate(self, x, beta, context, nbrs, mask):
        beta = beta.view(beta.size(0), 1, 1)  # (B, 1, 1)
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(context)))
        hist_enc_one = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        # Masked scatter
        soc_enc = torch.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], -1, soc_enc.shape[3])

        # Mask for attention
        mask_enc = mask.contiguous().view(mask.shape[0], -1, mask.shape[3])
        mask_enc = mask_enc[:, :, 0].bool()

        # Attention
        hist_enc = hist_enc_one.unsqueeze(1).tile(1, soc_enc.size(1), 1)
        new_hs = torch.cat((hist_enc, soc_enc), dim=2)
        e = self.linear1(self.tanh(new_hs)).squeeze(2)
        e = e.masked_fill(mask_enc == 0, -1e9)
        attn = self.softmax(e)
        enc = torch.bmm(attn.unsqueeze(1), soc_enc)
        context = torch.cat((hist_enc_one, enc.squeeze(1)), dim=1)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        # time_emb: [11, 1, 3]
        # context: [11, 1, 256]
        ctx_emb = torch.cat([time_emb, context.unsqueeze(1)], dim=-1).repeat(1, 12, 1).unsqueeze(2)
        # x: 11, 10, 20, 2
        # ctx_emb: 11, 10, 1, 259
        x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, 24, 512)
        # x: 110, 20, 512
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 12, 24, 512)
        # trans: 11, 10, 20, 512
        trans = self.concat3.batch_generate(ctx_emb, trans)
        trans = self.concat4.batch_generate(ctx_emb, trans)
        return self.linear.batch_generate(ctx_emb, trans)
