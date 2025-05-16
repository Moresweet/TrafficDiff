import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder
from data.dataloader_ngsim import outputActivation


class LEDInitializer(nn.Module):
    def __init__(self, t_h: int = 8, d_h: int = 6, t_f: int = 40, d_f: int = 2, k_pred: int = 20):
        '''
        Parameters
        ----
        t_h: history timestamps,
        d_h: dimension of each historical timestamp,
        t_f: future timestamps,
        d_f: dimension of each future timestamp,
        k_pred: number of predictions.

        '''
        super(LEDInitializer, self).__init__()
        self.n = k_pred
        self.input_dim = t_h * d_h
        self.output_dim = t_f * d_f * k_pred
        self.fut_len = t_f

        self.enc_lstm = torch.nn.LSTM(32, 128, 1)
        self.dyn_emb = torch.nn.Linear(128, 128)
        self.ip_emb = torch.nn.Linear(2, 32)
        self.linear1 = torch.nn.Linear(256, 1)
        self.ego_var_encoder = torch.nn.LSTM(32, 256, 1)
        self.ego_mean_encoder = torch.nn.LSTM(32, 256, 1)
        self.ego_scale_encoder = torch.nn.LSTM(32, 256, 1)
        self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())
        self.ego_nbr_emb = torch.nn.Linear(256, 256)
        # Output layers:
        # self.op_mean = torch.nn.Linear(self.decoder_size, 5)
        # # Output layers:
        # self.op_var = torch.nn.Linear(self.decoder_size, 5)
        # # Output layers:
        # self.op_scale = torch.nn.Linear(self.decoder_size, t_f * d_f * k_pred)

        # self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())
        #
        self.o_var_decoder = MLP(128, t_f * d_f * k_pred, hid_feat=(1024, 1024), activation=nn.ReLU())
        self.o_mean_decoder = MLP(128, 2, hid_feat=(256, 128), activation=nn.ReLU())
        self.o_scale_decoder = MLP(128, 1, hid_feat=(256, 128), activation=nn.ReLU())
        # self.scale_encoder = torch.nn.LSTM(1, 32)
        self.var_decoder = torch.nn.LSTM(256 * 2 + 32, 128)
        self.mean_decoder = torch.nn.LSTM(256 * 2, 128)
        self.scale_decoder = torch.nn.LSTM(256 * 2, 128)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, nbrs, mask=None):
        '''
        x: batch size, t_p, 6
        '''
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # social_embed = self.social_encoder(x, mask)
        # social_embed = social_embed.squeeze(1)
        ## Forward pass hist:
        _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(x)))
        hist_enc_one = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        # Masked scatter
        soc_enc = torch.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], -1, soc_enc.shape[3])
        # soc_enc_array = soc_enc.detach().cpu().numpy()

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
        social_embed = torch.cat((hist_enc_one, enc.squeeze(1)), dim=1)
        # B, 256
        _, (var_enc, _) = self.ego_var_encoder(self.leaky_relu(self.ip_emb(x)))
        ego_var_embed = self.leaky_relu(self.ego_nbr_emb(var_enc.view(var_enc.shape[1], var_enc.shape[2])))
        # ego_var_embed = self.ego_var_encoder(x)
        _, (mean_enc, _) = self.ego_mean_encoder(self.leaky_relu(self.ip_emb(x)))
        ego_mean_embed = self.leaky_relu(self.ego_nbr_emb(mean_enc.view(mean_enc.shape[1], mean_enc.shape[2])))
        # ego_mean_embed = self.ego_mean_encoder(x)
        _, (scale_enc, _) = self.ego_scale_encoder(self.leaky_relu(self.ip_emb(x)))
        ego_scale_embed = self.leaky_relu(self.ego_nbr_emb(scale_enc.view(scale_enc.shape[1], scale_enc.shape[2])))
        # ego_scale_embed = self.ego_scale_encoder(x)
        # B, 256
        mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
        guess_mean = self.mean_decode_layer(mean_total)
        scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
        # guess_scale = self.scale_decoder(scale_total)
        guess_scale = self.scale_decode_layer(scale_total)

        guess_scale_feat = self.scale_encoder(guess_scale)
        guess_scale_feat = guess_scale_feat.squeeze(1)
        var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
        # guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)
        guess_var = self.var_decode_layer(var_total)

        return guess_var, guess_mean, guess_scale.squeeze(1)

    def mean_decode_layer(self, enc):
        enc = enc.repeat(self.fut_len, 1, 1)
        enc = enc.float()
        h_dec, _ = self.mean_decoder(enc)
        h_dec = self.o_mean_decoder(h_dec)
        # h_dec = h_dec.permute(1, 0, 2)
        fut_pred = h_dec.contiguous().view(-1, self.fut_len, 2)
        return fut_pred

    def var_decode_layer(self, enc):
        enc = enc.repeat(1, 1, 1)
        enc = enc.float()
        h_dec, _ = self.var_decoder(enc)
        h_dec = self.o_var_decoder(h_dec)
        var = h_dec.reshape(enc.size(1), self.n, self.fut_len, 2)
        return var

    def scale_decode_layer(self, enc):
        enc = enc.repeat(1, 1, 1)
        enc = enc.float()
        h_dec, _ = self.scale_decoder(enc)
        h_dec = self.o_scale_decoder(h_dec)
        h_dec = h_dec.permute(1, 0, 2)
        return h_dec
