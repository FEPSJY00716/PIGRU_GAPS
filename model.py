import torch
import torch.nn as nn
import torchdiffeq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)

        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)

        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, 4)

        self.init_weights()
        self.bn1 = nn.BatchNorm1d(1)

    def init_weights(self):
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.normal_(module.bias, mean=0.0, std=0.01)

    def forward(self, input, max, min, hidden=None):
        input = input.unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(input.size(0), 1, self.hidden_size).to(device)

        reset_gate = self.sigmoid(self.Wr(input) + self.Ur(hidden))
        update_gate = self.sigmoid(self.Wz(input) + self.Uz(hidden))
        current_state_proposal = self.bn1(self.Wh(input) + reset_gate * self.Uh(hidden))
        current_state_proposal = self.relu(current_state_proposal)
        current_state = (1 - update_gate) * hidden + update_gate * current_state_proposal
        out = self.out(current_state)

        Irad_future, Tout_future, Hair_future, Tair_future = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

        return current_state, Irad_future, Tout_future, Hair_future, Tair_future

class GRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=8, number_feature=12, ccap=1000, c_l=2450.0, lai=2.6, r_b=150.0, c_r=0.0001,
                 g_v=0.3, p_air=1.225, c_p_air=1005.0, t_tot=0.7, a2=5.0, t_step=300):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.ModuleList([GRUCell(input_size, hidden_size) for _ in range(num_layers)])
        self.ccap = ccap
        self.c_l = c_l
        self.lai = lai
        self.r_b = r_b
        self.c_r = c_r
        self.g_v = g_v
        self.p_air = p_air
        self.c_p_air = c_p_air
        self.t_tot = t_tot
        self.a2 = a2
        self.t_step = t_step
        self.feature = number_feature
        self.bn1 = nn.BatchNorm1d(12)

    def forward(self, input_seq, max, min):
        batch_size, seq_len, input_size = input_seq.size()
        hidden = None
        for gru_cell in self.gru_cell:
            Irad_future = []
            Tout_future = []
            Hair_future = []
            Tair_future = []
            for i in range(seq_len):
                hidden, Irad_future_i, Tout_future_i, Hair_future_i, Tair_future_i = gru_cell(input_seq[:, i, :], max, min, hidden)
                Irad_future.append(Irad_future_i)
                Tout_future.append(Tout_future_i)
                Hair_future.append(Hair_future_i)
                Tair_future.append(Tair_future_i)

        Irad_future = torch.cat(Irad_future, dim=1)
        Tout_future = torch.cat(Tout_future, dim=1)
        Hair_future = torch.cat(Hair_future, dim=1)
        Tair_future = torch.cat(Tair_future, dim=1)

        Irad_max, Tout_max, Hair_max, Tair_max = max[0], max[1], max[2], max[3]
        Irad_min, Tout_min, Hair_min, Tair_min = min[0], min[1], min[2], min[3]
        Irad_future, Tout_future, Hair_future, Tair_future = Irad_future * (Irad_max - Irad_min) + Irad_min, Tout_future * (Tout_max - Tout_min) + Tout_min, Hair_future * (
            Hair_max - Hair_min) + Hair_min, Tair_future * (Tair_max - Tair_min) + Tair_min

        def dydt(t, Tair):
            Tair = torch.clamp(Tair, max=500.0)
            Qsun = 10 * self.t_tot * Irad_future
            Qcov = self.a2 * (Tair - Tout_future)
            Hair_sat = 5.5638 * torch.exp(0.0572 * Tair)
            c_e = 0.7584 * torch.exp(0.0518 * Tair)
            Rn = 0.86 * (1 - torch.exp(-0.7 * torch.tensor(self.lai))) * Qsun
            r_s = (82 + 570 * torch.exp(-self.c_r * (Rn / self.lai))) * (1 + 0.023 * (Tair - 5) ** 2)
            g_e = 2 * self.lai / ((1 + c_e) * self.r_b + r_s)
            Hcrop = Hair_sat + ((c_e * self.r_b / (2 * self.lai)) * (Rn / self.c_l))
            Qtrans = g_e * -self.c_l * (Hcrop - Hair_future)
            Qvent = -self.g_v * -self.p_air * -self.c_p_air * (Tair - Tout_future)
            return (1 / -self.ccap) * (Qsun - Qcov - Qtrans - Qvent)

        t = torch.linspace(0., 60, 5).to(device)
        Tair_future_sol = torchdiffeq.odeint(dydt, Tair_future, t, method='dopri5')
        Tair_future = Tair_future_sol[-1]

        out = torch.cat((Irad_future.unsqueeze(2), Tout_future.unsqueeze(2), Hair_future.unsqueeze(2)), 2)
        return Tair_future, out, Tair_future
