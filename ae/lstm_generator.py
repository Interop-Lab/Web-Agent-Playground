import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from har_analyze import har_analyze
import matplotlib.pyplot as plt
from matplotlib import font_manager

torch.manual_seed(0)

class ScientificWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_mse = nn.Parameter(torch.zeros(1))
        self.log_var_ce = nn.Parameter(torch.zeros(1))
        
    def forward(self, outputs, targets):
        mse_loss = F.mse_loss(outputs[:, :, 0], targets[:, :, 0])
        
        ce_targets = torch.argmax(targets[:, :, 1:], dim=2)
        ce_outputs = outputs[:, :, 1:].reshape(-1, 7)
        ce_loss = F.cross_entropy(ce_outputs, ce_targets.reshape(-1))
        
        total_loss = torch.exp(-self.log_var_mse) * mse_loss + self.log_var_mse + \
                     torch.exp(-self.log_var_ce) * ce_loss + self.log_var_ce
        
        return total_loss

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc_reg = nn.Linear(hidden_dim, 1)
        self.fc_cls = nn.Linear(hidden_dim, 7)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        encoded, (hn, cn) = self.encoder_lstm(x, (h0, c0))
        
        decoded, _ = self.decoder_lstm(encoded, (hn, cn))
        
        reg_output = self.fc_reg(decoded)
        cls_output = self.fc_cls(decoded)
        
        outputs = torch.cat([reg_output, cls_output], dim=2)
        
        return outputs, hn, cn

def train_autoencoder(model, dataloader, dataloader2, criterion, optimizer, num_epochs=100):
    train_losses = []
    test_losses = []
    mse_losses = []
    ce_losses = []
    weight_history = []
    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for data in dataloader:
            inputs = data[0]
            
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        model.eval()
        epoch_test_loss = 0
        epoch_mse = 0
        epoch_ce = 0
        with torch.no_grad():
            for data in dataloader2:
                inputs = data[0]
                outputs, _, _ = model(inputs)
                
                mse_loss = F.mse_loss(outputs[:, :, 0], inputs[:, :, 0]).item()
                ce_loss = F.cross_entropy(
                    outputs[:, :, 1:].reshape(-1,7),
                    torch.argmax(inputs[:, :, 1:],2).reshape(-1)
                ).item()
                
                epoch_mse += mse_loss
                epoch_ce += ce_loss
                epoch_test_loss += criterion(outputs, inputs).item()
                
        avg_test_loss = epoch_test_loss / len(dataloader2)
                
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            torch.save(model.encoder_lstm, 'ae/best_encoder.pth')
        
        current_weights = {
            'mse_weight': torch.exp(-criterion.log_var_mse).item(),
            'ce_weight': torch.exp(-criterion.log_var_ce).item()
        }
        weight_history.append(current_weights)
        
        train_losses.append(epoch_train_loss/len(dataloader))
        test_losses.append(epoch_test_loss/len(dataloader2))
        mse_losses.append(epoch_mse/len(dataloader2))
        ce_losses.append(epoch_ce/len(dataloader2))
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  总损失 - 训练: {train_losses[-1]:.4f} | 测试: {test_losses[-1]:.4f}')
            print(f'  MSE损失: {mse_losses[-1]:.4f} | CE损失: {ce_losses[-1]:.4f}')
            print(f'  自动权重 - MSE: {current_weights["mse_weight"]:.4f} | CE: {current_weights["ce_weight"]:.4f}')
            print('-' * 50)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] 当前测试损失: {avg_test_loss:.4f} | 最佳测试损失: {best_loss:.4f} (Epoch {best_epoch+1})')
    
    return model

def prepare_data(har_file):
    t, a = har_analyze(har_file)
    data = torch.zeros(1, len(t), 8)
    temp = 0
    for i in range(len(t)):
        data[0, i, 0] = t[i] - temp
        data[0, i, a[i]] = 1
        temp = t[i]
    return [(data,)]

if __name__ == "__main__":
    input_dim = 8
    hidden_dim = 64
    num_epochs = 1500
    learning_rate = 0.001
    
    dataloader = prepare_data('data/x_train.har')
    dataloader2 = prepare_data('data/x_test.har')
    
    model = LSTM_Autoencoder(input_dim, hidden_dim)
    criterion = ScientificWeightedLoss()
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': [criterion.log_var_mse, criterion.log_var_ce], 'lr': 0.01}
    ], lr=learning_rate)
    
    print("开始训练...")
    trained_model = train_autoencoder(model, dataloader, dataloader2, 
                                    criterion, optimizer, num_epochs)
    print("训练完成！")
    
    final_weights = {
        'MSE权重': torch.exp(-criterion.log_var_mse).item(),
        'CE权重': torch.exp(-criterion.log_var_ce).item()
    }
    print("\n最终自动学习到的权重:")
    for k, v in final_weights.items():
        print(f"{k}: {v:.4f}")
