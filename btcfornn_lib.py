# bitcoin NN forecasting library

import warnings
import pandas as pd
import numpy as np
#import missingno
import matplotlib.pyplot as plt
#mport seaborn as sns
from statsmodels.tsa.stattools import adfuller
#from datetime import date
#from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import random
#from sklearn.inspection import permutation_importance
#import pywt

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#########################
# PROCEDURES
#########################

def check_stationarity(series, name):
    adf_result = adfuller(series.dropna())
    print(f"ADF Test for {name}:")
    print(f"  Test Statistic: {adf_result[0]:.4f}")
    print(f"  P-value: {adf_result[1]:.4f}")
    print("  Critical Values:")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.4f}")
    if adf_result[1] <= 0.05:
        print("=> Data is stationary (reject null hypothesis)\n")
    else:
        print("=> Data is NOT stationary (fail to reject null hypothesis)\n")
        
def Pause_Code():
    import sys
    print()
    print()
    stop = input('Type Y to go ahead, N to stop code: ')
    if stop == 'N':
        sys.exit()
        
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# PyTorch Model definitions - ORIGINAL MODELS
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        if self.dropout:
            lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # Take the last time step
        if self.dropout:
            gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)
        return output

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        # Bidirectional LSTM outputs 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = bilstm_out[:, -1, :]  # Take the last time step
        if self.dropout:
            bilstm_out = self.dropout(bilstm_out)
        output = self.fc(bilstm_out)
        return output

# NEW MODELS WITH ATTENTION MECHANISM
class Attention(nn.Module):
    """Bahdanau Attention Mechanism"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        # Repeat hidden for each time step
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.W(encoder_outputs) + self.U(hidden))
        attention_scores = self.v(energy).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to encoder outputs
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # [batch_size, hidden_size]
        
        return context_vector, attention_weights

class LSTMAttentionModel(nn.Module):
    """LSTM with Bahdanau Attention"""
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM returns output for each time step
        lstm_out, (hidden, _) = self.lstm(x)
        # hidden: [1, batch_size, hidden_size]
        hidden = hidden[-1]  # Take the last layer's hidden state
        
        # Apply attention
        context_vector, attention_weights = self.attention(hidden, lstm_out)
        
        if self.dropout:
            context_vector = self.dropout(context_vector)
        
        output = self.fc(context_vector)
        return output

class GRUAttentionModel(nn.Module):
    """GRU with Bahdanau Attention"""
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        # hidden: [1, batch_size, hidden_size]
        hidden = hidden[-1]  # Take the last layer's hidden state
        
        # Apply attention
        context_vector, attention_weights = self.attention(hidden, gru_out)
        
        if self.dropout:
            context_vector = self.dropout(context_vector)
        
        output = self.fc(context_vector)
        return output

class BiLSTMAttentionModel(nn.Module):
    """Bidirectional LSTM with Attention"""
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0):
        super(BiLSTMAttentionModel, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)  # Bidirectional has double hidden size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        bilstm_out, (hidden, _) = self.bilstm(x)
        # For bidirectional, we concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch_size, hidden_size*2]
        
        # Apply attention
        context_vector, attention_weights = self.attention(hidden, bilstm_out)
        
        if self.dropout:
            context_vector = self.dropout(context_vector)
        
        output = self.fc(context_vector)
        return output


# class SelfAttentionModel(nn.Module):
#     """Self-Attention Mechanism (like Transformer)"""
#     def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0, num_heads=4):
#         super(SelfAttentionModel, self).__init__()
#         # For older PyTorch versions, we need to handle batch dimension manually
#         self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads,
#                                               dropout=dropout)
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else None
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.num_heads = num_heads
#         self.input_size = input_size
       
#     def forward(self, x):
#         # For older PyTorch versions, MultiheadAttention expects [seq_len, batch_size, embed_dim]
#         # So we need to transpose from [batch_size, seq_len, embed_dim]
#         x_transposed = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
       
#         # Self-attention
#         attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
       
#         # Transpose back to [batch_size, seq_len, embed_dim]
#         attn_output = attn_output.transpose(0, 1)
       
#         # LSTM on attention output
#         lstm_out, _ = self.lstm(attn_output)
#         lstm_out = lstm_out[:, -1, :]  # Take the last time step
       
#         if self.dropout:
#             lstm_out = self.dropout(lstm_out)
       
#         output = self.fc(lstm_out)
#         return output

class SelfAttentionModel(nn.Module):
    """Self-Attention Mechanism (like Transformer)"""
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.0, num_heads=4):
        super(SelfAttentionModel, self).__init__()
        
        # Adjust num_heads to be compatible with input_size
        # Find the largest divisor of input_size that's <= num_heads
        possible_heads = [h for h in range(1, min(num_heads, input_size) + 1) 
                         if input_size % h == 0]
        
        if not possible_heads:
            # If no divisor found, pad the input to make it divisible
            self.pad_needed = True
            self.pad_size = (num_heads - (input_size % num_heads)) % num_heads
            self.padding = nn.ConstantPad1d((0, self.pad_size), 0)
            effective_input_size = input_size + self.pad_size
            self.attention = nn.MultiheadAttention(embed_dim=effective_input_size, 
                                                  num_heads=num_heads,
                                                  dropout=dropout, batch_first=False)
        else:
            self.pad_needed = False
            # Use the largest compatible number of heads
            actual_heads = possible_heads[-1]
            if actual_heads != num_heads:
                print(f"Adjusting num_heads from {num_heads} to {actual_heads} "
                      f"to be compatible with input_size={input_size}")
            self.attention = nn.MultiheadAttention(embed_dim=input_size, 
                                                  num_heads=actual_heads,
                                                  dropout=dropout, batch_first=False)
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_heads = num_heads
        self.input_size = input_size
       
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        
        # Pad if necessary
        if hasattr(self, 'pad_needed') and self.pad_needed:
            x_padded = self.padding(x.transpose(1, 2)).transpose(1, 2)
            x_transposed = x_padded.transpose(0, 1)
        else:
            x_transposed = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        # Self-attention
        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
        
        # If we padded, remove the padding before LSTM
        if hasattr(self, 'pad_needed') and self.pad_needed:
            attn_output = attn_output[:, :, :self.input_size]
            attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        else:
            attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # LSTM on attention output
        lstm_out, _ = self.lstm(attn_output)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
       
        if self.dropout:
            lstm_out = self.dropout(lstm_out)
       
        output = self.fc(lstm_out)
        return output


# Model builders

def build_lstm_model(input_shape, net_units=50, act_f='relu', dropout=0, 
                     for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = LSTMModel(input_size=input_size, hidden_size=net_units, 
                      output_size=for_steps, dropout=dropout)
    return model

def build_gru_model(input_shape, net_units=50, act_f='relu', dropout=0,
                    for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = GRUModel(input_size=input_size, hidden_size=net_units,
                     output_size=for_steps, dropout=dropout)
    return model

def build_bilstm_model(input_shape, net_units=50, act_f='relu', dropout=0,
                       for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = BiLSTMModel(input_size=input_size, hidden_size=net_units,
                        output_size=for_steps, dropout=dropout)
    return model

# NEW MODEL BUILDERS WITH ATTENTION
def build_lstm_attention_model(input_shape, net_units=50, act_f='relu', dropout=0, 
                               for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = LSTMAttentionModel(input_size=input_size, hidden_size=net_units, 
                               output_size=for_steps, dropout=dropout)
    return model

def build_gru_attention_model(input_shape, net_units=50, act_f='relu', dropout=0,
                              for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = GRUAttentionModel(input_size=input_size, hidden_size=net_units,
                              output_size=for_steps, dropout=dropout)
    return model

def build_bilstm_attention_model(input_shape, net_units=50, act_f='relu', dropout=0,
                                 for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = BiLSTMAttentionModel(input_size=input_size, hidden_size=net_units,
                                 output_size=for_steps, dropout=dropout)
    return model

def build_self_attention_model(input_shape, net_units=50, act_f='relu', dropout=0,
                               for_steps=1, opt_rate=0.001, opt_clip=None, metric='mse'):
    input_size = input_shape[-1]  # Number of features
    model = SelfAttentionModel(input_size=input_size, hidden_size=net_units,
                               output_size=for_steps, dropout=dropout, num_heads=4)
    return model

def prepare_lstm_input(df, 
                       target, 
                       lags, 
                       n_steps=1, 
                       target_type='price',
                       flagDIR='Y'):
    """
    Prepare data for LSTM. Supports direct (T+n_steps) or rolling (T+1) forecasting.
    Now supports both price and logreturns as target.
    """
    sequences = []
    targets = []
   
    # Make a copy to avoid modifying the original
    df_modified = df.copy()
   
    # Remove MKPRU_logreturns column if it exists (we don't want it as a feature)
    if 'MKPRU_logreturns' in df_modified.columns:
        df_modified = df_modified.drop(columns=['MKPRU_logreturns'])
   
    if target_type == 'logreturns':
        # Calculate log returns for target variable
        df_target = np.log(df_modified[target] / df_modified[target].shift(1))
        df_target = df_target.fillna(0)  # Fill first NaN with 0
        # Replace target column with logreturns
        df_modified[target] = df_target
   
    if flagDIR == 'Y':
        # Direct forecasting: target = T + n_steps
        max_index = len(df_modified) - n_steps
        for i in range(lags, max_index + 1):
            seq = df_modified.iloc[i - lags:i].drop(columns=[target]).values
            target_val = df_modified.iloc[i + n_steps - 1][target]
            sequences.append(seq)
            targets.append(target_val)
    else:
        # Rolling forecasting: target = T+1
        for i in range(lags, len(df_modified)):
            seq = df_modified.iloc[i - lags:i].drop(columns=[target]).values
            target_val = df_modified.iloc[i][target]
            sequences.append(seq)
            targets.append(target_val)
   
    X = np.array(sequences)
    y = np.array(targets)
   
    return X, y


def convert_predictions_to_price(logreturn_pred, initial_prices):
    """
    Convert log return predictions back to price predictions.
    logreturn_pred: predicted log returns
    initial_prices: actual prices at time t-1 for each prediction
    """
    # Log returns to price: P_t = P_{t-1} * exp(r_t)
    price_pred = initial_prices * np.exp(logreturn_pred)
    return price_pred

def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=100, 
                        batch_size=64, learning_rate=0.001, patience=15, device='cpu'):
    """
    Train a PyTorch model with early stopping.
    """
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor).item()
            val_losses.append(val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def predict_pytorch_model(model, X, device='cpu'):
    """
    Make predictions with PyTorch model.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()

def rolling_forecast_aligned(model, X_scaled, scaler_y, n_steps, device='cpu'):
    """
    Perform a rolling N-step-ahead forecast, ensuring the output is aligned
    with the test (or train) set size.
    """
    # Initialize predictions with NaNs to align with the input size
    predictions_scaled = np.full(len(X_scaled), np.nan)
    
    # Convert to tensor
    X_scaled_tensor = torch.FloatTensor(X_scaled).to(device)
    model.eval()
    
    # Rolling forecast loop
    with torch.no_grad():
        for i in range(len(X_scaled) - n_steps):
            # Take the input window
            X_input = X_scaled_tensor[i:i+1]
            
            # Predict n_steps ahead
            y_pred_scaled = model(X_input)
            
            # Store the prediction in the aligned array
            predictions_scaled[i + n_steps] = y_pred_scaled.cpu().numpy()[:, -1]
    
    # Inverse transform the predictions
    predictions_scaled = predictions_scaled.reshape(-1, 1)  # Reshape for inverse transform
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # Return aligned predictions as a Pandas Series
    pred = pd.Series(predictions.flatten(), index=range(len(X_scaled)))
    pred_scaled = pd.Series(predictions_scaled.flatten(), index=range(len(X_scaled)))
    return pred, pred_scaled

def compute_feature_importance_lstm(model, X, y, feature_names, metric_function, 
                                   n_repeats=5, device='cpu'):
    """
    Compute feature importance for an LSTM model using permutation importance.
    """
    print()
    print('Features Importances Calculation ...')
    print()
    
    # Get baseline prediction
    baseline_pred = predict_pytorch_model(model, X, device).flatten()
    baseline_error = metric_function(y.flatten(), baseline_pred)
    
    importances = np.zeros(X.shape[2])
    
    for feature_idx in range(X.shape[2]):  # Loop over features
        temp_error = []
        
        for _ in range(n_repeats):  # Repeat permutation multiple times
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, feature_idx])  # Shuffle one feature
            
            permuted_pred = predict_pytorch_model(model, X_permuted, device).flatten()
            error = metric_function(y.flatten(), permuted_pred)
            temp_error.append(error)
        
        importances[feature_idx] = np.mean(temp_error) - baseline_error
    
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    
    return importance_df

def compute_feature_importance_weights_pytorch(model, feature_names):
    """
    Compute feature importance using PyTorch model weights.
    """
    # Get the first recurrent layer weights
    for name, param in model.named_parameters():
        if 'lstm.weight_ih_l0' in name or 'gru.weight_ih_l0' in name:
            weights = param.data.cpu().numpy()
            break
    else:
        # For BiLSTM
        for name, param in model.named_parameters():
            if 'bilstm.weight_ih_l0' in name:
                weights = param.data.cpu().numpy()
                break
        else:
            raise ValueError("Could not find recurrent layer weights")
    
    # Calculate absolute weights across all gates
    abs_weights = np.mean(np.abs(weights), axis=1)
    
    # Normalize to percentage
    importance = 100 * (abs_weights / abs_weights.sum())
    
    # Match length with feature names (might need adjustment for multi-lag)
    importance = importance[:len(feature_names)]
    
    importance_df = pd.DataFrame({
        "Feature": feature_names[:len(importance)],
        "Importance": importance
    }).sort_values("Importance", ascending=False)
    
    return importance_df

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def trading_simulation(predictions, close_prices, initial_investment, transaction_cost, dates, 
                       mode="additive", entry_threshold=0.01, exit_threshold=-0.01):
    """
    Trading strategy that considers transaction costs and dynamic buy/sell thresholds.
    """
    balance = initial_investment
    position = 0  
    equity_curve = []
    initial_buy_done = False  

    for i in range(len(predictions)):
        predicted_return = (predictions[i] - close_prices.iloc[i]) / close_prices.iloc[i]
        price = close_prices.iloc[i]

        if predicted_return > entry_threshold:
            if mode == "additive" or (mode == "hold_position" and not initial_buy_done):
                shares_to_buy = balance // (price * (1 + transaction_cost))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + transaction_cost)
                    balance -= cost
                    position += shares_to_buy
                    initial_buy_done = True
        
        if predicted_return < exit_threshold and position > 0:
            proceeds = position * price * (1 - transaction_cost)
            balance += proceeds
            position = 0
            initial_buy_done = False  

        equity_curve.append(balance + position * price)

    equity_curve = pd.Series(equity_curve, index=dates)
    
    returns = equity_curve.pct_change()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  
    max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()
    final_value = equity_curve.iloc[-1]
    years = len(equity_curve) / 252  
    cagr = (final_value / initial_investment) ** (1 / years) - 1

    return equity_curve, sharpe_ratio, max_drawdown, final_value, cagr

def add_seasonal_features(df,
                          flagM,
                          flagW,
                          flagD):
    """Add time-based seasonal features to dataframe"""
    df = df.copy()
    
    if flagM == 'Y':
        df['month'] = df.index.month
        df = pd.get_dummies(df, columns=['month'], prefix='month')
        
    if flagW == 'Y':
        df['week'] = df.index.isocalendar().week
        df = pd.get_dummies(df, columns=['week'], prefix='week')
        
    if flagD == 'Y':
        df['dayofweek'] = df.index.dayofweek
        df = pd.get_dummies(df, columns=['dayofweek'], prefix='dow')
    
    return df

def add_halving_features(df):
    """Add days since last Bitcoin halving as feature"""
    halvings = {
        1: "2012-11-28",
        2: "2016-07-09",  # Corrected date for 2nd halving
        3: "2020-05-11",
        4: "2024-04-27"
    }
    
    df = df.copy()
    temp_df = df.reset_index()[['date']].copy()
    
    # Calculate days difference for each halving
    halving_cols = []
    for key, date_str in halvings.items():
        halving_date = pd.to_datetime(date_str)
        col_name = f'days_since_halving_{key}'
        temp_df[col_name] = (temp_df['date'] - halving_date).dt.days
        temp_df[col_name] = np.where(temp_df[col_name] < 0, np.nan, temp_df[col_name])
        halving_cols.append(col_name)
    
    # Find nearest future halving
    temp_df["days_since_last_halving"] = temp_df[halving_cols].min(axis=1, skipna=True)
    df["days_since_last_halving"] = temp_df["days_since_last_halving"].values
    
    # Forward fill missing values (for dates before first halving)
    df["days_since_last_halving"].ffill(inplace=True)
    
    return df

# def adaptive_wavelet_filter(series):
#     """Denoise series using level-adaptive wavelet thresholding"""
#     # Auto-determine decomposition level
#     max_level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet_type).dec_len)
#     coeffs = pywt.wavedec(series, wavelet_type, level=max_level)
    
#     # Calculate energy distribution
#     energy = [np.sum(np.square(c)) for c in coeffs]
#     total_energy = np.sum(energy)
    
#     # Find optimal level (keep 99% energy)
#     cum_energy = np.cumsum(energy[::-1])[::-1]  # Reverse cumulative sum
#     opt_level = np.argmax(cum_energy > 0.99 * total_energy) 
    
#     # Threshold detail coefficients (skip approximation)
#     threshold = np.sqrt(2 * np.log(len(series))) * np.median(np.abs(coeffs[-1])) / 0.6745
#     coeffs_th = [coeffs[0]]  # Keep approximation
#     for i in range(1, opt_level + 1):
#         coeffs_th.append(pywt.threshold(coeffs[i], threshold, mode=threshold_type))
    
#     # Reconstruct signal
#     return pywt.waverec(coeffs_th, wavelet_type)[:len(series)]

# def wavelet_filter_features(df, target='MKPRU'):
#     """Apply wavelet filtering to all features except target"""
#     df_filtered = df.copy()
#     for col in df.columns:
#         if col != target:
#             clean_series = adaptive_wavelet_filter(df[col].values)
#             df_filtered[col] = clean_series
#     return df_filtered