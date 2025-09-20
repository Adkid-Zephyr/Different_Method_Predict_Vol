import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# 导入新的绘图库
import plotly.graph_objects as go
# 导入用于创建子图的模块
from plotly.subplots import make_subplots

# --- 1. 加载数据 ---
file_path = '50ETF_1min.csv'
df = pd.read_csv(file_path)

# --- 2. 基础数据清洗 ---
df['t'] = pd.to_datetime(df['t'])
df.set_index('t', inplace=True)

# --- 3. 计算分钟对数收益率 ---
df['log_return'] = np.log(df['close']).diff()

# --- 4. 计算日度已实现波动率 (RV) ---
def calculate_realized_volatility(x):
    x = x.dropna()
    return np.sqrt(np.sum(x**2))

daily_rv = df['log_return'].resample('D').apply(calculate_realized_volatility)
daily_rv = daily_rv[daily_rv > 1e-6]
daily_rv_df = pd.DataFrame(daily_rv)
daily_rv_df.columns = ['realized_volatility']

# --- 5. 创建HAR模型特征 ---
har_df = daily_rv_df.copy()
har_df['rv_d'] = har_df['realized_volatility'].shift(1)
har_df['rv_w'] = har_df['realized_volatility'].rolling(window=5).mean().shift(1)
har_df['rv_m'] = har_df['realized_volatility'].rolling(window=22).mean().shift(1)
har_df = har_df.dropna()

# --- 6. 准备训练和测试数据 (HAR) ---
X_har = har_df[['rv_d', 'rv_w', 'rv_m']]
y_har = har_df['realized_volatility']
X_train_har, X_test_har, y_train_har, y_test_har = train_test_split(
    X_har, y_har, test_size=0.2, shuffle=False
)

# --- 7. 训练HAR-RV模型 ---
har_model = LinearRegression()
har_model.fit(X_train_har, y_train_har)
print("HAR-RV模型训练完成。")

# --- 8. 准备训练和测试数据 (LSTM) ---
scaler = MinMaxScaler(feature_range=(0, 1))
rv_scaled = scaler.fit_transform(daily_rv_df[['realized_volatility']])
window_size = 10
X_lstm, y_lstm = [], []
for i in range(window_size, len(rv_scaled)):
    X_lstm.append(rv_scaled[i-window_size:i, 0])
    y_lstm.append(rv_scaled[i, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
split_index = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split_index], X_lstm[split_index:]
y_train_lstm, y_test_lstm = y_lstm[:split_index], y_lstm[split_index:]

# --- 9. 搭建和训练LSTM模型 ---
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
print("\n开始训练LSTM模型...")
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
print("LSTM模型训练完成。")

# --- 10. 生成并对齐预测结果 ---
har_predictions = pd.Series(har_model.predict(X_test_har), index=y_test_har.index)
predictions_lstm_scaled = lstm_model.predict(X_test_lstm)
predictions_lstm_unscaled = scaler.inverse_transform(predictions_lstm_scaled)
lstm_test_dates = daily_rv_df.index[-len(y_test_lstm):]
lstm_predictions = pd.Series(predictions_lstm_unscaled.flatten(), index=lstm_test_dates)

comparison_df = pd.DataFrame({
    'Actual': y_test_har,
    'HAR_Prediction': har_predictions
})
comparison_df = comparison_df.join(pd.DataFrame({'LSTM_Prediction': lstm_predictions}), how='inner')

# --- 11. 【原】使用Plotly创建合并的交互式图表 ---
print("\n正在创建合并的交互式图表...")
fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Actual'], mode='lines', name='Actual Volatility', line=dict(color='blue', width=2)))
fig_combined.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['HAR_Prediction'], mode='lines', name='HAR Prediction', line=dict(color='green', width=2, dash='dash')))
fig_combined.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['LSTM_Prediction'], mode='lines', name='LSTM Prediction', line=dict(color='red', width=2, dash='dot')))
fig_combined.update_layout(title='50ETF Volatility Prediction Comparison (Interactive)', xaxis_title='Date', yaxis_title='Realized Volatility', legend_title='Legend', template='plotly_white')
fig_combined.write_html("interactive_prediction_comparison.html")
# fig_combined.show() # 默认不显示，避免弹出两个窗口
print("\n合并的交互式图表已保存为 'interactive_prediction_comparison.html'")


# --- 12. 【新】创建堆叠式子图表 ---
print("\n正在创建堆叠式子图表...")

# 创建一个3行1列的子图画布，并共享X轴
fig_subplots = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.05,
                           subplot_titles=('HAR Prediction', 'LSTM Prediction', 'Actual Volatility'))

# 确定统一的Y轴范围
y_min = comparison_df.min().min() * 0.95
y_max = comparison_df.max().max() * 1.05
y_range = [y_min, y_max]

# 上图：HAR Prediction
fig_subplots.add_trace(go.Scatter(
    x=comparison_df.index, 
    y=comparison_df['HAR_Prediction'],
    mode='lines', name='HAR',
    line=dict(color='green')
), row=1, col=1)

# 中图：LSTM Prediction
fig_subplots.add_trace(go.Scatter(
    x=comparison_df.index, 
    y=comparison_df['LSTM_Prediction'],
    mode='lines', name='LSTM',
    line=dict(color='red')
), row=2, col=1)

# 下图：Actual Volatility
fig_subplots.add_trace(go.Scatter(
    x=comparison_df.index, 
    y=comparison_df['Actual'],
    mode='lines', name='Actual',
    line=dict(color='blue')
), row=3, col=1)

# 更新整体布局和所有Y轴的范围
fig_subplots.update_layout(
    title_text='Stacked Comparison of Volatility Predictions',
    height=800, # 增加图表高度以容纳3个子图
    showlegend=False, # 隐藏图例，因为子图标题已说明内容
    template='plotly_white'
)
# 统一所有子图的Y轴范围
fig_subplots.update_yaxes(range=y_range, row=1, col=1)
fig_subplots.update_yaxes(range=y_range, row=2, col=1)
fig_subplots.update_yaxes(range=y_range, row=3, col=1)


# 保存为HTML文件并在浏览器中显示
fig_subplots.write_html("subplots_prediction_comparison.html")
fig_subplots.show()

print("\n堆叠式交互图表已保存为 'subplots_prediction_comparison.html'")
