# 导入pandas库，用于数据处理和分析，特别是其DataFrame结构
import pandas as pd
# 导入numpy库，用于进行高效的数值计算，特别是数组操作
import numpy as np

# 从scikit-learn库中导入用于划分数据集的工具
from sklearn.model_selection import train_test_split
# 从scikit-learn库中导入线性回归模型
from sklearn.linear_model import LinearRegression
# 从scikit-learn库中导入用于评估模型性能的指标函数
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 从scikit-learn库中导入用于数据归一化的工具
from sklearn.preprocessing import MinMaxScaler

# 从tensorflow.keras中导入用于搭建神经网络序列模型的工具
from tensorflow.keras.models import Sequential
# 从tensorflow.keras中导入LSTM层和全连接层
from tensorflow.keras.layers import LSTM, Dense

# 导入plotly库，用于创建交互式图表
import plotly.graph_objects as go
# 从plotly库中导入用于创建子图的工具
from plotly.subplots import make_subplots

# ==============================================================================
# 评估指标计算函数 (Evaluation Metric Functions)
# ==============================================================================

def calculate_mape(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE - Mean Absolute Percentage Error)。
    这个指标衡量的是预测误差占真实值的平均百分比，结果更直观。
    """
    # 将输入的真实值和预测值转换为numpy数组，方便计算
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 创建一个布尔掩码，标记出真实值不为零的位置，以避免计算中出现除以零的错误
    non_zero_mask = y_true != 0
    # 仅对真实值不为零的数据点计算百分比误差，然后求平均值，最后乘以100得到百分数
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """
    计算方向准确率 (DA - Directional Accuracy)。
    这个指标衡量模型预测第二天波动率“上涨”或“下跌”这个方向的准确度。
    """
    # 使用np.diff计算真实值序列中相邻元素之间的差异（今天的值 - 昨天的值）
    y_true_diff = np.diff(y_true)
    # 同样计算预测值序列的日度变化
    y_pred_diff = np.diff(y_pred)
    # 使用np.sign获取每个差异值的符号（+1代表上涨, -1代表下跌, 0代表不变）
    # 如果真实值和预测值的符号相同，则说明方向预测正确
    correct_direction = (np.sign(y_true_diff) == np.sign(y_pred_diff))
    # 计算方向正确的比例，并乘以100得到百分数
    return np.mean(correct_direction) * 100

# ==============================================================================
# 主流程开始 (Main Workflow)
# ==============================================================================

# --- 步骤 1: 数据加载与预处理 ---
print("--- 步骤 1: 正在加载和预处理数据... ---")
# 定义数据文件的路径
file_path = '50ETF_1min.csv'
# 使用pandas的read_csv函数读取数据到DataFrame中
df = pd.read_csv(file_path)
# 将名为't'的列（原始格式为字符串）转换为pandas的datetime对象，使其能够被识别为时间
df['t'] = pd.to_datetime(df['t'])
# 将't'列设置为DataFrame的索引，这样可以方便地按时间进行操作和重采样
df.set_index('t', inplace=True)
# 计算对数收益率：log(P_t) - log(P_{t-1})。使用.diff()可以方便地计算相邻元素的差值
df['log_return'] = np.log(df['close']).diff()

# --- 步骤 2: 计算已实现波动率 (RV) ---
print("--- 步骤 2: 正在计算日度已实现波动率... ---")
def calculate_realized_volatility(x):
    """一个辅助函数，用于计算单个交易日的已实现波动率"""
    # 在计算前，先移除可能存在的NaN值（比如一天的第一个分钟收益率）
    x = x.dropna()
    # 计算该日所有分钟对数收益率的平方和，然后开方
    return np.sqrt(np.sum(x**2))
# 使用.resample('D')将分钟级数据按天(Day)进行分组
# 然后对每天的数据应用我们定义的calculate_realized_volatility函数
daily_rv = df['log_return'].resample('D').apply(calculate_realized_volatility)
# 移除波动率极小（接近0）的行，这些通常是周末或节假日等非交易日，没有有效数据
daily_rv = daily_rv[daily_rv > 1e-6]
# 将计算出的Series转换为DataFrame，并为其命名
daily_rv_df = pd.DataFrame(daily_rv)
daily_rv_df.columns = ['realized_volatility']

# --- 步骤 3: 为HAR模型创建特征 ---
print("--- 步骤 3: 正在为HAR模型创建特征... ---")
# 复制一份日度RV数据，用于创建特征，避免修改原始数据
har_df = daily_rv_df.copy()
# 使用.shift(1)获取前一天(T-1)的RV，作为日度特征
har_df['rv_d'] = har_df['realized_volatility'].shift(1)
# 使用.rolling(window=5)创建一个5天的滑动窗口，计算窗口内RV的均值，再用.shift(1)获取T-1时刻的周度特征
har_df['rv_w'] = har_df['realized_volatility'].rolling(window=5).mean().shift(1)
# 同理，创建22天的滑动窗口来计算月度特征
har_df['rv_m'] = har_df['realized_volatility'].rolling(window=22).mean().shift(1)
# 由于shift和rolling操作会在数据开头产生空值(NaN)，这里将其全部移除
har_df = har_df.dropna()

# --- 步骤 4: 训练HAR-RV模型 ---
print("--- 步骤 4: 正在训练HAR-RV模型... ---")
# 定义模型的输入特征X（日、周、月三个维度的RV）
X_har = har_df[['rv_d', 'rv_w', 'rv_m']]
# 定义模型的预测目标y（当天的RV）
y_har = har_df['realized_volatility']
# 使用train_test_split划分训练集和测试集，test_size=0.2表示80%训练，20%测试
# shuffle=False至关重要，因为它能确保时间序列数据按时间顺序划分，不会打乱
X_train_har, X_test_har, y_train_har, y_test_har = train_test_split(
    X_har, y_har, test_size=0.2, shuffle=False
)
# 初始化一个线性回归模型对象
har_model = LinearRegression()
# 使用.fit()方法，用训练数据来训练模型
har_model.fit(X_train_har, y_train_har)
print("HAR-RV模型训练完成。")

# --- 步骤 5: 训练LSTM模型 ---
print("--- 步骤 5: 正在准备数据并训练LSTM模型... ---")
# 初始化一个MinMaxScaler，它会将数据缩放到0到1之间，这是神经网络训练的标准步骤
scaler = MinMaxScaler(feature_range=(0, 1))
# 使用.fit_transform()方法对日度RV数据进行归一化
rv_scaled = scaler.fit_transform(daily_rv_df[['realized_volatility']])
# 定义LSTM的输入时间窗口大小，即用过去10天的数据来预测未来
window_size = 10
# 初始化两个列表，用于存放处理后的输入样本X和输出标签y
X_lstm, y_lstm = [], []
# 遍历归一化后的数据，构建滑动窗口样本
for i in range(window_size, len(rv_scaled)):
    # 截取从i-window_size到i的序列作为输入特征
    X_lstm.append(rv_scaled[i-window_size:i, 0])
    # 将第i个值作为该序列对应的标签
    y_lstm.append(rv_scaled[i, 0])
# 将列表转换为numpy数组，便于后续处理
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
# 重塑X_lstm的形状为(样本数, 时间步长, 特征数)，这是LSTM层要求的3D输入格式
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
# 计算训练集和测试集的分割点
split_index = int(len(X_lstm) * 0.8)
# 按时间顺序分割数据
X_train_lstm, X_test_lstm = X_lstm[:split_index], X_lstm[split_index:]
y_train_lstm, y_test_lstm = y_lstm[:split_index], y_lstm[split_index:]

# 初始化一个序贯模型，这是搭建神经网络最常用的方式
lstm_model = Sequential()
# 添加第一个LSTM层，units=50表示有50个神经元。input_shape定义了输入数据的形状。
# return_sequences=True表示该层会输出完整的序列，以供下一个LSTM层使用
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
# 添加第二个LSTM层，它只输出最后一个时间步的结果
lstm_model.add(LSTM(units=50))
# 添加一个全连接层（Dense），units=1表示输出一个数值，即我们的预测值
lstm_model.add(Dense(units=1))
# 编译模型，指定优化器（adam常用且高效）和损失函数（均方误差）
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
# 训练模型，epochs=20表示对整个训练集进行20轮训练，batch_size=32表示每次更新权重用32个样本
# verbose=0表示训练过程中不打印详细日志
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
print("LSTM模型训练完成。")

# --- 步骤 6: 生成并对齐预测结果 ---
print("--- 步骤 6: 正在生成并对齐预测结果... ---")
# 使用训练好的HAR模型对测试集进行预测，并将结果存为带日期索引的Series
har_predictions = pd.Series(har_model.predict(X_test_har), index=y_test_har.index)
# 使用训练好的LSTM模型对测试集进行预测
predictions_lstm_scaled = lstm_model.predict(X_test_lstm)
# 使用scaler的.inverse_transform()方法将预测结果从0-1范围还原到原始的波动率尺度
predictions_lstm_unscaled = scaler.inverse_transform(predictions_lstm_scaled)
# 获取LSTM测试集对应的日期索引
lstm_test_dates = daily_rv_df.index[-len(y_test_lstm):]
# 将反归一化后的预测结果存为带日期索引的Series
lstm_predictions = pd.Series(predictions_lstm_unscaled.flatten(), index=lstm_test_dates)

# 创建一个新的DataFrame，用于存放真实值和两个模型的预测值
comparison_df = pd.DataFrame({
    'Actual': y_test_har,
    'HAR_Prediction': har_predictions
})
# 使用.join()方法，依据日期索引将LSTM的预测结果合并进来
# how='inner'确保只保留所有数据源（真实值、HAR、LSTM）都存在的日期，实现完美对齐
comparison_df = comparison_df.join(pd.DataFrame({'LSTM_Prediction': lstm_predictions}), how='inner')

# --- 步骤 7: 评估模型并打印报告 ---
print("\n--- 步骤 7: 正在评估模型性能... ---")
# 从对齐后的DataFrame中提取真实值和预测值序列
y_true = comparison_df['Actual']
har_pred = comparison_df['HAR_Prediction']
lstm_pred = comparison_df['LSTM_Prediction']

# 计算HAR模型的各项评估指标
mape_har = calculate_mape(y_true, har_pred)
r2_har = r2_score(y_true, har_pred)
da_har = calculate_directional_accuracy(y_true, har_pred)

# 计算LSTM模型的各项评估指标
mape_lstm = calculate_mape(y_true, lstm_pred)
r2_lstm = r2_score(y_true, lstm_pred)
da_lstm = calculate_directional_accuracy(y_true, lstm_pred)

# 使用格式化字符串打印出一个清晰的评估报告
print("="*50)
print("          模型性能评估报告")
print("="*50)
print(f"{'指标':<25} {'HAR 模型':<15} {'LSTM 模型':<15}")
print("-"*50)
print(f"{'MAE (Mean Absolute Error)':<25} {mean_absolute_error(y_true, har_pred):<15.6f} {mean_absolute_error(y_true, lstm_pred):<15.6f}")
print(f"{'MSE (Mean Squared Error)':<25} {mean_squared_error(y_true, har_pred):<15.6f} {mean_squared_error(y_true, lstm_pred):<15.6f}")
print(f"{'MAPE (Mean Abs. % Error)':<25} {mape_har:<14.2f}% {mape_lstm:<14.2f}%")
print(f"{'R-squared (R2 Score)':<25} {r2_har:<15.4f} {r2_lstm:<15.4f}")
print(f"{'DA (Directional Accuracy)':<25} {da_har:<14.2f}% {da_lstm:<14.2f}%")
print("-"*50)

# --- 步骤 8: 创建交互式图表 ---
print("\n--- 步骤 8: 正在创建交互式图表... ---")
# 使用make_subplots创建一个3行1列的子图画布，并共享X轴，设置子图标题
fig_subplots = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=('HAR Prediction', 'LSTM Prediction', 'Actual Volatility'))
# 计算所有数据的最大最小值，以确定一个统一的Y轴范围，方便公平比较
y_min = comparison_df.min().min() * 0.95
y_max = comparison_df.max().max() * 1.05
y_range = [y_min, y_max]
# 在第一个子图(row=1, col=1)中绘制HAR模型的预测曲线
fig_subplots.add_trace(go.Scatter(x=comparison_df.index, y=har_pred, mode='lines', name='HAR', line=dict(color='green')), row=1, col=1)
# 在第二个子图(row=2, col=1)中绘制LSTM模型的预测曲线
fig_subplots.add_trace(go.Scatter(x=comparison_df.index, y=lstm_pred, mode='lines', name='LSTM', line=dict(color='red')), row=2, col=1)
# 在第三个子图(row=3, col=1)中绘制真实的波动率曲线
fig_subplots.add_trace(go.Scatter(x=comparison_df.index, y=y_true, mode='lines', name='Actual', line=dict(color='blue')), row=3, col=1)
# 更新整个图表的布局，如标题、高度、图例等
fig_subplots.update_layout(title_text='Stacked Comparison of Volatility Predictions', height=800, showlegend=False, template='plotly_white')
# 将所有子图的Y轴范围设置为我们之前计算的统一范围
fig_subplots.update_yaxes(range=y_range, row=1, col=1)
fig_subplots.update_yaxes(range=y_range, row=2, col=1)
fig_subplots.update_yaxes(range=y_range, row=3, col=1)
# 将生成的交互式图表保存为一个HTML文件
fig_subplots.write_html("subplots_prediction_comparison.html")
# 调用.show()可以在浏览器中自动打开图表（在某些环境中可能需要手动打开HTML文件）
# fig_subplots.show()
print("\n堆叠式交互图表已保存为 'subplots_prediction_comparison.html'")
print("--- 所有步骤执行完毕 ---")
