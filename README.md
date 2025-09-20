LSTM_Predict_Vol
Replicate the study 'Can LSTM outperform volatility-econometric models?' with SSE 50 etf
English | 中文

<a name="english"></a>

Realized Volatility Prediction using HAR and LSTM Models
This project aims to reproduce the core ideas from the paper "Can LSTM outperform volatility-econometric models?" by applying them to the minute-level high-frequency data of the SSE 50 ETF. It provides a comparative analysis between a classic econometric model (HAR-RV) and a deep learning model (LSTM) for predicting daily realized volatility.

✨ Features
Data Processing: Implements a complete workflow for calculating daily Realized Volatility from high-frequency, minute-level data.

Model Implementation: Contains full implementations of both the classic HAR-RV model and an LSTM model based on TensorFlow/Keras.

Improved Visualization (Version 2.0): In addition to a combined interactive chart, the updated version generates a stacked subplot chart using Plotly, which offers a clearer, more direct comparison of the models' predictions against the actual values.

Modular Code: The code is structured cleanly, making it easy to understand and extend.

📂 File Structure
.
├── 50ETF_1min.csv          # Raw minute-level data
├── volatility_prediction.py # Main script
├── requirements.txt        # Project dependencies
├── interactive_prediction_comparison.html  # Generated interactive chart (combined)
├── version2.0/
│   ├── subplots_prediction_comparison.html     # Generated interactive chart (subplots)
│   ├── ... # Other files for version 2.0
├── prediction_comparison.png               # Generated static chart
└── README.md               # This file
🚀 How to Run
Clone the Repository

Bash
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_PROJECT_FOLDER]
Create and Activate a Virtual Environment

Bash
# Create the environment
python3 -m venv venv
# Activate on macOS/Linux
source venv/bin/activate
# Activate on Windows
.\venv\Scripts\activate
Install Dependencies

Bash
pip install -r requirements.txt
Run the Script

Bash
python volatility_prediction.py
After execution, the script will automatically generate *.png and *.html chart files in the repository. The new stacked subplots will be placed in the version2.0 folder.

📊 Experimental Results
This project was backtested on the SSE 50 ETF data from 2005 to 2022. The performance metrics of the two models on the test set are as follows:

HAR-RV Model: MSE = 0.000020, MAE = 0.002487

LSTM Model: MSE = 0.000021, MAE = 0.002869

The results indicate that the classic HAR-RV model slightly outperformed the baseline LSTM model in this experiment. This aligns with the original paper's conclusion that deep learning models require careful fine-tuning to surpass strong benchmarks.

Prediction Results Comparison Chart:

📚 References
Rodikov, G., & Antulov-Fantulin, N. (2022). Can LSTM outperform volatility-econometric models?. arXiv preprint arXiv:2202.11581.

Bucci, A. (2020). Realized Volatility Forecasting with Neural Networks. Journal of Financial Econometrics, 18(3), 502-531.

<a name="中文"></a>

基于HAR与LSTM模型的已实现波动率预测
本项目旨在复现论文《Can LSTM outperform volatility-econometric models?》中的核心思想，通过使用上证50ETF的分钟级高频数据，对比分析了经典的计量经济学模型（HAR-RV）与深度学习模型（LSTM）在预测日度已实现波动率方面的表现。

✨ 项目特点
数据处理：实现了从分钟级高频数据计算日度已实现波动率（Realized Volatility）的完整流程。

模型实现：包含了经典的HAR-RV模型和基于TensorFlow/Keras的LSTM模型的完整实现。

改进的可视化（版本2.0）：除了合并的交互式图表外，更新版本使用 Plotly 生成了堆叠式子图表，提供了更清晰、更直观的模型预测结果与实际值对比。

模块化代码：代码结构清晰，易于理解和扩展。

📂 文件结构
.
├── 50ETF_1min.csv          # 原始分钟级数据
├── volatility_prediction.py # 主代码文件
├── requirements.txt        # 项目依赖库
├── interactive_prediction_comparison.html  # 生成的合并交互图
├── version2.0/
│   ├── subplots_prediction_comparison.html     # 生成的堆叠交互图
│   ├── ... # Version 2.0 的其他文件
├── prediction_comparison.png               # 生成的静态结果图
└── README.md               # 本说明文件
🚀 如何运行
克隆代码库

Bash
git clone [您的代码库URL]
cd [您的项目文件夹]
创建并激活虚拟环境

Bash
# 创建
python3 -m venv venv
# 激活 (macOS/Linux)
source venv/bin/activate
# 激活 (Windows)
.\venv\Scripts\activate
安装依赖

Bash
pip install -r requirements.txt
运行脚本

Bash
python volatility_prediction.py
脚本运行后，会自动在仓库中生成 *.png 和 *.html 的结果图表文件。新的堆叠子图表将保存在 version2.0 文件夹内。

📊 实验结果
本项目在2005年至2022年的上证50ETF数据上进行了回测。在测试集上，两个模型的性能指标如下：

HAR-RV 模型: MSE = 0.000020, MAE = 0.002487

LSTM 模型: MSE = 0.000021, MAE = 0.002869

预测结果对比图:

从结果来看，经典的HAR-RV模型在本次实验中的表现略优于基准的LSTM模型，这与原论文中“深度学习模型需要精细调参才能超越强基准”的结论相符。

📚 参考文献
Rodikov, G., & Antulov-Fantulin, N. (2022). Can LSTM outperform volatility-econometric models?. arXiv preprint arXiv:2202.11581.

Bucci, A. (2020). Realized Volatility Forecasting with Neural Networks. Journal of Financial Econometrics, 18(3), 502-531.
