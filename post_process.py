import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体（如果需要）
try:
    # 使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception as e:
    print(f"Warning: 无法设置中文字体: {e}")

# 读取预测结果
def load_prediction_results(file_path='hybrid_predictions.csv'):
    """读取预测结果文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return None

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 重命名列以匹配原代码中的列名
        column_mapping = {
            'YTest': 'YTest',
            'V_SEM': 'A_V_SEM',
            'V_DDM': 'A_V_DDM',
            'V_Hybrid': 'A_V_Hybrid',
            'Error_SEM': 'Error_SEM',
            'Error_DDM': 'Error_DDM',
            'Error_Hybrid': 'Error_Hybrid'
        }

        # 只重命名存在的列
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

        return df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# 计算评估指标
def calculate_metrics(observed, predicted):
    """计算评估指标"""
    # 计算R方值
    mean_observed = np.mean(observed)
    ss_tot = np.sum((observed - mean_observed) ** 2)
    ss_res = np.sum((observed - predicted) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 计算均方根误差
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    
    # 计算平均绝对误差
    mae = np.mean(np.abs(predicted - observed))
    
    # 计算平均相对误差
    re = 100 * np.abs((predicted - observed) / observed)
    mean_re = np.mean(re)
    max_re = np.max(re)
    min_re = np.min(re)
    
    return {
        'R方值': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        '平均相对误差(%)': mean_re,
        '最大相对误差(%)': max_re,
        '最小相对误差(%)': min_re
    }

# 绘制预测结果图表
def plot_results(df):
    """绘制预测结果图表"""
    # 创建一个2x1的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 设置时间点
    time_points = np.arange(len(df))
    
    # 第一个子图：观测值和预测值对比
    ax1.plot(time_points, df['YTest'], 'b-', label='观测值')
    ax1.plot(time_points, df['A_V_Hybrid'], 'r.-', label='混合模型预测值')
    ax1.set_ylabel('电压 (V)')
    ax1.set_title('观测值与预测值对比')
    ax1.legend()
    ax1.grid(True)
    
    # 第二个子图：预测误差
    error = df['A_V_Hybrid'] - df['YTest']
    ax2.stem(time_points, error, basefmt='b-')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('误差 (V)')
    ax2.set_title('混合模型预测误差')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("预测对比图已保存为 'prediction_comparison.png'")
    
    # 绘制各模型预测对比图
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, df['YTest'], 'b-', label='观测值')
    plt.plot(time_points, df['A_V_SEM'], 'g--', label='SEDM模型')
    plt.plot(time_points, df['A_V_DDM'], 'm--', label='DDM模型')
    plt.plot(time_points, df['A_V_Hybrid'], 'r.-', label='混合模型')
    plt.xlabel('时间步')
    plt.ylabel('电压 (V)')
    plt.title('各模型预测结果对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
    print("模型对比图已保存为 'models_comparison.png'")
    
    # 绘制相对误差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(100 * np.abs((df['A_V_Hybrid'] - df['YTest']) / df['YTest']), bins=50, alpha=0.7, color='blue')
    plt.xlabel('相对误差 (%)')
    plt.ylabel('频数')
    plt.title('混合模型相对误差分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("误差分布图已保存为 'error_distribution.png'")
    
    # 显示所有图表
    plt.show()

# 生成详细的评估报告
def generate_report(df):
    """生成详细的评估报告"""
    print("=" * 60)
    print("模型预测评估报告")
    print("=" * 60)
    
    # 计算各模型的评估指标
    metrics_sem = calculate_metrics(df['YTest'], df['A_V_SEM'])
    metrics_ddm = calculate_metrics(df['YTest'], df['A_V_DDM'])
    metrics_hybrid = calculate_metrics(df['YTest'], df['A_V_Hybrid'])
    
    # 打印SEDM模型指标
    print("\n1. SEDM模型评估指标:")
    for key, value in metrics_sem.items():
        print(f"   {key}: {value:.6f}")
    
    # 打印DDM模型指标
    print("\n2. DDM模型评估指标:")
    for key, value in metrics_ddm.items():
        print(f"   {key}: {value:.6f}")
    
    # 打印混合模型指标
    print("\n3. 混合模型评估指标:")
    for key, value in metrics_hybrid.items():
        print(f"   {key}: {value:.6f}")
    
    # 计算改进百分比
    sem_to_hybrid_improvement = 100 * (metrics_sem['平均相对误差(%)'] - metrics_hybrid['平均相对误差(%)']) / metrics_sem['平均相对误差(%)']
    ddm_to_hybrid_improvement = 100 * (metrics_ddm['平均相对误差(%)'] - metrics_hybrid['平均相对误差(%)']) / metrics_ddm['平均相对误差(%)']
    
    print("\n4. 混合模型改进情况:")
    print(f"   相比SEDM模型的相对误差减少: {sem_to_hybrid_improvement:.2f}%")
    print(f"   相比DDM模型的相对误差减少: {ddm_to_hybrid_improvement:.2f}%")
    
    # 保存评估报告到文件
    with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型预测评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. SEDM模型评估指标:\n")
        for key, value in metrics_sem.items():
            f.write(f"   {key}: {value:.6f}\n")
        
        f.write("\n2. DDM模型评估指标:\n")
        for key, value in metrics_ddm.items():
            f.write(f"   {key}: {value:.6f}\n")
        
        f.write("\n3. 混合模型评估指标:\n")
        for key, value in metrics_hybrid.items():
            f.write(f"   {key}: {value:.6f}\n")
        
        f.write("\n4. 混合模型改进情况:\n")
        f.write(f"   相比SEDM模型的相对误差减少: {sem_to_hybrid_improvement:.2f}%\n")
        f.write(f"   相比DDM模型的相对误差减少: {ddm_to_hybrid_improvement:.2f}%\n")
    
    print("\n评估报告已保存为 'evaluation_report.txt'")

# 生成额外的分析
# 生成残差分析
def analyze_residuals(df):
    """分析预测残差"""
    # 计算残差
    residuals = df['A_V_Hybrid'] - df['YTest']
    observed = df['YTest']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(observed, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('观测值 (V)')
    plt.ylabel('残差 (V)')
    plt.title('残差与观测值关系图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
    print("残差分析图已保存为 'residual_analysis.png'")

# 主函数
def main():
    print("开始后处理分析...")
    
    # 加载预测结果
    df = load_prediction_results()
    if df is None:
        print("无法继续分析，退出程序。")
        return
    
    print(f"成功加载预测结果，共 {len(df)} 个数据点。")
    
    # 生成评估报告
    generate_report(df)
    
    # 绘制结果图表
    plot_results(df)
    
    # 残差分析
    analyze_residuals(df)
    
    print("\n后处理分析完成！")

if __name__ == "__main__":
    main()