import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('ggplot')  # 使用ggplot风格美化图表

# 自定义色盘
CUSTOM_COLORS = ["#2878B5", "#9AC9DB", "#C82423", "#F8AC8C", "#1E9F75", "#95C66F", "#B365D3", "#F28E2B"]

# 数据加载函数
def load_data(enhanced_file='enhanced_wind_turbine_data.csv', original_file='wind_turbine_data.csv'):
    """
    加载原始数据集和增强数据集
    
    参数:
        enhanced_file: 增强数据集文件路径
        original_file: 原始数据集文件路径
        
    返回:
        df_enhanced: 增强数据集DataFrame
        df_original: 原始数据集DataFrame (如果存在)
    """
    # 加载增强数据集
    df_enhanced = pd.read_csv(enhanced_file)
    if '时间' in df_enhanced.columns:
        df_enhanced['时间'] = pd.to_datetime(df_enhanced['时间'])
    
    # 尝试加载原始数据集(用于对比)
    try:
        df_original = pd.read_csv(original_file)
        if '时间' in df_original.columns:
            df_original['时间'] = pd.to_datetime(df_original['时间'])
        print(f"成功加载原始数据集和增强数据集")
        return df_enhanced, df_original
    except:
        print(f"未找到原始数据集，仅加载增强数据集")
        return df_enhanced, None

# 数据集基本统计信息
def dataset_stats(df_enhanced, df_original=None):
    """显示数据集的基本统计信息"""
    
    enhanced_stats = {
        "样本数量": len(df_enhanced),
        "特征数量": len(df_enhanced.columns) - 1,  # 排除故障标签
        "故障样本数": df_enhanced['故障标签'].sum(),
        "故障样本比例": f"{df_enhanced['故障标签'].mean()*100:.2f}%",
    }
    
    print("\n=== 增强数据集统计信息 ===")
    for key, value in enhanced_stats.items():
        print(f"{key}: {value}")
    
    if df_original is not None:
        original_stats = {
            "样本数量": len(df_original),
            "特征数量": len(df_original.columns) - 1,  # 排除故障标签
            "故障样本数": df_original['故障标签'].sum(),
            "故障样本比例": f"{df_original['故障标签'].mean()*100:.2f}%"
        }
        
        print("\n=== 原始数据集统计信息 ===")
        for key, value in original_stats.items():
            print(f"{key}: {value}")
        
        # 计算增强比例
        enhancement_ratio = {
            "样本增强比例": f"{(len(df_enhanced) / len(df_original) - 1) * 100:.2f}%",
            "故障样本增强比例": f"{(df_enhanced['故障标签'].sum() / df_original['故障标签'].sum() - 1) * 100:.2f}%"
        }
        
        print("\n=== 数据增强比例 ===")
        for key, value in enhancement_ratio.items():
            print(f"{key}: {value}")

# 1. 数据集对比可视化
def plot_dataset_comparison(df_enhanced, df_original=None, features=None):
    """
    对比增强数据集和原始数据集的分布情况
    
    参数:
        df_enhanced: 增强数据集
        df_original: 原始数据集
        features: 要对比的特征列表，如果为None则使用默认特征
    """
    if features is None:
        features = ['风速(m/s)', '转速(rpm)', '功率输出(kW)', '温度(℃)', '振动(mm/s)']
    
    if df_original is None:
        print("未提供原始数据集，无法进行对比可视化")
        return
    
    # 1.1 特征分布对比
    fig, axes = plt.subplots(len(features), 2, figsize=(18, 4*len(features)))
    fig.suptitle('增强数据集与原始数据集特征分布对比', fontsize=16, y=0.98)
    
    for i, feature in enumerate(features):
        if feature in df_enhanced.columns and feature in df_original.columns:
            # 原始数据集分布
            sns.histplot(df_original[feature], kde=True, ax=axes[i, 0], 
                        color=CUSTOM_COLORS[0], alpha=0.7, label='原始数据')
            axes[i, 0].set_title(f'原始数据集 - {feature}分布')
            axes[i, 0].set_xlabel(feature)
            axes[i, 0].set_ylabel('频数')
            
            # 增强数据集分布
            sns.histplot(df_enhanced[feature], kde=True, ax=axes[i, 1], 
                        color=CUSTOM_COLORS[2], alpha=0.7, label='增强数据')
            axes[i, 1].set_title(f'增强数据集 - {feature}分布')
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('频数')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('增强_原始数据分布对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.2 故障样本分布对比
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('故障样本比例对比', fontsize=16)
    
    # 原始数据集故障分布
    original_counts = df_original['故障标签'].value_counts()
    axes[0].pie(original_counts, labels=['正常', '故障'], autopct='%1.1f%%', 
               startangle=90, colors=[CUSTOM_COLORS[0], CUSTOM_COLORS[2]])
    axes[0].set_title('原始数据集故障样本比例')
    
    # 增强数据集故障分布
    enhanced_counts = df_enhanced['故障标签'].value_counts()
    axes[1].pie(enhanced_counts, labels=['正常', '故障'], autopct='%1.1f%%', 
               startangle=90, colors=[CUSTOM_COLORS[0], CUSTOM_COLORS[2]])
    axes[1].set_title('增强数据集故障样本比例')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('故障样本比例对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.3 关键物理关系对比散点图
    key_relationships = [
        ('风速(m/s)', '功率输出(kW)', '风速-功率关系'),
        ('转速(rpm)', '振动(mm/s)', '转速-振动关系'),
        ('功率输出(kW)', '温度(℃)', '功率-温度关系'),
    ]
    
    fig, axes = plt.subplots(len(key_relationships), 2, figsize=(16, 5*len(key_relationships)))
    fig.suptitle('关键物理关系对比', fontsize=16, y=0.98)
    
    for i, (x_feature, y_feature, title) in enumerate(key_relationships):
        if x_feature in df_original.columns and y_feature in df_original.columns:
            # 原始数据集散点图
            axes[i, 0].scatter(df_original[x_feature], df_original[y_feature], 
                              c=df_original['故障标签'], cmap='coolwarm', 
                              alpha=0.6, s=30, edgecolor='k')
            axes[i, 0].set_title(f'原始数据集 - {title}')
            axes[i, 0].set_xlabel(x_feature)
            axes[i, 0].set_ylabel(y_feature)
            axes[i, 0].grid(True, alpha=0.3)
            
            # 增强数据集散点图
            axes[i, 1].scatter(df_enhanced[x_feature], df_enhanced[y_feature], 
                              c=df_enhanced['故障标签'], cmap='coolwarm', 
                              alpha=0.6, s=30, edgecolor='k')
            axes[i, 1].set_title(f'增强数据集 - {title}')
            axes[i, 1].set_xlabel(x_feature)
            axes[i, 1].set_ylabel(y_feature)
            axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('关键物理关系对比.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. 物理关系可视化
def plot_physics_relationships(df_enhanced):
    """
    可视化增强数据集中的物理约束关系
    
    参数:
        df_enhanced: 增强数据集
    """
    # 2.1 风速与功率输出的三次方关系曲线图
    plt.figure(figsize=(12, 8))
    plt.scatter(df_enhanced['风速(m/s)'], df_enhanced['功率输出(kW)'], 
               c=df_enhanced['故障标签'], cmap='coolwarm', 
               alpha=0.6, s=40, edgecolor='k')
    
    # 拟合三次函数曲线 (贝兹公式: P ~ v^3)
    wind_x = np.linspace(df_enhanced['风速(m/s)'].min(), df_enhanced['风速(m/s)'].max(), 100)
    plt.title('风速与功率关系（验证贝兹公式）', fontsize=16)
    plt.xlabel('风速(m/s)')
    plt.ylabel('功率输出(kW)')
    
    # 只选择正常工作点来拟合曲线
    normal_samples = df_enhanced[df_enhanced['故障标签'] == 0]
    valid_samples = normal_samples[(normal_samples['风速(m/s)'] > 3) & 
                                  (normal_samples['风速(m/s)'] < 25) &
                                  (normal_samples['功率输出(kW)'] > 0)]
    
    # 拟合贝兹公式 P = k * v^3
    x = valid_samples['风速(m/s)']
    y = valid_samples['功率输出(kW)']
    
    # 使用非线性最小二乘拟合
    from scipy.optimize import curve_fit
    
    def betz_formula(v, k):
        return k * v**3
    
    try:
        params, _ = curve_fit(betz_formula, x, y)
        k = params[0]
        
        # 绘制拟合曲线
        plt.plot(wind_x, betz_formula(wind_x, k), 'r--', linewidth=2, 
                label=f'贝兹公式: P = {k:.4f} * $v³$')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='故障标签')
        plt.tight_layout()
        plt.savefig('风速功率贝兹关系.png', dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("贝兹公式拟合失败，可能数据不足或不符合理论模型")
    
    # 2.2 转速与振动的二次关系散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(df_enhanced['转速(rpm)'], df_enhanced['振动(mm/s)'], 
               c=df_enhanced['故障标签'], cmap='coolwarm', 
               alpha=0.6, s=40, edgecolor='k')
    
    # 拟合二次函数曲线
    rpm_x = np.linspace(df_enhanced['转速(rpm)'].min(), df_enhanced['转速(rpm)'].max(), 100)
    
    # 只选择正常工作点来拟合曲线
    normal_samples = df_enhanced[df_enhanced['故障标签'] == 0]
    
    # 拟合二次方程 v = a*rpm^2 + b*rpm + c
    x = normal_samples['转速(rpm)']
    y = normal_samples['振动(mm/s)']
    
    try:
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        
        a, b, c = z
        plt.plot(rpm_x, p(rpm_x), 'r--', linewidth=2, 
                label=f'振动 = {a:.4f}*转速$^2$ + {b:.4f}*转速 + {c:.4f}')
                        

        plt.title('转速与振动的二次关系', fontsize=16)
        plt.xlabel('转速(rpm)')
        plt.ylabel('振动(mm/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='故障标签')
        plt.tight_layout()
        plt.savefig('转速振动二次关系.png', dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("转速-振动二次关系拟合失败")
    
    # 2.3 功率与温度的线性关系图
    plt.figure(figsize=(12, 8))
    
    if '发电机温度(℃)' in df_enhanced.columns:
        temp_feature = '发电机温度(℃)'
    else:
        temp_feature = '温度(℃)'
    
    plt.scatter(df_enhanced['功率输出(kW)'], df_enhanced[temp_feature], 
               c=df_enhanced['故障标签'], cmap='coolwarm', 
               alpha=0.6, s=40, edgecolor='k')
    
    # 拟合线性函数
    power_x = np.linspace(df_enhanced['功率输出(kW)'].min(), df_enhanced['功率输出(kW)'].max(), 100)
    
    # 只选择正常工作点来拟合曲线
    normal_samples = df_enhanced[df_enhanced['故障标签'] == 0]
    
    # 拟合线性方程 T = a*P + b
    x = normal_samples['功率输出(kW)']
    y = normal_samples[temp_feature]
    
    try:
        a, b = np.polyfit(x, y, 1)
        plt.plot(power_x, a*power_x + b, 'r--', linewidth=2, 
                label=f'温度 = {a:.4f}*功率 + {b:.4f}')
        
        plt.title('功率与温度的线性关系', fontsize=16)
        plt.xlabel('功率输出(kW)')
        plt.ylabel(temp_feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='故障标签')
        plt.tight_layout()
        plt.savefig('功率温度线性关系.png', dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("功率-温度线性关系拟合失败")
    
    # 2.4 风速-转速-功率三者关系热力图
    fig = plt.figure(figsize=(16, 12))
    
    # 创建3D散点图
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_enhanced['风速(m/s)'], 
                       df_enhanced['转速(rpm)'], 
                       df_enhanced['功率输出(kW)'],
                       c=df_enhanced['故障标签'],
                       cmap='coolwarm',
                       s=30,
                       alpha=0.6)
    
    ax.set_xlabel('风速(m/s)')
    ax.set_ylabel('转速(rpm)')
    ax.set_zlabel('功率输出(kW)')
    ax.set_title('风速-转速-功率三维关系图', fontsize=16)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('故障标签')
    
    plt.tight_layout()
    plt.savefig('风速转速功率三维关系.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. 故障模式分析
def plot_fault_analysis(df_enhanced):
    """
    分析增强数据集中的故障模式
    
    参数:
        df_enhanced: 增强数据集
    """
    # 3.1 故障样本在各物理量下的分布热图
    key_features = ['风速(m/s)', '转速(rpm)', '功率输出(kW)', '温度(℃)', 
                   '振动(mm/s)', '油温(℃)', '油位(%)']
    
    # 确保所有特征都在数据集中
    key_features = [f for f in key_features if f in df_enhanced.columns]
    
    # 对于每个特征，创建故障和正常样本的分布对比
    fig, axes = plt.subplots(len(key_features), 1, figsize=(14, 4*len(key_features)))
    fig.suptitle('故障状态下各参数分布对比', fontsize=16)
    
    for i, feature in enumerate(key_features):
        # 分别获取正常和故障样本数据
        normal_data = df_enhanced[df_enhanced['故障标签'] == 0][feature]
        fault_data = df_enhanced[df_enhanced['故障标签'] == 1][feature]
        
        # 在同一子图中绘制两组数据的KDE曲线
        sns.kdeplot(normal_data, ax=axes[i], label='正常', color=CUSTOM_COLORS[0], fill=True, alpha=0.3)
        sns.kdeplot(fault_data, ax=axes[i], label='故障', color=CUSTOM_COLORS[2], fill=True, alpha=0.3)
        
        axes[i].set_title(f'{feature}在不同状态下的分布')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('密度')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('故障参数分布对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 故障样本在多维特征空间的降维可视化 (t-SNE)
    # 提取特征和标签
    X = df_enhanced[key_features].values
    y = df_enhanced['故障标签'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 创建降维可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=y, cmap='coolwarm', 
                         alpha=0.7, s=50, edgecolor='k')
    
    plt.title('增强数据集故障模式的t-SNE降维可视化', fontsize=16)
    plt.xlabel('t-SNE特征1')
    plt.ylabel('t-SNE特征2')
    plt.colorbar(label='故障标签')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('故障模式tsne可视化.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.3 故障样本特征雷达图
    # 计算正常和故障样本在各特征上的平均值
    normal_means = df_enhanced[df_enhanced['故障标签'] == 0][key_features].mean()
    fault_means = df_enhanced[df_enhanced['故障标签'] == 1][key_features].mean()
    
    # 将数据标准化，使各特征处于相同尺度
    # 首先确定每个特征的最大值和最小值
    max_values = df_enhanced[key_features].max()
    min_values = df_enhanced[key_features].min()
    
    # 标准化平均值
    normal_means_norm = (normal_means - min_values) / (max_values - min_values)
    fault_means_norm = (fault_means - min_values) / (max_values - min_values)
    
    # 创建雷达图
    angles = np.linspace(0, 2*np.pi, len(key_features), endpoint=False).tolist()
    
    # 闭合雷达图
    normal_means_norm = pd.concat([normal_means_norm, pd.Series([normal_means_norm.iloc[0]], index=[normal_means_norm.index[0]])])
    fault_means_norm = pd.concat([fault_means_norm, pd.Series([fault_means_norm.iloc[0]], index=[fault_means_norm.index[0]])])
    angles = angles + [angles[0]]
    
    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 正常样本雷达图
    ax.plot(angles, normal_means_norm, 'o-', linewidth=2, label='正常样本', color=CUSTOM_COLORS[0])
    ax.fill(angles, normal_means_norm, alpha=0.25, color=CUSTOM_COLORS[0])
    
    # 故障样本雷达图
    ax.plot(angles, fault_means_norm, 'o-', linewidth=2, label='故障样本', color=CUSTOM_COLORS[2])
    ax.fill(angles, fault_means_norm, alpha=0.25, color=CUSTOM_COLORS[2])
    
    # 设置雷达图参数
    ax.set_thetagrids(np.degrees(angles[:-1]), key_features)
    ax.set_title('正常与故障样本特征雷达图', fontsize=16, y=1.08)
    ax.grid(True)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('故障特征雷达图.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. 特征重要性可视化
def plot_feature_importance(df_enhanced):
    """
    分析特征重要性并可视化
    
    参数:
        df_enhanced: 增强数据集
    """
    # 排除非数值列和目标列
    feature_columns = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    if '故障标签' in feature_columns:
        feature_columns.remove('故障标签')
    
    # 使用随机森林分类器来计算特征重要性
    X = df_enhanced[feature_columns].values
    y = df_enhanced['故障标签'].values
    
    # 训练随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 4.1 特征重要性条形图
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(importances)), importances[indices], color=CUSTOM_COLORS, align='center')
    plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.title('特征重要性排序', fontsize=16)
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.tight_layout()
    plt.savefig('特征重要性排序.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.2 相关性热图
    plt.figure(figsize=(16, 14))
    corr = df_enhanced[feature_columns + ['故障标签']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
               linewidths=0.5, vmin=-1, vmax=1, center=0)
    plt.title('特征相关性热图', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('特征相关性热图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.3 特征对故障影响散点图矩阵
    # 选取前6个最重要的特征
    top_features = [feature_columns[i] for i in indices[:6]]
    
    g = sns.pairplot(df_enhanced[top_features + ['故障标签']], 
                    hue='故障标签', palette=['blue', 'red'],
                    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'w'},
                    diag_kind='kde')
    
    g.fig.suptitle('重要特征对故障影响的散点图矩阵', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('重要特征散点图矩阵.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. 新增：特征区分能力对比可视化
def plot_feature_discrimination_comparison(df_enhanced):
    """
    通过对比箱线图和特定二维散点图，可视化不同特征对故障的区分能力。
    
    参数:
        df_enhanced: 增强数据集
    """
    print("正在生成特征区分能力对比可视化...")

    # 6.1 对比箱线图
    # 选择要对比的关键特征
    features_to_compare = [
        '振动(mm/s)', 
        '齿轮箱油位(%)', 
        '油位(%)', 
        '发电机温度(℃)', 
        '齿轮箱油温(℃)', 
        '油温(℃)', 
        '功率输出(kW)'
    ]
    # 确保特征存在
    features_to_compare = [f for f in features_to_compare if f in df_enhanced.columns]
    
    n_features = len(features_to_compare)
    plt.figure(figsize=(18, 5 * n_features // 2 if n_features % 2 == 0 else 5 * (n_features // 2 + 1))) # 调整画布大小
    plt.suptitle('不同特征对正常/故障状态的区分能力 (箱线图对比)', fontsize=18, y=1.02)

    for i, feature in enumerate(features_to_compare):
        ax = plt.subplot( (n_features + 1) // 2, 2, i + 1) # 两列布局
        
        # 使用seaborn绘制箱线图，按故障标签分组
        sns.boxplot(
            x='故障标签', 
            y=feature, 
            data=df_enhanced, 
            palette=[CUSTOM_COLORS[0], CUSTOM_COLORS[2]], # 正常用蓝色，故障用红色
            ax=ax,
            showfliers=False # 不显示异常值点，使图像更清晰
        )
        
        ax.set_title(f'{feature} 分布对比', fontsize=14)
        ax.set_xlabel('状态 (0: 正常, 1: 故障)', fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('特征区分能力_箱线图对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6.2 特定二维特征空间散点图对比
    plt.figure(figsize=(18, 8))
    plt.suptitle('特征空间散点图对比 (区分能力)', fontsize=16, y=1.0)

    # 图一：高区分度特征对 (振动 vs 齿轮箱油位)
    ax1 = plt.subplot(1, 2, 1)
    if '振动(mm/s)' in df_enhanced.columns and '齿轮箱油位(%)' in df_enhanced.columns:
        scatter1 = ax1.scatter(
            df_enhanced['振动(mm/s)'], 
            df_enhanced['齿轮箱油位(%)'], 
            c=df_enhanced['故障标签'], 
            cmap='coolwarm', 
            alpha=0.6, s=30, edgecolor='k'
        )
        ax1.set_title('高区分度特征空间 (振动 vs 齿轮箱油位)', fontsize=14)
        ax1.set_xlabel('振动(mm/s)', fontsize=12)
        ax1.set_ylabel('齿轮箱油位(%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        # 添加颜色条
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('故障标签')
    else:
        ax1.text(0.5, 0.5, '缺少所需特征', horizontalalignment='center', verticalalignment='center')
        ax1.set_title('高区分度特征空间 (特征缺失)', fontsize=14)


    # 图二：相对低区分度特征对 (发电机温度 vs 齿轮箱油温)
    ax2 = plt.subplot(1, 2, 2)
    temp_feature = '发电机温度(℃)' if '发电机温度(℃)' in df_enhanced.columns else '温度(℃)'
    if temp_feature in df_enhanced.columns and '齿轮箱油温(℃)' in df_enhanced.columns:
        scatter2 = ax2.scatter(
            df_enhanced[temp_feature], 
            df_enhanced['齿轮箱油温(℃)'], 
            c=df_enhanced['故障标签'], 
            cmap='coolwarm', 
            alpha=0.6, s=30, edgecolor='k'
        )
        ax2.set_title('相对低区分度特征空间 (温度 vs 油温)', fontsize=14)
        ax2.set_xlabel(temp_feature, fontsize=12)
        ax2.set_ylabel('齿轮箱油温(℃)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
         # 添加颜色条
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('故障标签')
    else:
        ax2.text(0.5, 0.5, '缺少所需特征', horizontalalignment='center', verticalalignment='center')
        ax2.set_title('相对低区分度特征空间 (特征缺失)', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('特征区分能力_散点图对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("特征区分能力对比可视化完成!")

# 5. 交互式可视化
def create_interactive_visualizations(df_enhanced):
    """
    创建交互式可视化图表
    
    参数:
        df_enhanced: 增强数据集
    """
    # 5.1 创建3D交互式散点图
    fig_3d = px.scatter_3d(
        df_enhanced, 
        x='风速(m/s)', 
        y='转速(rpm)', 
        z='功率输出(kW)',
        color='故障标签',
        size='振动(mm/s)',
        opacity=0.7,
        color_discrete_sequence=['blue', 'red'],
        title='风速-转速-功率三维关系交互式可视化'
    )
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='风速(m/s)',
            yaxis_title='转速(rpm)',
            zaxis_title='功率输出(kW)'
        ),
        width=1000,
        height=800
    )
    
    fig_3d.write_html('风速转速功率_3D交互图.html')
    
    # 5.2 创建时间序列交互式图表(如果有时间列)
    if '时间' in df_enhanced.columns:
        # 按日期聚合，计算每日故障样本数量
        df_enhanced['日期'] = df_enhanced['时间'].dt.date
        daily_faults = df_enhanced.groupby('日期')['故障标签'].sum().reset_index()
        daily_faults['日期'] = pd.to_datetime(daily_faults['日期'])
        
        fig_time = px.line(
            daily_faults,
            x='日期',
            y='故障标签',
            title='每日故障数量变化趋势'
        )
        
        fig_time.update_layout(
            xaxis_title='日期',
            yaxis_title='故障样本数量',
            width=1000,
            height=600
        )
        
        fig_time.write_html('每日故障趋势_交互图.html')
    
    # 5.3 创建多参数交互式仪表盘
    key_features = ['风速(m/s)', '转速(rpm)', '功率输出(kW)', '温度(℃)', '振动(mm/s)']
    key_features = [f for f in key_features if f in df_enhanced.columns]
    
    # 创建多页面仪表盘
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '风速与功率关系', '转速与振动关系',
            '功率与温度关系', '故障样本分布',
            '特征重要性', '参数状态分布'
        )
    )
    
    # 1. 风速与功率关系
    fig.add_trace(
        go.Scatter(
            x=df_enhanced['风速(m/s)'],
            y=df_enhanced['功率输出(kW)'],
            mode='markers',
            marker=dict(
                size=8,
                color=df_enhanced['故障标签'],
                colorscale='Viridis',
                showscale=True
            ),
            name='风速vs功率'
        ),
        row=1, col=1
    )
    
    # 2. 转速与振动关系
    fig.add_trace(
        go.Scatter(
            x=df_enhanced['转速(rpm)'],
            y=df_enhanced['振动(mm/s)'],
            mode='markers',
            marker=dict(
                size=8,
                color=df_enhanced['故障标签'],
                colorscale='Viridis',
                showscale=False
            ),
            name='转速vs振动'
        ),
        row=1, col=2
    )
    
    # 3. 功率与温度关系
    temp_feature = '发电机温度(℃)' if '发电机温度(℃)' in df_enhanced.columns else '温度(℃)'
    fig.add_trace(
        go.Scatter(
            x=df_enhanced['功率输出(kW)'],
            y=df_enhanced[temp_feature],
            mode='markers',
            marker=dict(
                size=8,
                color=df_enhanced['故障标签'],
                colorscale='Viridis',
                showscale=False
            ),
            name='功率vs温度'
        ),
        row=2, col=1
    )
    
    # 4. 故障样本分布 - 改为柱状图
    fault_counts = df_enhanced['故障标签'].value_counts()
    fig.add_trace(
        go.Bar(
            x=['正常', '故障'],
            y=[fault_counts[0], fault_counts[1]],
            marker_color=[CUSTOM_COLORS[0], CUSTOM_COLORS[2]]
        ),
        row=2, col=2
    )
    
    # 5. 特征重要性 - 训练随机森林获取特征重要性
    feature_columns = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    if '故障标签' in feature_columns:
        feature_columns.remove('故障标签')
    
    X = df_enhanced[feature_columns].values
    y = df_enhanced['故障标签'].values
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-5:]  # 只取前5个重要特征
    
    fig.add_trace(
        go.Bar(
            x=[feature_columns[i] for i in indices],
            y=importances[indices],
            marker_color=CUSTOM_COLORS[:5]
        ),
        row=3, col=1
    )
    
    # 6. 参数状态分布 - 小提琴图 (转为箱线图，因为plotly没有直接的小提琴图)
    # 为了简化，这里只展示振动参数
    fig.add_trace(
        go.Box(
            y=df_enhanced[df_enhanced['故障标签']==0]['振动(mm/s)'],
            name='正常',
            marker_color=CUSTOM_COLORS[0],
            boxmean=True
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=df_enhanced[df_enhanced['故障标签']==1]['振动(mm/s)'],
            name='故障',
            marker_color=CUSTOM_COLORS[2],
            boxmean=True
        ),
        row=3, col=2
    )
    
    # 更新布局
    fig.update_layout(
        title_text='风力发电机参数交互式仪表盘',
        height=1000,
        width=1200,
        showlegend=False
    )
    
    fig.write_html('风力发电机参数交互式仪表盘.html')

# 主函数
def main():
    """主函数：执行所有可视化分析"""
    print("正在加载数据...")
    df_enhanced, df_original = load_data()
    
    # 显示数据集基本统计信息
    dataset_stats(df_enhanced, df_original)
    
    print("\n正在进行数据集对比可视化...")
    if df_original is not None:
        plot_dataset_comparison(df_enhanced, df_original)
    
    print("正在可视化物理关系...")
    plot_physics_relationships(df_enhanced)
    
    print("正在分析故障模式...")
    plot_fault_analysis(df_enhanced)
    
    print("正在计算特征重要性...")
    plot_feature_importance(df_enhanced)

    # 新增调用
    plot_feature_discrimination_comparison(df_enhanced)
    
    print("正在创建交互式可视化...")
    create_interactive_visualizations(df_enhanced)
    
    print("\n可视化分析完成！所有结果已保存到当前目录。")

if __name__ == "__main__":
    main() 