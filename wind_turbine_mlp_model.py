import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import matplotlib.font_manager as fm

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = 'sans-serif'    # 用来正常显示中文标签

# 检查字体是否可用，如果不可用则尝试其他中文字体
font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
font_found = False
for font in font_list:
    if any(f.name == font for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [font]
        print(f"使用中文字体: {font}")
        font_found = True
        break

if not font_found:
    print("警告：未找到合适的中文字体，图表中文可能显示为方框")
    # 尝试使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)



class WindTurbineModel:
    """风力发电机故障预测模型，从简单MLP开始，逐步添加物理约束"""
    
    def __init__(self, data_path='enhanced_wind_turbine_data.csv'):
        """初始化模型"""
        self.data_path = data_path # 数据路径
        self.data = None # 数据集
        self.X = None # 输入特征
        self.y = None # 标签
        self.X_train = None # 训练集输入特征
        self.X_test = None # 测试集输入特征
        self.y_train = None # 训练集标签
        self.y_test = None # 测试集标签
        self.scaler = StandardScaler() # 标准化器
        self.model = None # 模型
        self.history = None # 训练历史
        self.important_features = [
            'yaw_speed', 'oscillation', 'gen_temperature',  # 偏航速度、振动、发电机温度
            'gearbox_oil_level', 'oil_temp', 'gearbox_oil_temp', # 齿轮箱油位、油温、齿轮箱油温
            'wind_speed', 'rotation_speed', 'power_output' # 风速、转速、功率输出
        ]
        
    def load_data(self):
        """加载并预处理数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据集大小: {self.data.shape}")
        
        # 检查是否有故障标签列 - 同时检查英文"fault"和中文"故障标签"
        if 'fault' not in self.data.columns and '故障标签' not in self.data.columns:
            print("警告: 数据集中没有'fault'或'故障标签'列，尝试查找其他可能的故障标签列")
            fault_columns = [col for col in self.data.columns if 'fault' in col.lower() or '故障' in col]
            if fault_columns:
                print(f"找到可能的故障列: {fault_columns[0]}")
                self.data['fault'] = self.data[fault_columns[0]]
            else:
                raise ValueError("数据集中没有找到故障标签列")
        elif '故障标签' in self.data.columns:
            # 如果找到中文列名，创建一个英文别名
            print("找到中文故障标签列'故障标签'，创建英文别名'fault'")
            self.data['fault'] = self.data['故障标签']
        
        # 查看故障样本比例
        fault_ratio = self.data['fault'].mean()
        print(f"故障样本比例: {fault_ratio:.4f} ({int(fault_ratio * len(self.data))} / {len(self.data)})")
        
        # 提取特征和标签
        # 先检查重要特征是否都在数据集中
        missing_features = [f for f in self.important_features if f not in self.data.columns] 
        if missing_features:
            print(f"警告: 数据集中缺少以下重要特征: {missing_features}")
            # 尝试匹配中文列名到英文特征名
            chinese_to_english = {
                '偏航速度(度/s)': 'yaw_speed',
                '振动(mm/s)': 'oscillation',
                '发电机温度(℃)': 'gen_temperature',
                '齿轮箱油位(%)': 'gearbox_oil_level',
                '油温(℃)': 'oil_temp',
                '齿轮箱油温(℃)': 'gearbox_oil_temp',
                '风速(m/s)': 'wind_speed',
                '转速(rpm)': 'rotation_speed',
                '功率输出(kW)': 'power_output'
            }
            
            # 创建映射的副本列
            for zh_col, en_col in chinese_to_english.items(): # 遍历中文列名和英文特征名
                if zh_col in self.data.columns and en_col in missing_features: # 如果中文列名在数据集中，并且英文特征名在缺失特征列表中
                    print(f"将中文列 '{zh_col}' 映射到英文特征 '{en_col}'") # 打印映射信息
                    self.data[en_col] = self.data[zh_col] # 将中文列映射到英文特征
            
            # 更新缺失特征列表
            missing_features = [f for f in self.important_features if f not in self.data.columns]
            if missing_features:
                print(f"警告: 映射后仍缺少以下重要特征: {missing_features}")
                # 使用可用的重要特征
                self.important_features = [f for f in self.important_features if f in self.data.columns]
        
        # 使用重要特征作为输入
        self.X = self.data[self.important_features]
        self.y = self.data['fault']
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # 标准化特征
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("数据加载和预处理完成")
        return self
    
    def build_basic_mlp(self):
        """构建基础的多层感知机模型"""
        print("构建基础MLP模型...")
        
        # 获取输入特征数量
        n_features = self.X_train.shape[1] # 获取训练集输入特征的数量
        
        # 构建简单的MLP模型
        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)), # 输入层
            keras.layers.Dense(64, activation='relu'), # 全连接层，64个神经元，ReLU激活函数
            keras.layers.BatchNormalization(), # 批归一化
            keras.layers.Dropout(0.3), # 随机失活，防止过拟合
            keras.layers.Dense(32, activation='relu'), # 全连接层，32个神经元，ReLU激活函数
            keras.layers.BatchNormalization(), # 批归一化
            keras.layers.Dropout(0.2), # 随机失活，防止过拟合
            keras.layers.Dense(16, activation='relu'), # 全连接层，16个神经元，ReLU激活函数
            keras.layers.Dense(1, activation='sigmoid') # 输出层，1个神经元，Sigmoid激活函数
        ])
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), # 优化器，Adam优化器，学习率为0.001
            loss='binary_crossentropy', # 损失函数，二元交叉熵损失
            metrics=['accuracy'] # 评估指标，准确率
        )
        
        self.model = model # 将模型赋值给实例变量
        print("基础MLP模型构建完成") # 打印构建完成信息
        return self # 返回实例对象
    
    def train(self, epochs=50, batch_size=32, patience=10):
        """训练模型"""
        print(f"开始训练模型，epochs={epochs}, batch_size={batch_size}...") # 打印训练信息
        
        # 设置早停
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', # 监控验证损失
            patience=patience, # 早停，验证损失不再下降的epoch数
            restore_best_weights=True # 恢复最佳模型权重
        )
        
        # 训练模型
        self.history = self.model.fit(
            self.X_train, self.y_train, # 训练集输入特征和标签
            epochs=epochs, # 训练轮数
            batch_size=batch_size, # 批量大小
            validation_split=0.2, # 验证集比例
            callbacks=[early_stopping], # 回调函数
            verbose=1 # 打印训练信息
        )
        
        print("模型训练完成") # 打印训练完成信息
        return self
    
    def evaluate(self):
        """评估模型性能"""
        print("评估模型性能...") # 打印评估信息
        
        # 在测试集上预测
        y_pred_prob = self.model.predict(self.X_test) # 预测概率
        y_pred = (y_pred_prob > 0.5).astype(int) # 预测标签
        
        # 计算各种评估指标
        acc = accuracy_score(self.y_test, y_pred) # 准确率
        precision = precision_score(self.y_test, y_pred) # 精确率
        recall = recall_score(self.y_test, y_pred) # 召回率
        f1 = f1_score(self.y_test, y_pred) # F1分数
        auc = roc_auc_score(self.y_test, y_pred_prob) # AUC
        
        print(f"准确率: {acc:.4f}") # 打印准确率
        print(f"精确率: {precision:.4f}") # 打印精确率
        print(f"召回率: {recall:.4f}") # 打印召回率
        print(f"F1分数: {f1:.4f}") # 打印F1分数
        print(f"AUC: {auc:.4f}") # 打印AUC
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14})
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred) # 混淆矩阵
        plt.figure(figsize=(10, 8)) # 设置图像大小
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}) # 绘制混淆矩阵
        plt.title('混淆矩阵', fontsize=18, pad=20) # 设置标题
        plt.ylabel('实际标签', fontsize=16) # 设置y轴标签
        plt.xlabel('预测标签', fontsize=16) # 设置x轴标签
        plt.tight_layout() # 调整布局
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight') # 保存图像
        
        # 绘制ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob) # 计算ROC曲线
        plt.figure(figsize=(10, 8)) # 设置图像大小
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2) # 绘制ROC曲线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2) # 绘制对角线
        plt.xlabel('假正例率 (FPR)', fontsize=16) # 设置x轴标签
        plt.ylabel('真正例率 (TPR)', fontsize=16) # 设置y轴标签
        plt.title('ROC曲线', fontsize=18, pad=20) # 设置标题
        plt.legend(fontsize=14) # 设置图例字体大小
        plt.grid(True, linestyle='--', alpha=0.7) # 绘制网格
        plt.tight_layout() # 调整布局
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight') # 保存图像
        
        # 训练历史
        plt.figure(figsize=(15, 6)) # 设置图像大小
        plt.subplot(1, 2, 1) # 创建子图
        plt.plot(self.history.history['loss'], label='训练损失', linewidth=2) # 绘制训练损失曲线
        plt.plot(self.history.history['val_loss'], label='验证损失', linewidth=2) # 绘制验证损失曲线
        plt.title('损失曲线', fontsize=18, pad=20) # 设置标题
        plt.xlabel('Epoch', fontsize=16) # 设置x轴标签
        plt.ylabel('损失', fontsize=16) # 设置y轴标签
        plt.legend(fontsize=14) # 设置图例字体大小
        plt.grid(True, linestyle='--', alpha=0.7) # 绘制网格
        
        plt.subplot(1, 2, 2) # 创建子图
        
        # 检查是否有accuracy指标
        if 'accuracy' in self.history.history:
            plt.plot(self.history.history['accuracy'], label='训练准确率', linewidth=2)
            if 'val_accuracy' in self.history.history:
                plt.plot(self.history.history['val_accuracy'], label='验证准确率', linewidth=2)
            plt.title('准确率曲线', fontsize=18, pad=20)
        else:
            # 如果没有accuracy，可以绘制其他可用指标
            for metric in ['bce_loss', 'physics_loss']:
                if metric in self.history.history:
                    plt.plot(self.history.history[metric], label=f'训练{metric}', linewidth=2)
            plt.title('训练指标曲线', fontsize=18, pad=20)
        
        
        plt.xlabel('Epoch', fontsize=16) # 设置x轴标签
        plt.ylabel('准确率', fontsize=16) # 设置y轴标签
        plt.legend(fontsize=14) # 设置图例字体大小
        plt.grid(True, linestyle='--', alpha=0.7) # 绘制网格


        plt.tight_layout() # 调整布局
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight') # 保存图像
        
        return {
            'accuracy': acc, # 准确率
            'precision': precision, # 精确率
            'recall': recall, # 召回率
            'f1': f1, # F1分数
            'auc': auc # AUC
        }
    
    def add_physics_constraints(self, physics_weight=0.001):
        """向模型添加真正的物理约束
        
        基于风力发电机的物理规律添加约束：
        1. 风速与功率关系 (功率 ∝ 风速^3)
        2. 风速与转速关系 (近似线性关系)
        3. 温度关系 (发电机温度随功率增加)
        4. 振动与转速关系 (振动 ∝ 转速^2)
        """
        print("添加物理约束到模型...")
        
        try:
            # 获取输入特征数量
            n_features = self.X_train.shape[1]
            
            # 获取特征索引字典，用于在损失函数中定位特征
            feature_indices = {}
            for i, feature in enumerate(self.important_features):
                feature_indices[feature] = i
            
            # 创建自定义模型，实现物理约束的损失函数
            class PhysicsInformedModel(keras.Model):
                def __init__(self, feature_indices, physics_weight=0.001, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.feature_indices = feature_indices
                    self.physics_weight = physics_weight
                    self.bce_loss_tracker = keras.metrics.Mean(name="bce_loss")
                    self.physics_loss_tracker = keras.metrics.Mean(name="physics_loss")
                    self.total_loss_tracker = keras.metrics.Mean(name="loss")
                    self.accuracy_metric = keras.metrics.BinaryAccuracy(name="accuracy")
                    
                @property
                def metrics(self):
                    # 跟踪损失组件和准确率
                    return [
                        self.total_loss_tracker,
                        self.bce_loss_tracker, 
                        self.physics_loss_tracker,
                        self.accuracy_metric
                    ]
                
                def train_step(self, data):
                    # 实现自定义训练步骤
                    x, y = data
                    
                    with tf.GradientTape() as tape:
                        # 前向传播
                        y_pred = self(x, training=True)
                        
                        # 计算标准交叉熵损失 - 添加类别权重
                        # 故障样本权重提高，处理数据不平衡问题
                        class_weight = tf.where(
                            tf.equal(y, 1), 
                            tf.ones_like(y, dtype=tf.float32) * 3.0,  # 给故障样本更高的权重，确保使用浮点类型
                            tf.ones_like(y, dtype=tf.float32)
                        )
                        bce_loss = keras.losses.binary_crossentropy(y, y_pred)
                        bce_loss = bce_loss * class_weight
                        bce_loss = tf.reduce_mean(bce_loss)
                        
                        # 计算物理约束损失
                        physics_loss = self._calculate_physics_loss(x, y_pred, y)
                        
                        # 总损失 = BCE损失 + 物理约束权重 * 物理约束损失
                        total_loss = bce_loss + self.physics_weight * physics_loss
                    
                    # 计算梯度并更新参数
                    trainable_vars = self.trainable_variables
                    gradients = tape.gradient(total_loss, trainable_vars)
                    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                    
                    # 更新度量
                    self.total_loss_tracker.update_state(total_loss)
                    self.bce_loss_tracker.update_state(bce_loss)
                    self.physics_loss_tracker.update_state(physics_loss)
                    self.accuracy_metric.update_state(y, y_pred)
                    
                    return {
                        "loss": self.total_loss_tracker.result(),
                        "bce_loss": self.bce_loss_tracker.result(),
                        "physics_loss": self.physics_loss_tracker.result(),
                        "accuracy": self.accuracy_metric.result()
                    }
                
                def _calculate_physics_loss(self, x, y_pred, y_true):
                    """计算基于物理规律的损失"""
                    physics_loss = 0.0
                    
                    # 创建平衡权重 - 改进权重分配
                    # 对于正常样本(y_true=0)，要求适度遵循物理规律，权重为1.0
                    # 对于故障样本(y_true=1)，更容许偏离物理规律，权重为0.5
                    balance_weight = 1.0 - 0.5 * tf.cast(y_true, tf.float32)  # 将y_true转换为float32类型
                    
                    # 物理约束1: 风速与功率关系 (功率 ∝ 风速^3)
                    if 'wind_speed' in self.feature_indices and 'power_output' in self.feature_indices:
                        wind_idx = self.feature_indices['wind_speed']
                        power_idx = self.feature_indices['power_output']
                        
                        wind_speed = x[:, wind_idx]
                        power_output = x[:, power_idx]
                        
                        # 风速为0时功率为0，风速过大时功率也为0（切出风速）
                        # 计算理论功率与实际功率的偏差
                        cut_in_speed = 3.0
                        rated_speed = 12.0
                        cut_out_speed = 25.0
                        
                        # 创建掩码: 工作范围内的风速
                        valid_wind_mask = tf.logical_and(
                            tf.greater_equal(wind_speed, cut_in_speed),
                            tf.less_equal(wind_speed, cut_out_speed)
                        )
                        valid_wind_mask = tf.cast(valid_wind_mask, tf.float32)
                        
                        # 计算理论功率 (简化模型)
                        # 考虑切入风速、额定风速和切出风速
                        normalized_wind = tf.clip_by_value(
                            (wind_speed - cut_in_speed) / (rated_speed - cut_in_speed),
                            0.0, 1.0
                        )
                        theoretical_power = normalized_wind * normalized_wind * normalized_wind * 2000  # 假设额定功率2000kW
                        
                        # 计算功率偏差，使用平衡权重
                        power_deviation = tf.abs(power_output - theoretical_power)
                        power_physics_loss = tf.reduce_mean(power_deviation * balance_weight * valid_wind_mask)
                        
                        physics_loss += 0.2 * power_physics_loss  # 降低权重为0.2
                    
                    # 物理约束2: 风速与转速关系（近似线性）
                    if 'wind_speed' in self.feature_indices and 'rotation_speed' in self.feature_indices:
                        wind_idx = self.feature_indices['wind_speed']
                        rpm_idx = self.feature_indices['rotation_speed']
                        
                        wind_speed = x[:, wind_idx]
                        rotation_speed = x[:, rpm_idx]
                        
                        cut_in_speed = 3.0
                        rated_speed = 12.0
                        cut_out_speed = 25.0
                        max_rpm = 22.0
                        
                        # 创建掩码：工作范围内的风速
                        valid_wind_mask = tf.logical_and(
                            tf.greater_equal(wind_speed, cut_in_speed),
                            tf.less_equal(wind_speed, cut_out_speed)
                        )
                        valid_wind_mask = tf.cast(valid_wind_mask, tf.float32)
                        
                        # 计算理论转速：线性关系
                        theoretical_rpm = tf.clip_by_value(
                            (wind_speed - cut_in_speed) / (rated_speed - cut_in_speed) * max_rpm,
                            0.0, max_rpm
                        )
                        
                        # 高风速区域限制为最大转速
                        high_wind_mask = tf.cast(tf.greater(wind_speed, rated_speed), tf.float32)
                        theoretical_rpm = high_wind_mask * max_rpm + (1 - high_wind_mask) * theoretical_rpm
                        
                        # 计算转速偏差，使用平衡权重
                        rpm_deviation = tf.abs(rotation_speed - theoretical_rpm)
                        rpm_physics_loss = tf.reduce_mean(rpm_deviation * balance_weight * valid_wind_mask)
                        
                        physics_loss += 0.2 * rpm_physics_loss  # 降低权重为0.2
                    
                    # 物理约束3: 温度关系 (发电机温度随功率增加)
                    if 'gen_temperature' in self.feature_indices and 'power_output' in self.feature_indices:
                        temp_idx = self.feature_indices['gen_temperature']
                        power_idx = self.feature_indices['power_output']
                        
                        gen_temp = x[:, temp_idx]
                        power_output = x[:, power_idx]
                        
                        # 简化的温度-功率关系：温度 = a * 功率 + b
                        a, b = 0.0146, 30.0  # 系数可能需要调整
                        expected_temp = a * power_output + b
                        
                        # 计算温度偏差，使用平衡权重
                        temp_deviation = tf.abs(gen_temp - expected_temp)
                        temp_physics_loss = tf.reduce_mean(temp_deviation * balance_weight)
                        
                        physics_loss += 0.2 * temp_physics_loss  # 降低权重为0.2
                    
                    # 物理约束4: 振动与转速关系 (振动 ∝ 转速^2)
                    if 'oscillation' in self.feature_indices and 'rotation_speed' in self.feature_indices:
                        osc_idx = self.feature_indices['oscillation']
                        rpm_idx = self.feature_indices['rotation_speed']
                        
                        oscillation = x[:, osc_idx]
                        rotation_speed = x[:, rpm_idx]
                        
                        # 振动基础值与转速的平方成正比
                        max_rpm = 22.0
                        expected_oscillation = 0.5 + (rotation_speed / max_rpm) ** 2 * 1.0
                        
                        # 计算振动偏差，使用平衡权重
                        oscillation_deviation = tf.abs(oscillation - expected_oscillation)
                        oscillation_physics_loss = tf.reduce_mean(oscillation_deviation * balance_weight)
                        
                        physics_loss += 0.2 * oscillation_physics_loss  # 降低权重为0.2
                    
                    # 添加故障检测增强 - 故障特征模式识别
                    # 故障通常表现为：振动增大，温度异常，转速异常等
                    if 'oscillation' in self.feature_indices and 'gen_temperature' in self.feature_indices:
                        osc_idx = self.feature_indices['oscillation']
                        temp_idx = self.feature_indices['gen_temperature']
                        
                        oscillation = x[:, osc_idx]
                        gen_temp = x[:, temp_idx]
                        
                        # 计算故障可能性特征
                        # 高振动或高温度都可能指示故障
                        fault_indicator = tf.reduce_mean(
                            tf.cast(y_true, tf.float32) * (1.0 - tf.sigmoid(-(oscillation - 1.5) * 5.0 - (gen_temp - 60.0) * 0.1))
                        )
                        
                        # 增加对故障的敏感度
                        physics_loss -= 0.2 * fault_indicator  # 负值鼓励检测故障
                    
                    # 添加对称约束 - 避免全预测为正常的捷径
                    # 增大权重以确保故障样本被正确预测
                    pred_balance_loss = tf.abs(tf.reduce_mean(y_pred) - tf.reduce_mean(tf.cast(y_true, tf.float32))) * 10.0
                    physics_loss += pred_balance_loss
                    
                    return physics_loss
            
            # 创建基础网络架构
            inputs = keras.layers.Input(shape=(n_features,))
            x = keras.layers.Dense(64, activation='relu')(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(32, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.Dense(16, activation='relu')(x)
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            
            # 创建物理约束模型
            model = PhysicsInformedModel(
                feature_indices=feature_indices,
                physics_weight=physics_weight,
                inputs=inputs,
                outputs=outputs
            )
            
            # 编译模型
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            self.model = model
            print(f"物理约束已添加到模型 (优化版) - 物理约束权重: {physics_weight}")
        
        except Exception as e:
            print(f"添加物理约束时出错: {e}")
            print("使用没有物理约束的模型继续")
            # 回退到基础MLP模型
            self.build_basic_mlp()
        
        return self
    
    def add_feature_interactions(self):
        """添加特征交互项"""
        print("添加特征交互项...")
        
        # 此方法将在数据预处理阶段添加特征交互项
        # 注意：需要在load_data之后，构建模型之前调用
        
        # 提取特征交互项前先检查X是否已经被加载和处理
        if self.X is None or self.X_train is None:
            print("错误: 请先调用load_data()方法加载数据")
            return self
        
        # 获取原始特征矩阵（未标准化）
        X_train_orig = self.scaler.inverse_transform(self.X_train) # 反标准化
        X_test_orig = self.scaler.inverse_transform(self.X_test) # 反标准化
        
        # 创建DataFrame以便于添加交互特征
        X_train_df = pd.DataFrame(X_train_orig, columns=self.important_features) # 创建DataFrame
        X_test_df = pd.DataFrame(X_test_orig, columns=self.important_features) # 创建DataFrame
        
        # 添加交互特征
        interactions = [
            # 根据散点图分析添加重要的交互特征
            ('yaw_speed', 'gearbox_oil_level'),  # 偏航速度和油位
            ('gen_temperature', 'gearbox_oil_temp'),  # 发电机温度和齿轮箱油温
            ('wind_speed', 'power_output'),  # 风速和功率
            ('rotation_speed', 'oscillation')  # 转速和摆动
        ]
        
        for feat1, feat2 in interactions: # 遍历交互特征
            if feat1 in X_train_df.columns and feat2 in X_train_df.columns: # 检查特征是否存在
                interaction_name = f'{feat1}_{feat2}' # 交互特征名称
                X_train_df[interaction_name] = X_train_df[feat1] * X_train_df[feat2] # 添加交互特征
                X_test_df[interaction_name] = X_test_df[feat1] * X_test_df[feat2] # 添加交互特征
                print(f"添加交互特征: {interaction_name}") # 打印添加交互特征信息
            else:
                print(f"警告: 无法添加交互特征 {feat1}_{feat2}，因为特征不可用") # 打印警告信息
        
        # 添加偏差特征 - 温度与功率的关系偏差
        if 'gen_temperature' in X_train_df.columns and 'power_output' in X_train_df.columns: # 检查特征是否存在
            # 简单线性关系: 温度 = a * 功率 + b
            # 这里使用预定义的系数，实际应用中应该通过回归确定
            a, b = 0.0146, 2.7304  # 从之前的分析中获得
            
            # 计算预期温度
            X_train_df['temp_power_deviation'] = X_train_df['gen_temperature'] - (a * X_train_df['power_output'] + b) # 计算预期温度
            X_test_df['temp_power_deviation'] = X_test_df['gen_temperature'] - (a * X_test_df['power_output'] + b) # 计算预期温度
            print("添加物理偏差特征: temp_power_deviation") # 打印添加物理偏差特征信息
        
        # 更新重要特征列表
        self.important_features = list(X_train_df.columns) # 更新重要特征列表
        
        # 重新标准化
        self.X_train = self.scaler.fit_transform(X_train_df) # 重新标准化
        self.X_test = self.scaler.transform(X_test_df) # 重新标准化
        
        print(f"特征交互添加完成，现在有 {self.X_train.shape[1]} 个特征") # 打印特征交互添加完成信息
        return self

def main():
    """主函数"""
    # 创建模型对象
    model = WindTurbineModel() # 创建模型对象
    
    # 1. 基础MLP模型
    print("\n=== 阶段1: 基础MLP模型 ===") # 打印阶段1信息
    model.load_data() # 加载数据
    model.build_basic_mlp() # 构建基础MLP模型
    model.train(epochs=50) # 训练模型
    basic_metrics = model.evaluate() # 评估模型
    
    # 2. 添加特征交互项
    print("\n=== 阶段2: 带特征交互的MLP模型 ===") # 打印阶段2信息
    model = WindTurbineModel()  # 重新初始化模型
    model.load_data() # 加载数据
    model.add_feature_interactions() # 添加特征交互项
    model.build_basic_mlp() # 构建基础MLP模型
    model.train(epochs=50) # 训练模型
    interactions_metrics = model.evaluate() # 评估模型
    
    # 3. 添加物理约束（调优后）
    print("\n=== 阶段3: 带平衡物理约束的MLP模型 ===") # 打印阶段3信息
    model = WindTurbineModel()  # 重新初始化模型
    model.load_data() # 加载数据
    model.add_feature_interactions() # 添加特征交互项
    model.add_physics_constraints(physics_weight=0.001) # 添加物理约束，使用更小的权重0.001
    model.train(epochs=50) # 训练模型
    physics_metrics = model.evaluate() # 评估模型
    
    # 比较不同模型的性能
    print("\n=== 模型性能比较 ===") # 打印模型性能比较信息
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc'] # 评估指标
    metrics_zh = ['准确率', '精确率', '召回率', 'F1分数', 'AUC'] # 评估指标中文
    models = ['基础MLP', '特征交互MLP', '平衡物理约束MLP'] # 模型名称
    results = [basic_metrics, interactions_metrics, physics_metrics] # 模型评估结果
    
    comparison = pd.DataFrame({
        metric: [result[metric] for result in results] # 获取每个模型的评估指标
        for metric in metrics
    }, index=models) # 创建DataFrame
    
    print(comparison) # 打印比较结果
    
    # 保存比较结果到CSV
    comparison.to_csv('model_comparison.csv') # 保存比较结果到CSV
    
    # 绘制比较图
    plt.figure(figsize=(15, 10)) # 设置图像大小
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # 设置不同的颜色，增强可读性
    
    for i, (metric, metric_zh) in enumerate(zip(metrics, metrics_zh)): # 遍历评估指标
        plt.subplot(2, 3, i+1) # 创建子图
        bars = plt.bar(models, comparison[metric], color=colors) # 绘制柱状图
        plt.title(metric_zh, fontsize=18, pad=15) # 设置标题
        plt.xticks(rotation=30, fontsize=14) # 设置x轴标签旋转
        plt.yticks(fontsize=14) # 设置y轴标签字体大小
        plt.ylim(0, 1) # 设置y轴范围
        
        # 在柱状图顶部添加数值标签
        for bar in bars: # 遍历柱状图
            height = bar.get_height() # 获取柱状图高度
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, # 添加数值标签
                    f'{height:.4f}', ha='center', va='bottom', fontsize=12) # 设置标签位置和字体大小
    
    plt.tight_layout(pad=4.0) # 调整布局
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight') # 保存图像
    
    print("模型比较完成，结果已保存到model_comparison.csv和model_comparison.png") # 打印完成信息

if __name__ == "__main__":
    main() 