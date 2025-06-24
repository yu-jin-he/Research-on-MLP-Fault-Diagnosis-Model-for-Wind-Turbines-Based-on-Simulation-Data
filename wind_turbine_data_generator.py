import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import math
from scipy.stats import weibull_min

# 设置随机种子，保证结果可重复
np.random.seed(42)
random.seed(42)

# 定义风机物理参数常量
class WindTurbineParams:
    """风力发电机物理参数"""
    # 风机物理参数
    ROTOR_DIAMETER = 90  # 风轮直径(m)
    HUB_HEIGHT = 80      # 轮毂高度(m)
    RATED_POWER = 2000   # 额定功率(kW)
    CUT_IN_SPEED = 3.0   # 切入风速(m/s)
    RATED_SPEED = 12.0   # 额定风速(m/s)
    CUT_OUT_SPEED = 25.0 # 切出风速(m/s)
    MAX_RPM = 22.0       # 最大转速(rpm)
    CP_MAX = 0.45        # 最大功率系数
    AIR_DENSITY = 1.225  # 空气密度(kg/m³)
    
    # 传动系统参数
    GEAR_RATIO = 91.0    # 齿轮箱传动比
    
    # 温度参数
    AMBIENT_TEMP_RANGE = {
        'spring': (10, 25),    # 春季温度范围
        'summer': (20, 35),    # 夏季温度范围
        'autumn': (10, 20),    # 秋季温度范围
        'winter': (-10, 10)    # 冬季温度范围
    }
    
    # 故障率参数
    BASE_FAILURE_RATE = 0.003  # 基础故障率
    
    # 传感器误差参数
    SENSOR_NOISE = {
        'wind_speed': 0.1,     # 风速传感器噪声
        'rpm': 0.05,           # 转速传感器噪声
        'power': 5.0,          # 功率传感器噪声
        'temp': 0.2,           # 温度传感器噪声
        'vibration': 0.05,     # 振动传感器噪声
        'oil_temp': 0.3,       # 油温传感器噪声
        'oil_level': 0.5       # 油位传感器噪声
    }

# 定义辅助函数
def get_season(date):
    """根据日期确定季节"""
    month = date.month
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'

def get_day_factor(hour):
    """获取一天中的时间因子，影响风速"""
    # 模拟一天中风速的变化：凌晨最低，中午最高
    return 0.8 + 0.4 * math.sin(math.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.8 # 6点到18点

def generate_wind_speed(date, base_speed):
    """根据日期和基础风速生成符合现实的风速"""
    season = get_season(date)
    hour = date.hour
    
    # 季节影响因子
    season_factors = {
        'spring': 1.1, # 春季风速较高
        'summer': 0.9, # 夏季风速较低
        'autumn': 1.0, # 秋季风速适中
        'winter': 1.2  # 冬季风速较高
    }
    
    # 日变化影响因子
    day_factor = get_day_factor(hour) # 6点到18点
    
    # 随机波动（韦伯分布更符合风速分布特性）
    random_factor = weibull_min.rvs(2, loc=0, scale=0.5) # 韦伯分布
    
    # 计算风速
    wind_speed = base_speed * season_factors[season] * day_factor * random_factor # 季节、日变化、随机波动
    
    # 增加小概率的极端天气
    if random.random() < 0.01:  # 1%概率出现大风
        wind_speed *= random.uniform(1.5, 2.0) # 大风
    
    return wind_speed

def calculate_power_output(wind_speed, params, failure_mode=None):
    """根据风速和风机参数计算功率输出"""
    # 计算扫风面积
    area = math.pi * (params.ROTOR_DIAMETER / 2) ** 2
    
    # 根据风速确定功率系数Cp
    if wind_speed < params.CUT_IN_SPEED:
        cp = 0
    elif wind_speed < params.RATED_SPEED:
        # Cp随风速变化，接近额定风速时达到最大
        cp = params.CP_MAX * (1 - math.exp(-0.5 * (wind_speed - params.CUT_IN_SPEED)))
    elif wind_speed <= params.CUT_OUT_SPEED:
        cp = params.CP_MAX
    else:
        cp = 0  # 超过切出风速，风机停机
    
    # 贝兹公式计算理论功率(W)
    theoretical_power = 0.5 * params.AIR_DENSITY * area * cp * (wind_speed ** 3)
    
    # 转换为kW并考虑效率损失
    power_kw = theoretical_power / 1000 * 0.95
    
    # 功率限制在额定功率以下
    power_kw = min(power_kw, params.RATED_POWER)
    
    # 如果存在电气故障，降低输出功率
    if failure_mode == 'electrical':
        power_kw *= random.uniform(0.7, 0.9)
    
    # 添加随机波动
    power_kw *= random.uniform(0.98, 1.02)
    
    return max(0, power_kw)

def calculate_rotor_rpm(wind_speed, params, failure_mode=None):
    """根据风速计算风轮转速"""
    if wind_speed < params.CUT_IN_SPEED:
        rpm = 0
    elif wind_speed < params.RATED_SPEED:
        # 转速与风速近似线性关系
        rpm = params.MAX_RPM * (wind_speed - params.CUT_IN_SPEED) / (params.RATED_SPEED - params.CUT_IN_SPEED)
    elif wind_speed <= params.CUT_OUT_SPEED:
        rpm = params.MAX_RPM
    else:
        rpm = 0  # 超过切出风速，风机停机
    
    # 添加随机波动
    rpm *= random.uniform(0.98, 1.02) # 转速波动
    
    # 如果存在机械故障，增加转速波动
    if failure_mode == 'mechanical': # 机械故障
        rpm *= random.uniform(0.85, 1.15) # 转速波动
    
    return max(0, rpm)

def calculate_temperature(ambient_temp, power_output, running_hours, params, failure_mode=None):
    """计算各部件温度"""
    # 基础温度受环境温度影响
    base_temp = ambient_temp # 基础温度
    
    # 发电机温度 = 基础温度 + 功率引起的温升
    generator_temp_rise = power_output / params.RATED_POWER * 30 # 发电机温度上升
    generator_temp = base_temp + generator_temp_rise # 发电机温度
    
    # 油温 = 基础温度 + 运行时间和功率引起的温升
    oil_temp_rise = (power_output / params.RATED_POWER * 20) + (min(running_hours, 12) / 12 * 5)
    oil_temp = base_temp + oil_temp_rise
    
    # 齿轮箱油温 = 基础温度 + 运行时间和功率引起的温升
    gearbox_oil_temp_rise = (power_output / params.RATED_POWER * 25) + (min(running_hours, 12) / 12 * 8) # 齿轮箱油温上升
    gearbox_oil_temp = base_temp + gearbox_oil_temp_rise # 齿轮箱油温
    
    # 如果存在润滑系统故障，增加油温
    if failure_mode == 'lubrication': # 润滑系统故障
        oil_temp += random.uniform(5, 15) # 油温增加
        gearbox_oil_temp += random.uniform(10, 20) # 齿轮箱油温增加
    
    # 如果存在电气故障，增加发电机温度
    if failure_mode == 'electrical': # 电气故障
        generator_temp += random.uniform(10, 20) # 发电机温度增加
    
    # 添加传感器噪声
    generator_temp += np.random.normal(0, params.SENSOR_NOISE['temp'])
    oil_temp += np.random.normal(0, params.SENSOR_NOISE['oil_temp'])
    gearbox_oil_temp += np.random.normal(0, params.SENSOR_NOISE['oil_temp'])
    
    return generator_temp, oil_temp, gearbox_oil_temp

def calculate_vibration(rpm, running_hours, params, failure_mode=None):
    """计算振动值"""
    # 振动基础值与转速的平方成正比
    base_vibration = 0.5 + (rpm / params.MAX_RPM) ** 2 * 1.0
    
    # 随时间轻微增加
    time_factor = 1.0 + running_hours / 1000 # 时间因素
    vibration = base_vibration * time_factor # 振动
    
    # 如果存在机械故障，大幅增加振动
    if failure_mode == 'mechanical': # 机械故障
        vibration *= random.uniform(1.5, 3.0) # 振动增加
    
    # 添加传感器噪声
    vibration += np.random.normal(0, params.SENSOR_NOISE['vibration']) # 振动传感器噪声
    
    return max(0, vibration)

def calculate_oil_levels(running_hours, params, failure_mode=None):
    """计算油位数据"""
    # 基础油位，随时间缓慢下降
    base_oil_level = 100 - (running_hours / 1000 * 5) 
    base_oil_level = max(80, base_oil_level)  # 正常不会低于80%
    
    # 齿轮箱油位
    gearbox_oil_level = base_oil_level - random.uniform(0, 3)
    
    # 液压油位
    oil_level = base_oil_level - random.uniform(0, 2) 
    
    # 如果存在润滑系统故障，大幅降低油位
    if failure_mode == 'lubrication': # 润滑系统故障
        oil_level -= random.uniform(10, 20) # 油位降低
        gearbox_oil_level -= random.uniform(15, 25) # 齿轮箱油位降低
    
    # 添加传感器噪声
    oil_level += np.random.normal(0, params.SENSOR_NOISE['oil_level']) # 油位传感器噪声
    gearbox_oil_level += np.random.normal(0, params.SENSOR_NOISE['oil_level']) # 齿轮箱油位传感器噪声
    
    return max(0, min(100, oil_level)), max(0, min(100, gearbox_oil_level))

def calculate_control_params(wind_speed, wind_direction, params, failure_mode=None):
    """计算控制系统参数：偏航角度、偏航速度、桨距角"""
    # 偏航角度：跟踪风向
    yaw_angle = wind_direction + random.uniform(-5, 5)
    
    # 偏航速度：通常较低
    yaw_speed = random.uniform(0, 3)
    
    # 桨距角：随风速调整
    if wind_speed < params.CUT_IN_SPEED:
        pitch_angle = 90  # 风速低时最大桨距角，减少阻力
    elif wind_speed < params.RATED_SPEED:
        # 风速低于额定风速时，桨距角减小以增加能量捕获
        pitch_angle = 90 - ((wind_speed - params.CUT_IN_SPEED) / (params.RATED_SPEED - params.CUT_IN_SPEED)) * 85
    else:
        # 风速高于额定风速时，增加桨距角以限制功率
        pitch_angle = 5 + ((wind_speed - params.RATED_SPEED) / (params.CUT_OUT_SPEED - params.RATED_SPEED)) * 40
    
    # 如果存在控制系统故障，产生异常偏差
    if failure_mode == 'control': # 控制系统故障
        yaw_angle += random.uniform(-45, 45) # 偏航角度
        yaw_speed += random.uniform(2, 8) # 偏航速度
        pitch_angle += random.uniform(-20, 20) # 桨距角
    
    # 确保角度在合理范围内
    yaw_angle = yaw_angle % 360 # 偏航角度
    yaw_speed = max(0, min(10, yaw_speed)) # 偏航速度
    pitch_angle = max(0, min(90, pitch_angle)) # 桨距角
    
    return yaw_angle, yaw_speed, pitch_angle

def detect_failure(current_data, previous_data, running_hours, params):
    """检测是否发生故障及故障类型"""
    # 基础故障率
    base_failure_rate = params.BASE_FAILURE_RATE 
    
    # 故障类型及其权重
    failure_types = {
        'mechanical': 0.35,   # 机械故障
        'electrical': 0.30,   # 电气故障
        'lubrication': 0.20,  # 润滑系统故障
        'control': 0.15       # 控制系统故障
    }
    
    # 故障概率随运行时间增加
    time_factor = 1.0 + running_hours / 5000 # 运行时间增加故障概率
    
    # 高功率运行增加故障概率
    power_factor = 1.0 # 功率因素
    if 'power_output' in current_data and current_data['power_output'] > 0.9 * params.RATED_POWER: # 如果功率输出大于额定功率的90%，则增加故障概率
        power_factor = 1.2 # 增加故障概率
    
    # 高温运行增加故障概率
    temp_factor = 1.0 # 温度因素
    if 'generator_temp' in current_data and current_data['generator_temp'] > 80: # 如果发电机温度大于80℃，则增加故障概率
        temp_factor = 1.3 # 增加故障概率
    
    # 高振动增加故障概率
    vibration_factor = 1.0 # 振动因素
    if 'vibration' in current_data and current_data['vibration'] > 2.5: # 如果振动大于2.5mm/s，则增加故障概率
        vibration_factor = 1.5 # 增加故障概率
    
    # 计算总故障概率
    total_failure_prob = base_failure_rate * time_factor * power_factor * temp_factor * vibration_factor # 总故障概率
    
    # 判断是否发生故障
    if random.random() < total_failure_prob: # 如果随机数小于总故障概率，则发生故障
        # 选择故障类型
        failure_probs = list(failure_types.values()) # 故障类型概率
        failure_types_list = list(failure_types.keys()) # 故障类型列表
        selected_index = random.choices(range(len(failure_types_list)), weights=failure_probs)[0] # 根据概率选择故障类型
        return failure_types_list[selected_index] # 返回故障类型
    
    return None

def generate_wind_turbine_data(num_samples, params=None):
    """生成风力发电机数据"""
    if params is None: # 如果params为None，则使用默认参数
        params = WindTurbineParams() # 使用默认参数
    
    # 初始化数据存储列表
    data = []
    
    # 设置起始时间
    start_time = datetime(2023, 1, 1, 0, 0, 0) # 2023年1月1日0时0分0秒
    
    # 初始化参数
    base_wind_speed = random.uniform(7, 12)  # 基础风速
    running_hours = 0  # 累计运行时间
    current_failure_mode = None  # 当前故障模式
    failure_duration = 0  # 故障持续时间
    previous_data = {}  # 上一时刻数据
    
    # 循环生成数据
    for i in range(num_samples):
        # 计算当前时间
        current_time = start_time + timedelta(minutes=i*10)
        
        # 获取当前季节
        season = get_season(current_time)
        
        # 计算环境温度
        temp_range = params.AMBIENT_TEMP_RANGE[season]
        ambient_temp = random.uniform(temp_range[0], temp_range[1])
        
        # 更新基础风速（带有自相关性）
        if random.random() < 0.1:  # 10%概率更新基础风速
            base_wind_speed = random.uniform(7, 12)
        
        # 生成风速
        wind_speed = generate_wind_speed(current_time, base_wind_speed)
        
        # 生成风向
        wind_direction = (previous_data.get('wind_direction', random.uniform(0, 360)) + 
                         random.uniform(-15, 15)) % 360
        
        # 检查是否发生新故障或故障是否结束
        if current_failure_mode is None:
            current_failure_mode = detect_failure(previous_data, None, running_hours, params)
            if current_failure_mode:
                failure_duration = random.randint(6, 24)  # 故障持续6-24个时间步
        else:
            failure_duration -= 1
            if failure_duration <= 0:
                current_failure_mode = None
        
        # 计算功率输出
        power_output = calculate_power_output(wind_speed, params, current_failure_mode)
        
        # 计算转速
        rotation_speed = calculate_rotor_rpm(wind_speed, params, current_failure_mode)
        
        # 累计运行时间
        if rotation_speed > 0:
            running_hours += 1/6  # 每10分钟累计1/6小时
        
        # 计算温度
        generator_temp, oil_temp, gearbox_oil_temp = calculate_temperature(
            ambient_temp, power_output, running_hours % 24, params, current_failure_mode)
        
        # 计算振动
        vibration = calculate_vibration(rotation_speed, running_hours, params, current_failure_mode)
        
        # 计算油位
        oil_level, gearbox_oil_level = calculate_oil_levels(running_hours, params, current_failure_mode)
        
        # 计算控制参数
        yaw_angle, yaw_speed, pitch_angle = calculate_control_params(
            wind_speed, wind_direction, params, current_failure_mode) # 控制参数
        
        # 计算气压和湿度
        air_pressure = random.uniform(980, 1030) # 气压
        humidity = random.uniform(30, 90) # 湿度
        
        # 故障标签（0表示正常，1表示故障）
        failure_label = 1 if current_failure_mode else 0
        
        # 存储当前数据用于下一时刻计算
        previous_data = {
            'wind_speed': wind_speed, # 风速
            'power_output': power_output, # 功率输出
            'rotation_speed': rotation_speed, # 转速
            'generator_temp': generator_temp, # 发电机温度
            'vibration': vibration, # 振动
            'wind_direction': wind_direction, # 风向
        }
        
        # 添加数据到列表
        data.append([
            current_time, # 时间
            wind_speed, # 风速
            rotation_speed, # 转速
            power_output, # 功率输出
            ambient_temp, # 环境温度
            vibration, # 振动
            oil_temp, # 油温
            oil_level, # 油位
            gearbox_oil_temp, # 齿轮箱油温
            gearbox_oil_level, # 齿轮箱油位
            generator_temp, # 发电机温度
            yaw_angle, # 偏航角度
            yaw_speed, # 偏航速度
            pitch_angle, # 桨距角
            wind_direction, # 风向
            air_pressure, # 气压
            humidity, # 湿度
            failure_label # 故障标签
        ])
    
    # 创建DataFrame
    columns = [
        '时间',
        '风速(m/s)',
        '转速(rpm)',
        '功率输出(kW)',
        '温度(℃)',
        '振动(mm/s)',
        '油温(℃)',
        '油位(%)',
        '齿轮箱油温(℃)',
        '齿轮箱油位(%)',
        '发电机温度(℃)',
        '偏航角度(度)',
        '偏航速度(度/s)',
        '桨距角(度)',
        '风向(度)',
        '气压(hPa)',
        '湿度(%)',
        '故障标签'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    return df

def print_data_statistics(df):
    """打印数据集统计信息"""
    # 基本信息
    print("\n数据集统计信息:")
    print(f"总样本数: {len(df)}")
    print(f"故障样本数: {df['故障标签'].sum()} ({df['故障标签'].mean()*100:.2f}%)")
    
    # 数值列统计信息
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col != '故障标签':
            print(f"\n{col} 统计:")
            print(f"  最小值: {df[col].min():.2f}")
            print(f"  最大值: {df[col].max():.2f}")
            print(f"  平均值: {df[col].mean():.2f}")
            print(f"  标准差: {df[col].std():.2f}")
    
    # 故障与非故障样本统计对比
    print("\n故障与非故障样本对比:")
    fault_df = df[df['故障标签'] == 1] # 故障样本
    normal_df = df[df['故障标签'] == 0] # 正常样本
    
    key_cols = ['风速(m/s)', '转速(rpm)', '功率输出(kW)', '振动(mm/s)', '发电机温度(℃)']
    for col in key_cols:
        print(f"\n{col}:")
        print(f"  故障样本平均值: {fault_df[col].mean():.2f}")
        print(f"  正常样本平均值: {normal_df[col].mean():.2f}")
        print(f"  差异百分比: {((fault_df[col].mean() - normal_df[col].mean()) / normal_df[col].mean())*100:.2f}%")

# 主函数
if __name__ == "__main__":
    # 生成数据
    num_samples = 10000
    wind_turbine_data = generate_wind_turbine_data(num_samples)
    
    # 保存为CSV文件
    wind_turbine_data.to_csv('enhanced_wind_turbine_data.csv', index=False, encoding='utf-8-sig')
    
    # 打印统计信息
    print_data_statistics(wind_turbine_data)
    
    print("\n增强版风力发电机数据文件生成完成，文件名为：enhanced_wind_turbine_data.csv")
    print("该数据集包含物理约束关系、多样化故障模式、季节和日变化特性，更适合PINN模型训练。") 