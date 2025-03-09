from infrastructure.queries import get_fuel_cell_data
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import scipy.io
import uvicorn
import json
import os
import io

app = FastAPI()

try:
    # 驾驶模式识别模型参数
    kmeans_Tr = scipy.io.loadmat('./modules/Transformation_Matrix.mat')['Tr']
    kmeans_centers = scipy.io.loadmat('./modules/Cluster_Centers.mat')['centers']
except FileNotFoundError as e:
    raise RuntimeError(f"模型加载失败: {e}")

def remove_empty_values(data):
    """
    递归移除字典或列表中的空值（空列表、空字符串、None）
    """
    if isinstance(data, dict):
        return {k: remove_empty_values(v) for k, v in data.items() if v or v == 0 or v is False}
    elif isinstance(data, list):
        return [remove_empty_values(v) for v in data if v or v == 0 or v is False]
    else:
        return data

def zscore_outlier(df, time):
    data = df
    pointer = 0
    for i in df:
        if i <= 0:
            del data[pointer]
            pointer = pointer + 1
        else:
            break

    abnormal = []
    time_data = []
    m = np.mean(data)
    sd = np.std(data)
    position = 0
    for u in data:
        position = position + 1
        if u != m:
            z = (u - m) / sd
            if np.abs(z) > 4:
                abnormal.append(u)
                time_data.append(time[position - 1])
    if len(abnormal) > 0:
        return {
            "name": df.name,
            "abnormal_values": abnormal,
            "abnormal_times": time_data
        }
    else:
        return None  # 如果没有异常值，返回 None

def vol_warning(vol_data, time):
    data = vol_data
    pointer = 0
    for i in vol_data:
        if i <= 0:
            del data[pointer]
            pointer = pointer + 1
        else:
            break
    excessive_data = []
    undersized_data = []
    e_time_data = []
    u_time_data = []
    position = 0
    for u in data:
        position = position + 1
        if u > 3.6:
            excessive_data.append(u)
            e_time_data.append(time[position - 1])

    position = 0
    for u in data:
        position = position + 1
        if 0 < u < 2.8:
            undersized_data.append(u)
            u_time_data.append(time[position - 1])

    result = {
        "excessive_voltage": excessive_data,
        "excessive_times": e_time_data,
        "undersized_voltage": undersized_data,
        "undersized_times": u_time_data
    }
    return remove_empty_values(result)  # 移除空值

def temp_warning(temp_data, time):
    excessive_data = []
    e_time_data = []
    pointer = 0
    for u in temp_data:
        pointer = pointer + 1
        if u > 45:
            excessive_data.append(u)
            e_time_data.append(time[pointer - 1])
    result = {
        "excessive_temperature": excessive_data,
        "excessive_times": e_time_data
    }
    return remove_empty_values(result)  # 移除空值

def safety_warning(folder_path):
    results = []
    for files in os.listdir(folder_path):
        file_path = f"{folder_path}/{files}"
        df = pd.read_csv(file_path)
        cell_vol = df.iloc[:, 1]
        cell_time = df.iloc[:, 0]
        cell_temp = df.iloc[:, 2]
        cell_res = df.iloc[:, 4]

        vol_outliers = zscore_outlier(cell_vol, cell_time)
        temp_outliers = zscore_outlier(cell_temp, cell_time)
        res_outliers = zscore_outlier(cell_res, cell_time)
        vol_warnings = vol_warning(cell_vol, cell_time)
        temp_warnings = temp_warning(cell_temp, cell_time)

        result = {
            "file": files,
            "vol_outliers": vol_outliers,
            "temp_outliers": temp_outliers,
            "res_outliers": res_outliers,
            "vol_warnings": vol_warnings,
            "temp_warnings": temp_warnings
        }
        results.append(remove_empty_values(result))  # 移除空值

    return json.dumps(results, indent=4, ensure_ascii=False)  # 将结果转换为 JSON 字符串

class PEMFuelCell:
    STANDARD_PRESSURE = 101325      # 标准大气压 [Pa]
    MOLAR_MASS_H2 = 2.016e-3        # 氢气摩尔质量 [kg/mol]
    LHV_H2 = 120e6                  # 氢气低热值 [J/kg]
    OXYGEN_RATIO = 0.21             # 空气中氧气体积比
    FARADAY_EFFICIENCY = 0.95       # 考虑实际损耗的法拉第效率

    def __init__(self, temp=353.15, pressure=101325):
        """初始化燃料电池参数"""
        self.T = temp               # 工作温度 [K]
        self.P = pressure           # 工作压力 [Pa]
        self.A = 100e-4             # 有效面积 [m²] (100 cm²)
        self.n_cells = 200          # 单电池数量
        self.E0 = 1.23              # 标准理论电压 [V]
        self.R = 8.314              # 理想气体常数 [J/(mol·K)]
        self.F = 96485              # 法拉第常数 [C/mol]
        self.r = 0.2                # 欧姆电阻 [Ω·cm²]
        self.i0 = 1e-4              # 交换电流密度 [A/cm²]
        self.il = 1.5               # 极限电流密度 [A/cm²]
        self.alpha = 0.5            # 电荷转移系数

    def _water_vapor_pressure(self):
        """计算水的饱和蒸气压 (Antoine方程近似)"""
         # Antoine方程系数（水，0-100°C范围）
        A = 8.07131
        B = 1730.63
        C = 233.426
        
        # 温度转换（开尔文 → 摄氏度）
        T_celsius = self.T - 273.15
        
        # 计算饱和蒸气压（mmHg）
        P_mmHg = 10 ** (A - B/(T_celsius + C))
        
        # 单位转换（mmHg → Pa）
        P_Pa = P_mmHg * 133.322
        
        # 对于低温情况返回0（避免负压）
        return max(P_Pa, 0.0)
        # return 47.4e3  # 示例值，需替换为实际计算

    def voltage(self, current_density):
        """计算电堆总电压"""
        if current_density <= 0:
            current_density = 1e-10

        # 分压计算（假设阳极氢气加湿）
        PH2 = (self.P - self._water_vapor_pressure())  # 氢气分压 [Pa]
        PO2 = self.P * self.OXYGEN_RATIO               # 氧气分压 [Pa]
        PH2O = self._water_vapor_pressure()            # 水蒸气分压 [Pa]

        # 能斯特方程（修正压力项包含水蒸气影响）
        pressure_term = (PH2 / self.STANDARD_PRESSURE) * \
                       np.sqrt(PO2 / self.STANDARD_PRESSURE) / \
                       (PH2O / self.STANDARD_PRESSURE)
        E_nernst = self.E0 + (self.R * self.T) / (2 * self.F) * np.log(pressure_term)
        PH2 = self.P
        PO2 = self.P * self.OXYGEN_RATIO
        pressure_term = (PH2/self.STANDARD_PRESSURE) * np.sqrt(PO2/self.STANDARD_PRESSURE)
        E_nernst = self.E0 + (self.R*self.T)/(2*self.F) * np.log(pressure_term)
        

        # 活化过电位（使用改进的电荷转移系数）
        eta_act = (self.R * self.T) / (self.alpha * self.F) * \
                 np.arcsinh(current_density / (2 * self.i0))

        # 欧姆过电位（单位统一为Ω·cm²和A/cm²）
        eta_ohm = current_density * self.r

        # 浓度过电位（修正为氧气还原反应的n=4）
        safe_ratio = np.clip(current_density / self.il, 0.0, 0.9999)
        eta_conc = (self.R * self.T) / (2* self.F) * np.log(1 - safe_ratio)

        # 单电池电压计算
        cell_voltage = E_nernst - eta_act - eta_ohm - eta_conc
        cell_voltage = max(cell_voltage, 0)  # 电压不低于零

        return cell_voltage * self.n_cells  # 电堆总电压

    def power(self, current_density):
        """计算电堆输出功率 [W]"""
        voltage = self.voltage(current_density)
        current = current_density * self.A  # 单电池电流 [A]
        return voltage * current  # 电堆功率 = 总电压 × 电流

    def efficiency(self, current_density):
        """基于LHV的总体效率计算"""
        # 电功率输出
        electrical_power = self.power(current_density)
        
        # 氢气消耗计算
        current = current_density * self.A  # 单电池电流 [A]
        molar_flow = (current * self.n_cells) / (2 * self.F)  # mol H2/s
        mass_flow = molar_flow * self.MOLAR_MASS_H2      # kg H2/s
        thermal_power = mass_flow * self.LHV_H2          # 输入热功率 [W]
        
        if thermal_power == 0:
            return 0.0
        return (electrical_power / thermal_power) * self.FARADAY_EFFICIENCY

    def find_operating_point(self, target_voltage):
        """二分法查找工作点"""
        VOLTAGE_TOLERANCE = 0.1     # 电压允许误差
        MAX_ITERATIONS = 100
        max_current = self.il * (self.A * 1e4)  # 基于极限电流计算最大电流
        
        low, high = 0, max_current
        
        for _ in range(MAX_ITERATIONS):
            mid = (low + high) / 2
            current_density = mid / (self.A * 1e4)  # 转换为A/cm²
            calculated_voltage = self.voltage(current_density)
            
            if abs(calculated_voltage - target_voltage) < VOLTAGE_TOLERANCE:
                return mid
            elif calculated_voltage < target_voltage:
                high = mid
            else:
                low = mid
        
        final_current = (low + high) / 2
        final_voltage = self.voltage(final_current / (self.A * 1e4))
        if abs(final_voltage - target_voltage) > 1:
            raise ValueError("无法找到有效工作点")
        return final_current

class ResponseData(BaseModel):
    voltage: float     # 输出电压 (V)
    current: float     # 输出电流 (A)
    power: float       # 输出功率 (W)
    h2_consumption: float  # 氢气消耗 (g/s)
    efficiency: float  # 电堆效率 (%)
    parameters: dict

# ----------------- 请求体定义 -------------------- #


# ------------------------------------------------ #

# 从数据库获取锂电池数据 -- 范晓非
@app.get("/data/lithium")
async def extract_lithium_data():
    pass

@app.get("/data/fuel")
async def fetch_fuel_cell_data(
    id: Optional[int] = Query(None, description="根据 ID 查询数据"),
    start_time: Optional[str] = Query(None, description="起始时间 (格式: YYYY-MM-DD HH:MM:SS)"),
    end_time: Optional[str] = Query(None, description="结束时间 (格式: YYYY-MM-DD HH:MM:SS)")
):
    try:
        results = get_fuel_cell_data(id, start_time, end_time)
        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

# 锂电池包数学建模
@app.get("/model/lithum")
async def run_lithium_model():
        # 读取数据
    file_name =  "./dataset/segment_614.csv"
    try:
        raw_data = pd.read_csv(file_name)
    except Exception as e:
        return {"error": f"Failed to read the file: {e}"}

    # 数据预处理
    try:
        dt = raw_data.iloc[:, 28].values  # 时间间隔数据
        valid_idx = dt > 0  # 找出时间间隔大于 0 的行
        raw_data = raw_data[valid_idx]  # 删除无效行

        # 重新读取数据
        N = len(raw_data)  # 更新数据点数量
        cur = -raw_data.iloc[:, 3].values  # 电流数据，取负号
        vol = raw_data.iloc[:, 26].values / 1000  # 端电压数据，单位转换为 V
        soc_real = raw_data.iloc[:, 1].values / 100  # SOC 参考值，范围 0-1
        dt = raw_data.iloc[:, 28].values  # 时间间隔数据
        time_seconds = raw_data.iloc[:, 27].values - raw_data.iloc[0, 27]  # 计算时间差
    except Exception as e:
        return {"error": f"Data processing error: {e}"}

    # 参数初始化
    cs = np.array([0.1, 0.1, 0.1])  # 初始参数估计
    P_k = np.eye(3) * 1000  # 初始协方差矩阵
    lambda_ = 0.99  # 遗忘因子
    Q = np.diag([1e-6, 1e-6])  # 过程噪声协方差矩阵
    R = 1e-4  # 测量噪声协方差
    x_hat = np.array([0.5, 0])  # 初始状态估计 [SOC; Vp]
    P_hat = np.diag([1, 1])  # 初始状态协方差矩阵
    soc_est = np.zeros(N)  # 存储 SOC 估计值
    soc_est[0] = 0.9  # SOC 初始预测值为 0.9
    Qn = 34.7 * 3600  # 电池容量
    E_k = np.zeros(N)  # 误差
    Ro_k = np.zeros(N)  # 欧姆内阻
    Rp_k = np.zeros(N)  # 极化内阻
    Cp_k = np.zeros(N)  # 极化电容

    # RLS 参数辨识
    for k in range(1, N):
        soc_k = soc_est[k - 1]
        ocv_k = (
            3.3174 + 0.007 * soc_k + 0.0367 * np.log(soc_k) - 0.0057 * np.log(1 - soc_k)
        )
        E_k[k] = ocv_k - vol[k]
        x_k = np.array([cur[k], E_k[k - 1], cur[k - 1]])

        K_k = P_k @ x_k / (lambda_ + x_k.T @ P_k @ x_k)
        P_k = (P_k - np.outer(K_k, x_k.T @ P_k)) / lambda_
        cs += K_k * (E_k[k] - x_k.T @ cs)

        Ro_k[k] = cs[0]
        Rp_k[k] = (cs[2] + cs[0] * cs[1]) / (1 - cs[1])
        Cp_k[k] = -dt[k] / (np.log(cs[1]) * Rp_k[k])

        A = np.array([[1, 0], [0, np.exp(-dt[k] / (Rp_k[k] * Cp_k[k]))]])
        B = np.array([-dt[k] / Qn, Rp_k[k] * (1 - np.exp(-dt[k] / (Rp_k[k] * Cp_k[k])))])
        x_hat_minus = A @ x_hat + B * cur[k]
        P_minus = A @ P_hat @ A.T + Q

        C = np.array([ocv_k, -1])
        y = vol[k] - (ocv_k - x_hat_minus[1] - Ro_k[k] * cur[k])
        S = C @ P_minus @ C.T + R
        K = P_minus @ C.T / S
        x_hat = x_hat_minus + K * y
        P_hat = (np.eye(2) - np.outer(K, C)) @ P_minus

        x_hat[0] = np.clip(x_hat[0], 0, 1)
        soc_est[k] = np.clip(x_hat[0], 0.001, 0.999)

    # 合并绘图：左右布局
    buffer = io.BytesIO()
    plt.figure(figsize=(16, 8))  # 更宽的画布，适合左右布局

    # 左图：分三行绘制 Ro, Rp, Cp 参数
    plt.subplot(3, 2, 1)
    plt.plot(time_seconds, Ro_k, linewidth=1.5)
    plt.title("Estimated R_o over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("R_o (Ω)")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(time_seconds, Rp_k, linewidth=1.5)
    plt.title("Estimated R_p over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("R_p (Ω)")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(time_seconds, Cp_k, linewidth=1.5)
    plt.title("Estimated C_p over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("C_p (F)")
    plt.grid(True)

    # 右图：绘制 SOC 估计与实际值比较
    plt.subplot(1, 2, 2)
    plt.plot(time_seconds, soc_real, linewidth=1.5, label="Actual SOC")
    plt.plot(time_seconds, soc_est, linewidth=1.5, label="Estimated SOC")
    plt.title("SOC Estimation vs Actual SOC")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.legend()
    plt.grid(True)

    # 调整布局以增加子图之间的间距
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    # 返回合并后的图像
    return StreamingResponse(buffer, media_type="image/png")

# 燃料电池系统数学建模 -- 刘俊，李泽宇
@app.get("/model/fuel")
async def run_fuel_model(
    target_voltage: float = Query(..., gt=0, le=300, title="目标电压"),
    temperature: float = Query(353.15, gt=273.15, le=373.15),
    pressure: float = Query(101325, gt=80000, le=300000),
    oxygen_ratio: float = Query(0.21, gt=0.1, le=1.0)
):
    try:
        fc = PEMFuelCell(temp=temperature, pressure=pressure)
        fc.OXYGEN_RATIO = oxygen_ratio
        
        max_ocv = fc.voltage(1e-10)  # 极小电流近似开路电压
        if target_voltage > max_ocv:
            raise ValueError(f"目标电压不能超过开路电压 {max_ocv:.2f}V")
        
        current = fc.find_operating_point(target_voltage)
        current_density = current / (fc.A * 1e4)
        # voltage_per_cell = fc.voltage(current_density) / fc.n_cells
        # current = current_density * fc.A  # 单电池电流 [A]
        # # 氢气消耗计算
        molar_flow = (current * fc.n_cells) / (2 * fc.F)  # mol H2/s
        h2_consumption = molar_flow * fc.MOLAR_MASS_H2*1000      # g H2/s
        # # 氢气消耗计算
        # h2_molar_flow = current*fc.n_cells / (2 * fc.F)  # mol/s
        # h2_consumption = h2_molar_flow * fc.MOLAR_MASS_H2  # g/s
        
        # 效率计算（使用LHV基于质量）
        # input_power = h2_consumption * (fc.LHV_H2 / 1000)  # 转换为J/g后计算
        # power = current * target_voltage
        # efficiency = (power / input_power) * 100 if input_power > 0 else 0
        fc_power=target_voltage*current
        eff =fc.efficiency(current_density)
        
        return {
            "voltage": round(target_voltage, 2),
            "current": round(current, 2),
            "power": round(fc_power, 2),
            "h2_consumption": round(h2_consumption, 6),
            "efficiency": round(eff, 2),
            "parameters": {
                "temperature_K": round(temperature, 2),
                "pressure_kPa": round(pressure/1000, 2),
                "oxygen_ratio": round(oxygen_ratio, 3)
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 车辆动力学数学建模 -- 朱展锋
@app.get("/model/car")
async def run_car_model(
        v_sequence: str = Query(..., alias="v", description="速度序列(m/s)"),
        A: float = Query(2.0, alias="S", description="车辆迎风面积(默认2m²)"),
        efficiency: float = Query(0.95, alias="η", description="传动效率(默认0.95)"),
        M: float = Query(1500.0, description="车辆质量(默认1500kg)"),
        g: float = Query(9.8, description="重力加速度(默认9.8m/s²)"),
        delta: float = Query(1.04, alias="Δ", description="旋转质量换算系数(默认1.04)"),
        rolling_resistance_coefficient: float = Query(0.015, alias="μ", description="滚动阻力系数(默认0.015)"),
        drag_coefficient: float = Query(0.3, alias="C", description="空气阻力系数(默认0.3)"),
        theta: float = Query(0.0, alias="θ", description="道路坡度(默认为0)"),
        air_density: float = Query(1.202, alias="ρ", description="空气密度(默认1.202kg/m³)"),
        a_sequence: Optional[str] = Query(None, alias="a", description="加速度序列(默认通过速度序列计算)")
):
    try:
        # 解析速度序列
        v_list = [float(v) for v in v_sequence.split()]  # 将字符串 "10 20 30" 转换为列表 [10, 20, 30]

        # 解析加速度序列（如果提供）
        a_list = [float(a) for a in a_sequence.split()] if a_sequence else None

        # 处理逻辑
        v_array = np.array(v_list)
        a_array = np.array(a_list) if a_list else np.concatenate(([0], np.diff(v_array)))

        # 计算功率
        rolling_resistance = rolling_resistance_coefficient * M * g * np.cos(theta)
        gradient_resistance = M * g * np.sin(theta)
        air_resistance = 0.5 * A * air_density * drag_coefficient * v_array ** 2
        acceleration_resistance = delta * M * a_array

        Pre = (rolling_resistance + gradient_resistance + air_resistance + acceleration_resistance) * v_array
        Pe = np.where(Pre > 0, Pre / efficiency, Pre * efficiency)

        structured_data = [
            {
                "time": i + 1,
                "velocity": v_list[i],
                "power": round(Pe[i], 2)
            }
            for i in range(len(v_list))
        ]
        return {"data": structured_data}
    
    except Exception as e:
        raise HTTPException(400, detail=f"错误: {str(e)}")
    

# 基于遗传算法的模型参数标定方法 -- 陆晓蒙
@app.get("/algorithm/litium/calibration/genetic")
async def run_genetic_algorithm():
    pass

# 基于马尔科夫链预测模型的车辆速度预测算法 -- 刘俊
@app.get("/algorithm/car/speed_predition/markov")
async def run_markov_algorithm():
    pass

# 基于K-means聚类算法车辆工况识别算法 -- 朱展锋
@app.get("/algorithm/car/condition_recognition/kmeans")
async def run_kmeans_algorithm(
        speed: str = Query(..., alias="V",
                           description="速度采样序列（单位：m/s）"),
        window_size: int = Query(100, alias="T",
                                 description="采样窗口长度（默认100s）")
):
    try:
        # 加载输入数据并进行单位转换（m/s -> km/h）
        speed_kmh = 3.6 * np.array([float(v) for v in speed.split()])
        total_points = len(speed_kmh)

        if total_points < window_size:
            raise HTTPException(400,
                                detail=f"速度序列长度需≥{window_size}（当前：{total_points}）")

        # 特征提取（保持原有逻辑）
        L = len(speed_kmh)
        features = np.zeros((L, 12))

        for i in range(window_size - 1, L):
            window = speed_kmh[i - window_size + 1: i + 1]
            delta_v = np.diff(window)
            pos_acc = delta_v[delta_v > 0]
            neg_acc = -delta_v[delta_v < 0]

            features[i, 0] = np.mean(window)
            features[i, 1] = np.max(window)
            features[i, 2:5] = [np.mean(pos_acc), np.max(pos_acc), np.std(pos_acc)] if pos_acc.size else 0
            features[i, 5:8] = [np.mean(neg_acc), np.max(neg_acc), np.std(neg_acc)] if neg_acc.size else 0
            features[i, 8] = len(pos_acc) / window_size
            features[i, 9] = len(neg_acc) / window_size

        # 聚类分析（保持原有逻辑）
        transformed = features @ kmeans_Tr
        clusters = np.argmin(cdist(transformed[window_size - 1:],
                                   kmeans_centers), axis=1) + 1

        # 生成完整标签序列（前window_size-1个点补默认值1）
        full_labels = np.concatenate([
            np.ones(window_size - 1, dtype=int),  # t=1~99s默认模式1
            clusters  # t=100s~正常识别结果
        ])

        # 构建结构化输出
        structured_data = []
        for idx in range(total_points):
            structured_data.append({"time": idx + 1, "velocity": round(speed_kmh[idx]/3.6, 2), "pattern": int(full_labels[idx])})
        # 模式分布统计
        unique, counts = np.unique(full_labels, return_counts=True)
        pattern_dist = {f"pattern_{int(k)}": int(v) for k, v in zip(unique, counts)}

        return {
            "data": structured_data,
            "pattern_statistics": pattern_dist
        }
    except ValueError:
        raise HTTPException(400, detail="速度序列格式错误，请使用空格分隔数字")
    except Exception as e:
        raise HTTPException(500, detail=f"算法执行错误: {str(e)}")

# 锂电池一致性评价算法 -- 王世超
@app.get("/algorithm/lithium/battery_consistency_assessment")
async def run_battery_consistency_assessment():
    dataset_folder = "./dataset/consistency_evaluation"
    excel_files = [f for f in os.listdir(dataset_folder) if f.endswith('.xlsx') or f.endswith('.xls')]
    excel_files.sort()
    df1 = None
    # 读取每个Excel文件并合并数据
    for file in excel_files:
        file_path = os.path.join(dataset_folder, file)
        data = pd.read_excel(file_path)  # 读取Excel文件
        device_sn = file.split('.')[0]
        if df1 is None:  # 初始化DataFrame
            df1 = pd.DataFrame(data['Time'], columns=['Time'])
        df1[device_sn] = data['单体电压']
    all_cell_voltage = []
    for device_sn in df1.columns[1:]:
        new_list = df1[device_sn].dropna().tolist()
        new_list = [float(item) for item in new_list if item != 0]
        mean_data = round(sum(new_list) / len(new_list), 2)
        all_cell_voltage.append(mean_data)
    voltage1 = all_cell_voltage
    # 极差系数计算
    sum_data = sum(voltage1)  # 求和
    mean_data = sum_data / len(voltage1)  # 求平均值
    polar_deviation_factor = round((max(voltage1) - min(voltage1)) / mean_data * 100, 3)  # [(Umax-Umin)/Up]*100
    # 标准差系数计算
    variance = sum((val - mean_data) ** 2 for val in voltage1) / len(voltage1)  # Σ（Ui-Up）²/n
    standard_deviation = (variance ** 0.5) / mean_data  # 开根（Σ（Ui-Um）²/n）/Up
    sd = round(standard_deviation * 100, 3)  # 开根（Σ（Ui-Um）²/n）/Up*100
    # 一致性判断评估
    evaluation_score1 = 0
    evaluation_score2 = 0
    if polar_deviation_factor <= 4.9:
        evaluation_score1 = 1
    elif 4.9 < polar_deviation_factor <= 8.2:
        evaluation_score1 = 2
    elif 8.2 < polar_deviation_factor <= 11.5:
        evaluation_score1 = 3
    elif 11.5 < polar_deviation_factor <= 15:
        evaluation_score1 = 4
    elif 15 < polar_deviation_factor <= 18:
        evaluation_score1 = 5
    if sd <= 1.5:
        evaluation_score2 = 1
    elif 1.5 < sd <= 2.5:
        evaluation_score2 = 2
    elif 2.5 < sd <= 3.5:
        evaluation_score2 = 3
    elif 3.5 < sd <= 4.5:
        evaluation_score2 = 4
    elif 4.5 < sd <= 5.5:
        evaluation_score2 = 5
    evaluation_score = evaluation_score1 + evaluation_score2
    if evaluation_score <= 2:
        grade = 'A(优秀）'
    elif 2 < evaluation_score <= 4:
        grade = 'B（较优秀）'
    elif 4 < evaluation_score <= 6:
        grade = 'C（良好）'
    elif 6 < evaluation_score <= 8:
        grade = 'D（中等）'
    elif 8 < evaluation_score <= 10:
        grade = 'E（合格）'
    else:
        grade = 'F（不合格）'
    return {
        "Range":polar_deviation_factor,
        "Standard_deviation":sd,
        "Consistency_evaluation":grade
    }

# 锂电池故障检测算法 -- 洪钟申
@app.get("/algorithm/lithium/fault_detection")
async def run_fault_detection():
    path = "./dataset/fault_warning/"
    json_output = safety_warning(path)
    return json.loads(json_output)

# 锂电池健康状态估计算法 -- 张颖
@app.get("/algorithm/lithium/soh")
async def run_soh_calculation():
    pass

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)