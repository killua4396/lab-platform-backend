from fastapi import FastAPI
import uvicorn

app = FastAPI()

# ----------------- 请求体定义 -------------------- #


# ------------------------------------------------ #

# 从数据库获取锂电池数据 -- 范晓非
@app.get("/data/lithium")
async def get_lithium_data():
    pass

# 从数据库获取燃料电池数据 -- 由佳茹
@app.get("/data/fuel")
async def get_fuel_data():
    pass

# 锂电池包数学建模 -- 高晋源，陆晓蒙
@app.get("/model/lithum")
async def run_lithium_model():
    pass

# 燃料电池系统数学建模 -- 刘俊，李泽宇
@app.get("/model/fuel")
async def run_fuel_model():
    pass

# 车辆动力学数学建模 -- 朱展锋
@app.get("/model/car")
async def run_car_model():
    pass

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
async def run_kmens_algorithm():
    pass

# 锂电池一致性评价算法 -- 王世超
@app.get("/algorithm/lithium/battery_consistency_assessment")
async def run_battery_consistency_assessment():
    pass

# 锂电池故障检测算法 -- 洪钟申
@app.get("/algorithm/lithium/fault_detection")
async def run_fault_detection():
    pass

# 锂电池健康状态估计算法 -- 张颖
@app.get("/algorithm/lithium/soh")
async def run_soh_calculation():
    pass

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)