from infrastructure.database import get_db_connection
from models.fuel_cell_data import FuelCellData
from typing import List, Optional
from fastapi import HTTPException
import mysql.connector

def get_fuel_cell_data(
    id: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[FuelCellData]:
    try:
        # 获取数据库连接
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # 构建查询语句
        query = """
            SELECT 
                id, time, status, air_flow, air_temperature, water_pressure, 
                stack_current, stack_voltage, real_power, stack_outlet_water_temperature, 
                stack_inlet_water_temperature, hydrogen_pressure, air_pressure, 
                max_single_voltage, min_single_voltage, avg_single_voltage, start_stop_control,
                control_mode_selection, target_current, dcdc_hotspot_temperature,
                dcdc_status, fault_info, output_voltage, output_current, input_voltage,
                input_current, hydrogen_concentration, insulation_resistance,single_voltage1,
                single_voltage2, single_voltage3, single_voltage4, single_voltage5, single_voltage6,
                single_voltage7, single_voltage8, single_voltage9, single_voltage10, single_voltage11,
                single_voltage12, single_voltage13, single_voltage14, single_voltage15, single_voltage16,
                single_voltage17, single_voltage18, single_voltage19, single_voltage20, single_voltage21,
                single_voltage22, single_voltage23, single_voltage_low, module_hydrogen_concentration_abnormal,
                inlet_outlet_temperature_difference_abnormal, outlet_water_temperature_high_abnormal, water_pump_fault,
                hydrogen_pump_abnormal, hydrogen_pressure_high_abnormal, hydrogen_export_pressure_sensor_abnormal,
                dc_communication_abnormal, dcdc_fault, heater_fault, hydrogen_tank_temperature_abnormal,
                hydrogen_tank_high_pressure_abnormal, hydrogen_tank_medium_pressure_abnormal, hydrogen_tank_low_pressure_abnormal,
                fcu_communication_abnormal, temperature_sensor_abnormal, hydrogen_pressure_sensor_self_check_abnormal,
                hydrogen_tank_soc_low, hydrogen_export_pressure_low_abnormal, air_pressure_low_abnormal, air_pressure_high_abnormal,
                air_temperature_high_abnormal, cooling_water_pressure_high_abnormal, single_voltage_high_abnormal,
                insulation_low_abnormal, hydrogen_air_pressure_difference_large_abnormal_negative, hydrogen_air_pressure_difference_large_abnormal_positive,
                start_hydrogen_pressure_low_below_20kpa, hydrogen_leak_check_failure, total_voltage, total_current,
                remaining_capacity, nominal_capacity, cycle_count, remaining_capacity_percentage, production_date,
                `range`, software_version, speed, vehicle_speed
            FROM fuel_cell_data
            WHERE 1=1
        """
        params = []

        # 根据 ID 查询
        if id is not None:
            query += " AND id = %s"
            params.append(id)

        # 根据时间范围查询
        if start_time and end_time:
            query += " AND time BETWEEN %s AND %s"
            params.extend([start_time, end_time])
        elif start_time:
            query += " AND time >= %s"
            params.append(start_time)
        elif end_time:
            query += " AND time <= %s"
            params.append(end_time)

        # 执行查询
        cursor.execute(query, params)
        results = cursor.fetchall()

        # 关闭连接
        cursor.close()
        connection.close()

        # 将查询结果转换为 Pydantic 模型
        return [FuelCellData(**result) for result in results]

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"数据库查询失败: {err}")