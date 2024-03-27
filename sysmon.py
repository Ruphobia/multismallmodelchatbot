import psutil
import GPUtil

def get_cpu_temperatures() -> tuple:
    try:
        cpu_temps = []
        temps = psutil.sensors_temperatures()
        max_temp = float('-inf')
        if 'coretemp' in temps:
            for temp in temps['coretemp']:
                temp_celsius = temp.current  
                cpu_temps.append(temp_celsius)
                max_temp = max(max_temp, temp_celsius)
            return 0, cpu_temps, max_temp
        else:
            print("Couldn't fetch CPU temperatures. Make sure the hardware and OS support this feature.")
            return -1, [], float('-inf')
    except Exception as e:
        print(f"Error occurred: {e}")
        return -1, [], float('-inf')
    
def get_gpu_temperatures() -> tuple:
    try:
        gpu_temps = []
        gpus = GPUtil.getGPUs()
        max_temp = float('-inf')
        for gpu in gpus:
            temp_celsius = gpu.temperature
            gpu_temps.append(temp_celsius)
            max_temp = max(max_temp, temp_celsius)
        return 0, gpu_temps, max_temp
    except Exception as e:
        print(f"Error occurred: {e}")
        return -1, [], float('-inf')

def get_cpu_usage() -> tuple:
    try:
        cpu_usage = psutil.cpu_percent(percpu=True)
        total_cpu_usage = sum(cpu_usage) / len(cpu_usage)
        return 0, cpu_usage, total_cpu_usage
    except Exception as e:
        print(f"Error occurred: {e}")
        return -1, [], 0
    
def get_gpu_usage() -> tuple:
    try:
        gpu_usages = []
        gpus = GPUtil.getGPUs()
        max_usage = float('-inf')
        for gpu in gpus:
            gpu_usages.append(gpu.load * 100) 
            max_usage = max(max_usage, gpu.load * 100)
        return 0, gpu_usages, max_usage
    except Exception as e:
        print(f"Error occurred: {e}")
        return -1, [], float('-inf')

def get_system_ram() -> tuple:
    try:
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_gb_in_use = round(ram.used / (1024 ** 3), 2)
        return 0, ram_percent, ram_gb_in_use
    except Exception as e:
        print(f"Error occurred: {e}")
        return -1, 0, 0
    
def get_gpu_ram_usage() -> tuple:
    try:
        gpu_percent_used = []
        gpu_gb_used = []
        total_gb_used = 0

        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_percent_used.append(gpu.memoryUtil * 100)
            gpu_gb_used.append(round(gpu.memoryUsed / 1024, 2))
            total_gb_used += gpu.memoryUsed / 1024

        total_gb_used = round(total_gb_used, 2)

        return 0, gpu_percent_used, gpu_gb_used, total_gb_used
    except Exception as e:
        print(f"Error occurred: {e}")
        return -1, [], [], 0