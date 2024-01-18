import os
import pandas as pd
num = 8

import pandas as pd

# 假设上述输出保存在一个名为 "output.txt" 的文件中
file_path = r'results/intermediate/dk2kaenv/inter_20240118_0351.log'

# 将DataFrame写入Excel文件
excel_path = file_path[:-3] + 'csv'
 
# 创建空的DataFrame来存储数据
data = {
    'Layer': [],
    # 'Evaluation Type': [],
    'UnfAccuracy@1': [],
    'UnfAccuracy@5': [],
    
    'Unf Mean Distance FaceNet': [],
    'Unf Mean Distance Inception-v3': [],
    'UnfPrecision': [],
    'UnfRecall': [],
    'UnfDensity': [],
    'UnfCoverage': [],
    'FAccuracy@1': [],
    'FAccuracy@5': [],
    'Mean Distance FaceNet': [],
    'Mean Distance Inception-v3': [],
    'Precision': [],
    'Recall': [],
    'Density': [],
    'Coverage': [],
    
    
}

# 读取文件并处理每一行
with open(file_path, 'r') as file:
    flag = True
    t = 'unf'
    for line in file:
        if flag and not line.startswith('Unfilter'):
            continue
        if 'Unfiltered Evaluation' in line or 'Filtered Evaluation' in line:
            if 'best' in line:
                continue
            flag = False
            parts = line.split()
            # print(parts)
            layer = int(parts[9].rstrip(':'))
            # eval_type = 'Unfiltered' if 'Unfiltered' in line else 'Filtered'
            accuracy_1 = float(parts[10].rstrip(',')[-8:])
            accuracy_5 = float(parts[12].rstrip(',')[-8:])

            # 更新DataFrame
            
            # data['Evaluation Type'].append(eval_type)
            if 'Unfiltered' in line:
                data['Layer'].append(layer)
                data['UnfAccuracy@1'].append(accuracy_1)
                data['UnfAccuracy@5'].append(accuracy_5)
            else:
                data['FAccuracy@1'].append(accuracy_1)
                data['FAccuracy@5'].append(accuracy_5)
            # data['Correct Confidence'].append(correct_conf)
            # data['Total Confidence'].append(total_conf)

        elif 'Precision' in line:
            parts = line.split()
            
            precision = float(parts[1].rstrip(','))
            recall = float(parts[3].rstrip(','))
            density = float(parts[5].rstrip(','))
            coverage = float(parts[7])

            if t == 'unf':
                data['UnfPrecision'].append(precision)
                data['UnfRecall'].append(recall)
                data['UnfDensity'].append(density)
                data['UnfCoverage'].append(coverage)
            else:
                # 更新DataFrame
                data['Precision'].append(precision)
                data['Recall'].append(recall)
                data['Density'].append(density)
                data['Coverage'].append(coverage)

        elif 'ean Distance' in line:
            parts = line.split()
            # layer = int(parts[5])
            mean_distance = float(parts[-1])
            # print(parts)
            # 更新DataFrame
            if 'Unfiltered' in line:
                if 'Inception-v3' in line:
                    data['Unf Mean Distance Inception-v3'].append(mean_distance)
                elif 'FaceNet' in line:
                    data['Unf Mean Distance FaceNet'].append(mean_distance)
            else:
                if 'Inception-v3' in line:
                    data['Mean Distance Inception-v3'].append(mean_distance)
                elif 'FaceNet' in line:
                    data['Mean Distance FaceNet'].append(mean_distance)
        elif 'Filtered metrics of layer' in line:
            t = 'f'
        elif 'Unfiltered metrics of layer' in line:
            t = 'unf'

for k, v in data.items():
    print(f'{k}: {len(v)}')
# 将数据转换为DataFrame
df = pd.DataFrame(data)


df.to_csv(excel_path, index=False)
