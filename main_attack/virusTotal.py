import os
import time
import requests
import csv
from v2_virustotal import detect_one_file


# 配置你的 API 密钥
API_KEY = '73d1f4e374bc6ade978749b84c592a2132707db27d473438f794362cd06d835e'
# VirusTotal API URL
API_URL = 'https://www.virustotal.com/api/v3/files'
feature_dir = '/data/b/shiwensong/dataset/feature_Nov30/'

if __name__ == '__main__':
    # 准备CSV文件
    csv_file = open('virustotal_results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'analysis_url'])
    # read attack samples from txt file
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    name = []
    for year in years:
        txt_file = f'{feature_dir}/mar10_attack_samples_{year}_60.txt'
        with open(txt_file, 'r') as file:
            tmp = file.readlines()
        tmp = tmp[:20]
        name.extend(tmp)

    print("selected_samples", len(name))
    print("selected_samples 0 ", name[0])
    
    all_file_path = []

    # 遍历文件夹中的所有文件
    for gexf_name in name:
        apk_name = gexf_name.strip().split('/')[-1].replace('.gexf', '.apk')
        filename = None
        if 'virusshare2018' in gexf_name:
            filename = '/data/c/shiwensong/dataset/virusshare_2018/' + apk_name
        elif 'virusshare2019' in gexf_name:
            filename = '/data/b/shiwensong/dataset/virusshare2019/' + apk_name
        elif 'virusshare2020' in gexf_name:
            filename = '/data/b/shiwensong/dataset/virusshare2020/' + apk_name
        elif 'virusshare2021' in gexf_name:
            filename = '/data/c/shiwensong/dataset/virusshare_2021/' + apk_name
        elif 'virusshare2022' in gexf_name:
            filename = '/data/c/shiwensong/dataset/virusshare_2022_2/' + apk_name
        elif 'virusshare2023' in gexf_name:
            filename = '/data/c/shiwensong/dataset/virusshare_2023_3/' + apk_name
        file_path = filename
        all_file_path.append(file_path)
        # print(f'Processing: {file_path}')
        # if not os.path.exists(file_path):
        #     print(f'File not found: {file_path}')
        #     continue

        # # 在主程序中使用这个函数
        # result = detect_one_file(apk_name, file_path)
        # print(result)
        # time.sleep(30)
        
    print("all_file_path", len(all_file_path))
    print("all_file_path", all_file_path)
    csv_file.close()
