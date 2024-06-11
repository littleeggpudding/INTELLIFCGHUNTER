import subprocess
import os
import glob

# 先获取当前Python文件所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到这个目录
os.chdir(current_dir)

# 虚拟环境激活命令
activate_env_cmd = "source /home/shiwensong/.virtualenvs/HRAT-copy-main/bin/activate"

feature_dir = '/data/b/shiwensong/dataset/feature_Nov30/'
output_path = '/data/b/shiwensong/dataset/120_samples_constraints/'

years = ['2018', '2019', '2020', '2021', '2022', '2023']
names = []
for year in years:
    txt_file = f'{feature_dir}/mar10_attack_samples_{year}_60.txt'
    with open(txt_file, 'r') as file:
        tmp = file.readlines()
    tmp = [line.strip() for line in tmp[:20]]  # 更优雅的去除换行符
    names.extend(tmp)

print("selected_samples", len(names))
print("selected_samples 0 ", names[0])

#已经提取完成的
sha256s = glob.glob('/data/b/shiwensong/dataset/120_samples_constraints/cons/*')
print("extracted samples", len(sha256s))
sha256s = [sha256.split('/')[-1].replace('.txt','') for sha256 in sha256s]
print("extracted samples", len(sha256s))
print("extracted samples 0 ", sha256s[0])



for name in names:
    file_path = name.replace('_gexf', '').replace('.gexf', '') + '.apk'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        continue

    name_sha256 = file_path.split('/')[-1].replace('.apk','')
    if not name_sha256 in sha256s:
        print(f"not Already extracted: {name}")
        continue

    # apk_path = file_path
    # # 构建完整的命令
    # command = f"python3 /data/b/shiwensong/project/HRAT/PreprocessAPK.py --apk_path {apk_path} --output_path {output_path}"
    #
    # # 将激活虚拟环境和执行命令组合在一起
    # full_command = f'{activate_env_cmd} && {command}'
    #
    # # 在shell中执行组合的命令
    # try:
    #     subprocess.run(full_command, check=True, shell=True, executable="/bin/bash")
    #     print(f"Command executed successfully for {name}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error occurred while executing command for {name}: {e}")
