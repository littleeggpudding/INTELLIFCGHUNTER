#!/bin/bash

# 定义Python环境和脚本路径
PYTHON_ENV="/home/shiwensong/.virtualenvs/malwareGA/bin/python"
SCRIPT_PATH="/data/c/shiwensong/malwareGA/task/MutateFCG.py"

# 启动10次进程，每个进程的第一个参数从0到9，第二个参数始终为10
for i in {0..29}
do
   $PYTHON_ENV $SCRIPT_PATH $i 30 &
done

# 等待所有后台进程完成
wait
echo "All processes have completed."