#!/bin/bash

# 定义Python环境和脚本路径
PYTHON_ENV="/home/shiwensong/.virtualenvs/malwareGA/bin/python"
SCRIPT_PATH="/data/b/shiwensong/project/malwareGA/task/baseline_malscan_avg_feature_add_edge_ab.py"

# 启动10次进程，每个进程的第一个参数从0到9，第二个参数始终为10
for i in {0..19}
do
   CUDA_VISIBLE_DEVICES=2 $PYTHON_ENV $SCRIPT_PATH $i 20 &
done

# 等待所有后台进程完成
wait
echo "All processes have completed.