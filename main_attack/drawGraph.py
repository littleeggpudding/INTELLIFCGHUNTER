import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


def generate_color_map():
    # 创建自定义color map
    colors = [
        "#fdf181",
        "#faf6ae",
        "#e5eca9",
        "#e3f2b1",
        "#d9ebc1",
        "#cce4cc",
        "#bbded0",
        "#b0ded4",
        "#b5dbd0",
        "#afdbd1",
        "#aedad1",
        "#8fb5b6",
        "#81b6bb",
        "#86a6bc"
    ]
    cus_cmap = mcolors.ListedColormap(colors)
    # cus_cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)
    return cus_cmap
    # colors = ["#B9E0A5", "#97D077", "#60A917"]  # 从较深的浅绿色开始，到中绿色，再到深绿色
    # cus_cmap = LinearSegmentedColormap.from_list("custom_green", colors, N=256)
    # return cus_cmap


# degree
# data = [[1, 0.703703704, 0.197530864, 0.012345679, 0],
#         [0.256756757, 1, 0.189189189, 0.013513514, 0],
#         [0.616438356, 0.712328767, 1, 0.02739726, 0.01369863],
#         [0.416666667, 0.25, 0.25, 1, 0.285714286],
#         [0.913043478, 0.260869565, 0.173913043, 0, 1]]

#katz
# data = [
#     [1, 0.177419355, 0.177419355, 0, 0.016129032],
#     [0.043478261, 1, 0.217391304, 0.057971014, 0.014492754],
#     [0.338461538, 0.815384615, 1, 0.030769231, 0],
#     [1, 1, 1, 1, 0],
#     [1, 1, 1, 0, 1]
# ]

#closeness
# data = [
#     [1, 0.447058824, 0.376470588, 0.011764706, 0.011764706],
#     [0.119402985, 1, 0.417910448, 0.014925373, 0.029850746],
#     [0.139534884, 0.348837209, 1, 0.046511628, 0.046511628],
#     [0.328358209, 0.925373134, 0.955223881, 1, 0],
#     [1, 0.923076923, 0.923076923, 0.384615385, 1]
# ]

#harmonic
# data = [
#     [1, 0.534090909, 0.590909091, 0.034090909, 0],
#     [0.12, 1, 0.32, 0.08, 0],
#     [0.013513514, 0.635135135, 1, 0.013513514, 0],
#     [0.838709677, 0.559139785, 0.397849462, 1, 0],
#     [0.979591837, 0.87755102, 0.87755102, 0.469387755, 1]
# ]

#avg
# data = [
#     [1, 0.444444444, 0.6, 0.044444444, 0.022222222],
#     [0.103448276, 1, 0.666666667, 0.034482759, 0],
#     [0.065789474, 0.578947368, 1, 0.039473684, 0.026315789],
#     [0.537037037, 0.333333333, 0.240740741, 1, 0.148148148],
#     [0.740740741, 0.888888889, 0.888888889, 0.148148148, 1]
# ]

#all
# data = [
#     [1, 0.550561798, 0.573033708, 0, 0.011235955],
#     [0.112359551, 1, 0.662921348, 0, 0.02247191],
#     [0.051948052, 0.597402597, 1, 0.025974026, 0.025974026],
#     [0.481927711, 0.108433735, 0.036144578, 1, 0],
#     [0.75862069, 0.908045977, 0.977011494, 0.172413793, 1]
# ]

#MLP
data = [[1.0, 0.012345679, 0.037037037, 0.024691358, 0.197530864, 0.086419753, 0.098765432, 0.185185185], [0.322580645, 1.0, 0.112903226, 0.080645161, 0.177419355, 0.209677419, 0.0, 0.387096774], [0.247058824, 0.011764706, 1.0, 0.247058824, 0.341176471, 0.470588235, 0.035294118, 0.082352941], [0.340909091, 0.0, 0.340909091, 1.0, 0.613636364, 0.784090909, 0.034090909, 0.102272727], [0.311111111, 0.0, 0.177777778, 0.277777778, 1.0, 0.455555556, 0.011111111, 0.222222222], [0.393258427, 0.02247191, 0.213483146, 0.292134831, 0.561797753, 1.0, 0.011235955, 0.134831461], [0.363636364, 0.0, 0.045454545, 0.0, 0.0, 0.0, 1.0, 0.136363636], [0.431034483, 0.0, 0.413793103, 0.362068966, 0.5, 0.551724138, 0.017241379, 1.0]]

#knn1
# data = [[1.0, 0.040540541, 0.094594595, 0.689189189, 0.662162162, 0.675675676, 0.013513514, 0.121621622], [0.115942029, 1.0, 0.043478261, 0.536231884, 0.47826087, 0.507246377, 0.014492754, 0.043478261], [0.402985075, 0.0, 1.0, 0.597014925, 0.432835821, 0.432835821, 0.059701493, 0.059701493], [0.12, 0.0, 0.16, 1.0, 1.0, 1.0, 0.04, 0.12], [0.252873563, 0.0, 0.126436782, 0.988505747, 1.0, 0.91954023, 0.034482759, 0.068965517], [0.202247191, 0.011235955, 0.146067416, 1.0, 0.95505618, 1.0, 0.078651685, 0.112359551], [0.233333333, 0.0, 0.333333333, 0.5, 0.233333333, 0.233333333, 1.0, 0.4], [0.105263158, 0.0, 0.447368421, 0.710526316, 0.184210526, 0.184210526, 0.263157895, 1.0]]

#knn3
# data = [[1.0, 0.068493151, 0.589041096, 0.726027397, 0.712328767, 0.712328767, 0.04109589, 0.054794521], [0.092307692, 1.0, 0.030769231, 0.415384615, 0.4, 0.4, 0.0, 0.015384615], [0.255813953, 0.023255814, 1.0, 0.581395349, 0.534883721, 0.534883721, 0.069767442, 0.139534884], [0.067567568, 0.027027027, 0.189189189, 1.0, 0.932432432, 0.932432432, 0.013513514, 0.108108108], [0.131578947, 0.013157895, 0.184210526, 0.986842105, 1.0, 0.986842105, 0.013157895, 0.092105263], [0.103896104, 0.012987013, 0.12987013, 1.0, 1.0, 1.0, 0.012987013, 0.155844156], [0.225806452, 0.032258065, 0.322580645, 0.290322581, 0.193548387, 0.193548387, 1.0, 0.483870968], [0.111111111, 0.022222222, 0.511111111, 0.222222222, 0.177777778, 0.177777778, 0.266666667, 1.0]]

#ab
# data = [[1.0, 0.104477612, 0.686567164, 0.089552239, 0.223880597, 0.149253731, 0.119402985, 0.074626866], [0.158333333, 1.0, 0.141666667, 0.35, 0.716666667, 0.108333333, 0.091666667, 0.091666667], [0.626865672, 0.134328358, 1.0, 0.597014925, 0.686567164, 0.179104478, 0.044776119, 0.029850746], [0.333333333, 0.247311828, 0.688172043, 1.0, 0.913978495, 0.720430108, 0.11827957, 0.032258065], [0.8, 0.0, 1.0, 0.5, 1.0, 1.0, 0.1, 0.1], [0.581395349, 0.023255814, 0.744186047, 0.11627907, 0.23255814, 1.0, 0.023255814, 0.023255814], [0.0, 0.0, 0.0, 0.0, 0.0, 0.03030303, 1.0, 0.03030303], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

#rf
# data = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.90625, 1.0, 0.416666667, 0.4375, 0.583333333, 0.34375, 0.25, 0.1875], [0.615384615, 0.769230769, 1.0, 0.307692308, 0.384615385, 0.461538462, 0.153846154, 0.153846154], [0.208333333, 0.833333333, 0.916666667, 1.0, 1.0, 1.0, 0.25, 0.166666667], [0.96, 0.04, 0.8, 0.76, 1.0, 0.64, 0.2, 0.04], [0.0, 0.909090909, 0.636363636, 0.090909091, 0.454545455, 1.0, 0.0, 0.0], [0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 0.111111111, 1.0, 0.111111111], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]]

sns.set(style="white")

# 设置全局字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.size"] = 16

# models = ['MLP', 'KNN-1', 'KNN-3', 'AB', 'RF']
models = [r'MalScan$_{D}$', r'MalScan$_{K}$', r'MalScan$_{C}$', r'MalScan$_{H}$', r'MalScan$_{AVG}$', r'MalScan$_{CON}$', 'MaMaDroid', 'APIGraph']
df = pd.DataFrame(data, index=models, columns=models)

# 使用生成的颜色映射
cus_cmap = generate_color_map()

# plt.figure(figsize=(12, 10))  # 调整图像大小
#for feature
fig, ax = plt.subplots(figsize=(12, 8))
#for model
# fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, cmap=cus_cmap, annot=True, fmt=".2f", ax=ax, cbar=False)  # 使用自定义的颜色映射，并添加数值标签

# plt.title('Attack Transfer Rates based on MLP')
font_properties = {
    # 'family' : 'serif',
    # 'color'  : 'darkred',
    # 'weight' : 'normal',
    'size'   : 18,
}

#for feature
ax.set_xlabel('Transfer Feature', fontsize=18, weight='bold')
ax.set_ylabel('Target Feature', fontsize=18, weight='bold')

#for model
# ax.set_xlabel('Transfer Model', fontsize=18)
# ax.set_ylabel('Target Model', fontsize=18)

# Rotating y-axis labels
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.xticks(rotation=25)


# plt.xlabel('Transfer Feature', fontdict=font_properties)
# plt.ylabel('Target Feature', fontdict=font_properties)

plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)


# 如果有颜色条，调整颜色条位置
# cbar = ax.collections[0].colorbar
# cbar.ax.set_position([0.85, 0.15, 0.05, 0.7])  # 调整颜色条的位置和大小

plt.savefig('MLP.png', dpi=300)  # 保存高清图像
plt.show()

