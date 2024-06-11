#从APK提取FCG并存到文件夹
import networkx as nx
from collections import defaultdict
import time
import os
from androguard.misc import AnalyzeAPK
# import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool as ThreadPool
from functools import partial
import zipfile

def get_call_graph(dx):
    t0 = time.time()
    # CG = nx.MultiDiGraph()
    CG = nx.DiGraph()
    nodes = dx.find_methods('.*', '.*', '.*', '.*')
    for m in nodes:
        API = m.get_method()
        class_name = API.get_class_name()
        method_name = API.get_name()
        descriptor = API.get_descriptor()
        api_call = class_name + '->' + method_name + descriptor
        # api_call = class_name + '->' + method_name

        if len(m.get_xref_to()) == 0:
            continue
        CG.add_node(api_call)

        for other_class, callee, offset in m.get_xref_to():
            _callee = callee.get_class_name() + '->' + callee.get_name() + callee.get_descriptor()
            # _callee = callee.get_class_name() + '->' + callee.get_name()
            CG.add_node(_callee)
            if not CG.has_edge(API, callee):
                CG.add_edge(api_call, _callee)

    return CG


def apk_to_callgraph(app_path, exist_files, out_path):
    apk_name = os.path.splitext(os.path.basename(app_path))[0]
    # apk_name = app_path.split('/')[-1].split('.apk')[0]
    print("apk_name", apk_name)
    if apk_name in exist_files: # 防止重复提取
        return None
    # elif not zipfile.is_zipfile(app_path): # 防止提取非APK文件
    #     print("not a zip file", app_path)
    #     return None
    else:
        try:
            print("dododo", app_path)
            a, d, dx = AnalyzeAPK(app_path)
            # 输出 APK 的 minSdkVersion 和 targetSdkVersion
            # min_sdk_version = a.get_min_sdk_version()
            # target_sdk_version = a.get_target_sdk_version()
            # print(f"APK Name: {apk_name}")
            # print(f"Min SDK Version: {min_sdk_version}")
            # print(f"Target SDK Version: {target_sdk_version}")

            call_graph = get_call_graph(dx=dx)
            print("check1", apk_name)
            file_cg = os.path.join(out_path, apk_name + '.gexf')
            print("check1", file_cg)
            nx.write_gexf(call_graph, file_cg)
        except Exception as e:

            print(f"Error: {e}")

            return None

#origin_path: the path of an APK file or a dir contains some APK files
def extract(origin_path, out_path):
    tic = time.time()
    # args = parse_args()

    #判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    exist_files = os.listdir(out_path)
    print("out_path", out_path)
    # print("exist_files", exist_files)  # 输出目录下的文件，防止重复提取
    exist_files = [f.split('.gexf')[0] for f in exist_files]

    #判断输入文件是一个APK文件还是一个文件夹
    if out_path[-1] == '/':
        out_path = out_path[:-1]
    else:
        out_path = out_path
    # print(out_path)
    if os.path.isdir(origin_path):
        if origin_path[-1] == '/':
            path = origin_path + '*.apk'
        else:
            path = origin_path + '/*.apk'
        print(path)
        apks = glob.glob(path)
        pool = ThreadPool(50)
        pool.map(partial(apk_to_callgraph, exist_files=exist_files, out_path=out_path), apks)
    else:
        print("111")
        apk_to_callgraph(origin_path, exist_files, out_path)

    print(time.time() - tic)


if __name__ == '__main__':
    # main()
    extract("/data/b/shiwensong/dataset/virusshare2021/","/data/b/shiwensong/dataset/virusshare2021_gexf/")
    # apis = obtain_sensitive_apis()
    # print(apis[0])

