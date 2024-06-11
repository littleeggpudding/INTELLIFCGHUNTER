import sys
import os
sys.path.append(os.path.abspath('../task'))

from ExtractFeature import obtain_sensitive_apis, extract_degree_centrality_from_CG, \
    extract_katz_centrality_from_CG, extract_harmonic_centrality_from_CG, extract_closeness_centrality_from_CG
import pickle
import json #序列化
import time



class Mutation:

    def __init__(self, apk_name):
        self.apk_name = apk_name
        self.mutation = {  # 用于存储每次变异的增量信息，log里面有该fcg的全部信息
            'feature_type': '',
            'add_edges': [],
            'remove_edges': [],
            'add_nodes': [],
            'remove_nodes': []
        }
        # print("init mutation!!")


    #每一次mutation的存储
    def save_log(self, note=""):
        # print(self.mutation)
        # 序列化字典为二进制数据
        try:
            json_string = json.dumps(self.mutation)
            file_name = self.apk_name + note + '.json'
            with open(file_name, 'a') as file:
                file.write(json_string + '\n')
            # print("save mutation log successfully!")
        except Exception as e:
            print("Failed to save log:", str(e))

    def read_log(self, note=""):
        file_name = self.apk_name + note + '.json'
        if not os.path.exists(file_name):
            return None

        loaded_data = []

        # 从文件中逐行读取JSON字符串并反序列化为字典
        with open(file_name, 'r') as file:
            for line in file:
                data = json.loads(line.strip())  # 移除行尾的换行符并反序列化
                loaded_data.append(data)
        # print(loaded_data)
        return loaded_data


    def clear_log(self):
        file_name = self.apk_name + '.json'

        # 使用写入模式打开文件并覆盖内容
        with open(file_name, 'w') as file:
            file.write('')

    def clear_mutation(self):
        self.mutation['feature_type'] = ''
        self.mutation['add_edges'] = []
        self.mutation['remove_edges'] = []
        self.mutation['add_nodes'] = []
        self.mutation['remove_nodes'] = []


if __name__ == '__main__':
    fcg = FCG()
    fcg.load('/data/c/shiwensong/Malscan/MalScan-code/benign_2018_gexf/FFE2031B63A7ED452226D44F87F47B43288C3573DC0EAB4B0C7C7F2597DDC16B.gexf', 0)
    # print(fcg.edges)
    m
    print("mutation ing edges",len(mutation.edges), len(fcg.edges))
    print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
    mutation.clear_log()
    mutation.rewiring()
    mutation.add_edge()
    mutation.remove_node()
    print("mutation ing edges", len(mutation.edges), len(fcg.edges))
    print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
    print(mutation.mutation)

    #('Lcom/unity3d/plugin/downloader/c/j;->a(Ljava/lang/String;)Ljava/security/PublicKey;', 'Ljava/security/KeyFactory;->getInstance(Ljava/lang/String;)Ljava/security/KeyFactory;')

    fcg.process_mutation(mutation.mutation)
    print("mutation ing edges", len(mutation.edges), len(fcg.edges))
    print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
    log = mutation.read_log()
    print(len(log))
    for l in log:
        print("------------------")
        # 解析字典中的键值对，提供默认值为空列表或空集合
        print(l)
        type = l.get('feature_type', '')
        add_edges = l.get('add_edges', [])
        remove_edges = l.get('remove_edges', [])
        add_nodes = l.get('add_nodes', set())
        remove_nodes = l.get('remove_nodes', set())
        print(add_edges)
        print(remove_edges)
        print(add_nodes)
        print(remove_nodes)
        print("------------------")

    fcg.save('/data/c/shiwensong/Malscan/MalScan-code/test_coding')
    # fcg.init_centralities()
