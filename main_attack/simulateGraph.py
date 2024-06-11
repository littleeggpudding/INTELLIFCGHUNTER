import networkx as nx
import pickle

def calculateCentralityChanges(orignal, current):
    changes = []
    for node in orignal.keys():
        if node not in current.keys():
            continue
        change = {
            "node": node,
            "change": current[node]-orignal[node]
        }
        changes.append(change)

    return changes
    #正数说明增加，负数说明减少，0说明不变
def analyze4Features(orignal, current):

    ori_degree = orignal[0]#original: dict, format: node: value
    ori_katz = orignal[1]
    ori_closeness = orignal[2]
    ori_harmonic = orignal[3]

    cur_degree = current[0]
    cur_katz = current[1]
    cur_closeness = current[2]
    cur_harmonic = current[3]

    degree_change = calculateCentralityChanges(ori_degree, cur_degree)
    katz_change = calculateCentralityChanges(ori_katz, cur_katz)
    closeness_change = calculateCentralityChanges(ori_closeness, cur_closeness)
    harmonic_change = calculateCentralityChanges(ori_harmonic, cur_harmonic)

    return degree_change, katz_change, closeness_change, harmonic_change

def calculateCentrality(graph):
    #计算中心度
    degree = nx.degree_centrality(graph)
    katz = nx.katz_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    harmonic = nx.harmonic_centrality(graph)

    return [degree, katz, closeness, harmonic]




if __name__ == '__main__':
    #创建一个有向图
    G = nx.DiGraph()
    # #添加一个节点
    # G.add_node(1)
    # #添加一列节点
    # G.add_nodes_from([2,3])

    #添加一列节点
    G.add_nodes_from(range(1,11))
    #添加边
    # G.add_edge(1,2)
    #添加一列边
    G.add_edges_from([(1,3),(1,4),(1,5),(1,7),(2,3),(2,4),(2,6),(8,2),(8,9),(10,9)])

    original = calculateCentrality(G)

    print("add node")


    #开始操作图
    G1 = G.copy()
    #添加节点
    G1.add_node(11)
    G1.add_edge(9,11)
    #计算中心度
    current = calculateCentrality(G1)

    #分析中心度变化
    degree_change, katz_change, closeness_change, harmonic_change = analyze4Features(original, current)
    print('degree')
    print(degree_change)
    print('katz')
    print(katz_change)
    print('closeness')
    print(closeness_change)
    print('harmonic')
    print(harmonic_change)

    print('----------------------')

    print("add edge")
    G2 = G1.copy()
    G2.add_edge(8, 11)
    G2_centrality = calculateCentrality(G2)

    # 分析中心度变化
    degree_change, katz_change, closeness_change, harmonic_change = analyze4Features(current, G2_centrality)
    print('degree')
    print(degree_change)
    print('katz')
    print(katz_change)
    print('closeness')
    print(closeness_change)
    print('harmonic')
    print(harmonic_change)

    print('----------------------')

    print("insert node for extending the node 11")

    # 开始操作图
    G3 = G2.copy()
    #删除一条边
    G3.add_node(12)
    G3.remove_edge(8,11)
    G3.add_edge(8,12)
    G3.add_edge(12,11)
    G3_centrality = calculateCentrality(G3)

    #分析中心度变化
    degree_change, katz_change, closeness_change, harmonic_change = analyze4Features(G2_centrality, G3_centrality)
    print('degree')
    print(degree_change)
    print('katz')
    print(katz_change)
    print('closeness')
    print(closeness_change)
    print('harmonic')
    print(harmonic_change)

    print('----------------------')

    print("delete node")
    G4 = G3.copy()
    G4.remove_node(11)
    G4_centrality = calculateCentrality(G4)

    # 分析中心度变化
    degree_change, katz_change, closeness_change, harmonic_change = analyze4Features(G3_centrality, G4_centrality)
    print('degree')
    print(degree_change)
    print('katz')
    print(katz_change)
    print('closeness')
    print(closeness_change)
    print('harmonic')
    print(harmonic_change)

    print('----------------------')
    G4.add_edges_from([(1, 2), (2, 3), (3, 10),(3,4),(4,10)])
    if nx.has_path(G4, 1, 10):
        res = nx.shortest_path(G4, 1, 10)
        res2 = nx.all_simple_paths(G4, 1, 10)
        print(res)
        for i in res2:
            print(i)

    degree = nx.degree_centrality(G4)
    katz = nx.katz_centrality(G4)
    closeness = nx.closeness_centrality(G4)
    harmonic = nx.harmonic_centrality(G4)
    print(degree)
    print(katz)
    print(closeness)
    print(harmonic)

    print('----------------------')

    #删除边
    G4.remove_edge(3,10)
    # G4.remove_edge(1,4)
    # G4.add_edge(8,10)
    # G4.remove_edge(1,3)
    
    res2 = nx.all_simple_paths(G4,1, 10)
    for i in res2:
        print(i)
    degree = nx.degree_centrality(G4)
    katz = nx.katz_centrality(G4)
    closeness = nx.closeness_centrality(G4)
    harmonic = nx.harmonic_centrality(G4)
    print(degree)
    print(katz)
    print(closeness)
    print(harmonic)
    






    # #删除边
    # G.remove_edge(1,2)
    # #删除一列边
    # G.remove_edges_from([(1,3),(1,4),(1,5),(1,6),(1,7)])
    # #删除节点
    # G.remove_node(1)
    # #删除一列节点
    # G.remove_nodes_from([2,3])

    #计算中心度

    test_data = "/data/a/shiwensong/dataset/test_data_2013-2018_everyyear500.pkl"
    df = open(test_data, "rb")
    data_test_dict = pickle.load(df)
    df.close()
    print('test--length', len(data_test_dict))