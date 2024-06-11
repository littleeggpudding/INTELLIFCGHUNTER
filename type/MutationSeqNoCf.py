

class MutationSeqNoCf:
    def __init__(self, add_many_nodes, relay_gene, directlink_gene,
                 relay_gene_detail, directlink_gene_detail):
        self.add_many_nodes = add_many_nodes
        self.relay_gene = relay_gene
        self.directlink_gene = directlink_gene
        self.relay_gene_detail = relay_gene_detail
        self.directlink_gene_detail = directlink_gene_detail
        self.final_group_list = [[]]