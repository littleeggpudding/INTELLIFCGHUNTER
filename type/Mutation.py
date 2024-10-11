import os
import json


class Mutation:

    def __init__(self, apk_name):
        self.apk_name = apk_name
        self.mutation = {
            'feature_type': '',
            'add_edges': [],
            'remove_edges': [],
            'add_nodes': [],
            'remove_nodes': []
        }


    def save_log(self, note=""):

        try:
            json_string = json.dumps(self.mutation)
            file_name = self.apk_name + note + '.json'
            with open(file_name, 'a') as file:
                file.write(json_string + '\n')
        except Exception as e:
            print("Failed to save log:", str(e))

    def read_log(self, note=""):
        file_name = self.apk_name + note + '.json'
        if not os.path.exists(file_name):
            return None

        loaded_data = []

        with open(file_name, 'r') as file:
            for line in file:
                data = json.loads(line.strip())  
                loaded_data.append(data)
        
        return loaded_data


    def clear_log(self):
        file_name = self.apk_name + '.json'

        with open(file_name, 'w') as file:
            file.write('')

    def clear_mutation(self):
        self.mutation['feature_type'] = ''
        self.mutation['add_edges'] = []
        self.mutation['remove_edges'] = []
        self.mutation['add_nodes'] = []
        self.mutation['remove_nodes'] = []
