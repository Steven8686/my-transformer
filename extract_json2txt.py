import json
import os


class PrepareData():
    @staticmethod
    def prepare():
        root_dir = 'C:\\Users\\18380\\Desktop\\University\\transformer_data\\new\\AA\\'
        ds = []
        for dir_path, dir_names, file_names in os.walk(root_dir):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                if "." in file_path:
                    continue
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            text = json.loads(line)["text"]
                            # print(text)
                            ds.append(text)
                        except json.JSONDecodeError:
                            print("格式不正确")
                print(file_name, 'done')
        print(len(ds))
        with open('C:\\Users\\18380\\Desktop\\University\\transformer_data\\sentence.txt', 'w', encoding='utf-8') as file:
            for i in ds:
                file.write(i + '\n')
        return ds

data_set = PrepareData.prepare()
for item in data_set:
    print(item)


