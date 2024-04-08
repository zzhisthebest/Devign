import json

# 读取 JSON 文件
with open('devign_dataset/devign-train-v0.json', 'r') as file:
    data = json.load(file)

# 检查是否存在 'targets' 属性，并将其改名为 'target'
for item in data:
    if 'targets' in item:
        item['target'] = item.pop('targets')

# 保存修改后的数据到新文件
with open('devign_dataset/devign-train-v0.json', 'w') as file:
    json.dump(data, file, indent=4)
print("train修改完毕")



# 读取 JSON 文件
with open('devign_dataset/devign-test-v0.json', 'r') as file:
    data = json.load(file)


for item in data:
    if 'targets' in item:
        item['target'] = item.pop('targets')

# 保存修改后的数据到新文件
with open('devign_dataset/devign-test-v0.json', 'w') as file:
    json.dump(data, file, indent=4)
#print(data)
print("test修改完毕")




# 读取 JSON 文件
with open('devign_dataset/devign-valid-v0.json', 'r') as file:
    data = json.load(file)

# 检查是否存在 'targets' 属性，并将其改名为 'target'
for item in data:
    if 'targets' in item:
        item['target'] = item.pop('targets')

# 保存修改后的数据到新文件
with open('devign_dataset/devign-valid-v0.json', 'w') as file:
    json.dump(data, file, indent=4)
print("valid修改完毕")