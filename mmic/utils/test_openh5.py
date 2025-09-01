import h5py
import numpy as np 
file_path = 'E:\\machine_learning\\projects\\fedadvan\\utils\\mnist_scaffoldmnist_scaffold_20240312_12_31_43.h5'
file = h5py.File(file_path,'r')

# 打开 HDF5 文件  
with h5py.File(file_path, 'r') as f:  
    # 遍历文件中的所有项（数据集和组）  
    for name in f:  
        item = f[name]  
          
        # 判断当前项是数据集还是组  
        if isinstance(item, h5py.Dataset):  
            # 读取数据集的数据  
            data = item[:]  
            # 打印数据集名称和数据形状  
            print(f"Dataset '{name}': shape {data.shape}, dtype {data.dtype}")  
            # 如果需要，你可以在这里对数据进行进一步处理  
            # ...  
            print(data)
            
        elif isinstance(item, h5py.Group):  
            # 如果是组，你可以递归地遍历它  
            print(f"Group '{name}'")  
            # 递归函数来读取组中的内容  
            def read_group(group):  
                for name in group:  
                    sub_item = group[name]  
                    if isinstance(sub_item, h5py.Dataset):  
                        data = sub_item[:]  
                        print(f"  Dataset '{name}': shape {data.shape}, dtype {data.dtype}")  
                    elif isinstance(sub_item, h5py.Group):  
                        print(f"  Group '{name}'")  
                        read_group(sub_item)  
            # 调用递归函数读取当前组的内容  
            read_group(item)  
        # 这里可以添加对属性的读取，如果需要的话  
        # attrs = item.attrs  
        # for attr_name, attr_value in attrs.items():  
        #    print(f"Attribute '{attr_name}': {attr_value}")  
  
# 文件自动关闭，因为使用了 with 语句