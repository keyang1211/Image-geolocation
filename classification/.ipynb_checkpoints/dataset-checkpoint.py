import os
import sys
import re
from math import ceil
from typing import Dict, List, Tuple, Union
from io import BytesIO
import random
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from PIL import Image
import torchvision
import torch
import msgpack


class MsgPackIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str, #msgpack
        target_mapping: Dict[str, list], #afterjson文件，key是imgid，value是lat lon的列表
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        cache_size=6 * 4096,
    ):

        super(MsgPackIterableDatasetMultiTargetWithDynLabels, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

    

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            # 在这里修改shards_index列表，就可以控制数据数量的百分比
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        
        return img, x["target"] #返回图片张量和经纬度的列表。。。。。。。也就是可迭代数据集迭代输出的每一项都是这样！！！！！
        
    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        
                        x["target"] = self.target_mapping[_id] # 是个列表，包含经纬度
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length
    

class TestsetIterableDataset(torch.utils.data.IterableDataset):
    """
    Data source: jpg files
    Target values are generated on the fly given a mapping (id->[lat, lon])
    """

    def __init__(
        self,
        path: str, #files
        target_mapping: Dict[str, list], #afterjson文件，key是imgid，value是lat lon的列表
        transformation=None,
        shuffle=True,
        cache_size=128,
    ):

        super(TestsetIterableDataset, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        

    

    
        self.length = len(self.target_mapping)

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(os.path.join(self.path, x["path"]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        
        return img, x["target"] #返回图片张量和经纬度的列表。。。。。。。也就是可迭代数据集迭代输出的每一项都是这样！！！！！
        
    def __iter__(self):
        files = os.listdir(self.path)

        # 筛选照片文件
        photo_extensions = ['.jpg', '.jpeg', '.png']  # 可能的照片文件扩展名

        photo_files = [file for file in files if os.path.splitext(file)[1].lower() in photo_extensions]

        photo_files = [{"path":x} for x in photo_files]
        cache = []
        
        for x in photo_files:
            
            try:
                # set target value dynamically 
                x["target"] = self.target_mapping[x["path"]] # 是个列表，包含经纬度
            except KeyError:
                # reject sample
                # print(f'reject {_id} {type(_id)}')
                continue

            if len(cache) < self.cache_size:
                cache.append(x)

            if len(cache) == self.cache_size:

                if self.shuffle:
                    random.shuffle(cache)
                while cache:
                    yield self._process_sample(cache.pop())
                    
                    
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())  
                    

    def __len__(self):
        return self.length
    
    
    
    
    


class MsgPackIterableDatasetfirst25(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str, #msgpack
        target_mapping: Dict[str, list], #afterjson文件，key是imgid，value是lat lon的列表
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        cache_size=6 * 4096,
    ):

        super(MsgPackIterableDatasetfirst25, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

    

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            # 在这里修改shards_index列表，就可以控制数据数量的百分比
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        
        return img, x["target"] #返回图片张量和经纬度的列表。。。。。。。也就是可迭代数据集迭代输出的每一项都是这样！！！！！
        
    def __iter__(self):

        shard_indices = list(range(len(self.shards)//4)) # 25%

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        
                        x["target"] = self.target_mapping[_id] # 是个列表，包含经纬度
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length
    
    
    

    
class MsgPackIterableDatasetfirst50(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str, #msgpack
        target_mapping: Dict[str, list], #afterjson文件，key是imgid，value是lat lon的列表
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        cache_size=6 * 4096,
    ):

        super(MsgPackIterableDatasetfirst50, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

    

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            # 在这里修改shards_index列表，就可以控制数据数量的百分比
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        
        return img, x["target"] #返回图片张量和经纬度的列表。。。。。。。也就是可迭代数据集迭代输出的每一项都是这样！！！！！
        
    def __iter__(self):

        shard_indices = list(range(len(self.shards)//2)) # 50%

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        
                        x["target"] = self.target_mapping[_id] # 是个列表，包含经纬度
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length
    
    
    
    
class MsgPackIterableDatasetfirst75(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str, #msgpack
        target_mapping: Dict[str, list], #afterjson文件，key是imgid，value是lat lon的列表
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        cache_size=6 * 4096,
    ):

        super(MsgPackIterableDatasetfirst75, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

    

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            # 在这里修改shards_index列表，就可以控制数据数量的百分比
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        
        return img, x["target"] #返回图片张量和经纬度的列表。。。。。。。也就是可迭代数据集迭代输出的每一项都是这样！！！！！
        
    def __iter__(self):

        shard_indices = list(range((len(self.shards)*3)//4)) # 75%

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        
                        x["target"] = self.target_mapping[_id] # 是个列表，包含经纬度
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length
    
    
    
    
class MsgPackIterableDatasetlast50(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str, #msgpack
        target_mapping: Dict[str, list], #afterjson文件，key是imgid，value是lat lon的列表
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        cache_size=6 * 4096,
    ):

        super(MsgPackIterableDatasetlast50, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

    

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            # 在这里修改shards_index列表，就可以控制数据数量的百分比
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        
        return img, x["target"] #返回图片张量和经纬度的列表。。。。。。。也就是可迭代数据集迭代输出的每一项都是这样！！！！！
        
    def __iter__(self):

        total_indices = list(range(len(self.shards)))
        first_half_indices = list(range(len(self.shards)//2))
        shard_indices = [index for index in total_indices if index not in first_half_indices]
        # last50

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        
                        x["target"] = self.target_mapping[_id] # 是个列表，包含经纬度
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length