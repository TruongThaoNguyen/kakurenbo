# import sys
from selectors import SelectorKey
from time import sleep
import torch
import torchvision
import numpy as np
import os
from typing import Union
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import math
import random
import re
from typing import List
import pdb
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.folder import make_dataset, find_classes, is_image_file, has_file_allowed_extension

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
):
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    idxs_len = []
    one_inst = []
    available_classes = set()
    count = 0
    
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue

        
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            # Save each length per class
            once = True
            c = [class_index,len(fnames)]
            idxs_len.append(c)

            for fname in sorted(fnames):
                # nomes = sorted(fnames)
                # print(nomes)
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if once:
                        one_inst.append(item)
                        # print(one_inst)
                    once = False
                    if target_class not in available_classes:
                        available_classes.add(target_class)
                # i += 1
            
            count += 1
            # if count > 30:
            #     exit(0)

    # print(len(one_inst))
    # exit(0)

    # for i in range(0,999):
    #     dato = one_inst[i]
    #     number = dato[1]
    #     if number == i:
    #         print(number)
    #     else:
    #         print(dato)
           


    # print(len(one_inst))
    # exit(0)
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, idxs_len, one_inst

class OneInstance_Imnet(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(datasets.DatasetFolder, self).__init__(root, transform=transform,target_transform=target_transform)

        print("From custom pytorch Loader....")
        classes, class_to_idx = self.find_classes(self.root)
        samples,class_len,one_inst = self.make_dataset(self.root, class_to_idx,extensions,is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes

        self.classes_len = class_len
        self.oneInst = one_inst
        # print(self.oneInst)
        # print(len(self.oneInst))


        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        print("Finished loading dataset...")

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::
            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        Args:
            directory(str): Root directory path, corresponding to ``self.root``
        Raises:
            FileNotFoundError: If ``dir`` has no class folders.
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # # if target >= 1000:
        # #    target = 0

        # try:
        instance = self.oneInst[target]
        new_path = instance[0]
        # except ValueError:
        #     print(target)


        sample = self.loader(new_path)
        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
       


