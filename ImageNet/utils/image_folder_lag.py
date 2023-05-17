#from .vision import VisionDataset
from torchvision.datasets import VisionDataset
from PIL import Image
#import threading

#Image.LOAD_TRUNCATED_IMAGES = True For ImageNet21K dataset...
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            class_file: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            enable_cache = False,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        self.class_file = class_file
        classes, class_to_idx = self._find_classes_from_file(self.class_file)

        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.all_samples = samples
        self.all_targets = [s[1] for s in samples]
        
        #For exchange
        self.samples = self.all_samples.copy()  
        self.targets = self.all_targets.copy()
        
        #Caching the loaded data
        #self.lock = threading.Lock()
        self.cache = {}
        self.enable_cache = enable_cache        

    def _find_classes_from_file(self, file_path: str) -> Tuple[List[str], Dict[str, int]]:
        with open(file_path) as f:
            lines = f.readlines()
            lines = [s.strip() for s in lines]
        
        classes_list = []
        for i in range(0, len(lines)):
            tmp = lines[i]
            split_tmp = tmp.split("\t")
            class_name = split_tmp[0].strip()
            classes_list.append(class_name)
        
        classes_list.sort()        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes_list)}
        return classes_list, class_to_idx
        
    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        #self.lock.acquire()
        path, target = self.samples[index]
        #print(self.rank, "get item", index, "from", path)
        if path is not None:
            if self.enable_cache:
                file_name = os.path.basename(path)
                if filename in self.cache:
                    sample, target = self.cache[filename]
                    #self.lock.release()
                    
                    if self.transform is not None:
                        sample = self.transform(sample)
                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    
                    return index, sample, target
            
            #self.lock.release()
            try:
                sample = self.loader(path)
            except:
                print("ERROR", self.rank, path, index)
                raise FileNotFoundError
        
            if self.enable_cache:
                #self.lock.acquire()
                # Cache raw item
                self.cache[filename] = (sample, target)
                #self.lock.release()
                
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        else:
            sample = None
            target = None
            #self.lock.release()
        #self.lock.release()
        return index, sample, target  #NguyenTT Return index just for log the importance
    
    
    def get_exchanged_item(self, index:int):
        #self.lock.acquire()
        path, target = self.all_samples[index]
        #print(self.rank, "get item", index, "from", path)
        if path is not None:
            if self.enable_cache:
                file_name = os.path.basename(path)
                if filename in self.cache:
                    sample, target = self.cache[filename]
                    #self.lock.release()
                    
                    if self.transform is not None:
                        sample = self.transform(sample)
                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    
                    return sample, target
            
            #self.lock.release()
            try:
                sample = self.loader(path)
            except:
                print("ERROR", self.rank, path, index)
                raise FileNotFoundError
        
            if self.enable_cache:
                #self.lock.acquire()
                # Cache raw item
                self.cache[filename] = (sample, target)
                #self.lock.release()
                
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        else:
            sample = None
            target = None
            #self.lock.release()
        #self.lock.release()
        return sample, target
        
    def __len__(self) -> int:
        return len(self.samples)

    def get_raw_item(self, index:int) -> Tuple[Any, Any, Any]:    
        path, target = self.samples[index]
        file_name = os.path.basename(path)  
        class_name = os.path.dirname(path)
        class_name = os.path.basename(class_name)
        
        if path is not None:
            if self.enable_cache:
                if filename in self.cache:
                    sample, target = self.cache[filename]
                    #self.lock.release()
                    return sample, file_name, class_name
                    
            sample = self.loader(path)
        else:
            sample = None
            print("WARNING: [{}] tried to get non existing sample idx: {}".format(self.rank,index),path)
          
        return sample, file_name, class_name
    
    def add_a_item(self, index:int, file_name, class_name, raw_data):
        ## Replace an item with a new one in the all_samples array
        
        # if self.samples[index] is (None , None):
            # pass
        # else:
            # print("WARNING! Replace an existing samples", index, "at rank", self.rank, path)
            
        target = self.class_to_idx[class_name]
            
        #Physically store file
        # TODO: Need check the target folder?
        if not self.enable_cache:
            filename = os.path.basename(file_name)
            out_folder = os.path.join(self.root,class_name)
            if os.path.exists(out_folder) is False:
                 os.makedirs(out_folder)
            out_file = os.path.join(out_folder, filename)
            
            if os.path.exists(out_file):
                # Work around to fix bug if 1 file have 2 indices... NEED FIX in the partitions.
                split_name = os.path.splitext(out_file)
                out_file = split_name[0] + "_1" + split_name[1]
            #print(self.rank, "save to", out_file, index, target)
            default_saver(out_file, raw_data)

        #self.lock.acquire()
        
        # Replace the item idx
        item = out_file, target
        self.all_samples[index] = item
        self.all_targets[index] = target
        
        #Cache
        if self.enable_cache:
            self.cache[filename] = (raw_data, target)
        #self.lock.release()
        
    def remove_an_item(self, index): 
        # Remove an item from list but still store the file physically.
        #self.lock.acquire()
        if self.samples[index] != None:
            if self.enable_cache:
                path, target = self.samples[index]
                file_name = os.path.basename(path) 
                del self.cache[file_name]
            item = (None,None)
            self.samples[index] = item
            self.targets[index] = None
        #self.lock.release()
    
    def delete_an_item(self, index):
        #self.lock.acquire()
    
        ## Remove and physically delete an item
        path, target = self.samples[index]
        
        # print("remove from ",self.rank, index, path)
        if path is not None:
            ## Becarefull if it is not local file
            os.remove(path)
        
        #self.lock.release()
        self.remove_an_item(index)
        
                
    def next_epoch(self):
        self.samples = self.all_samples.copy()    
        self.targets = self.all_targets.copy()  

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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


def default_saver(path:str, img):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise NotImplementedError
        #return accimage_saver(path)
    else:
        ## image should be an pil.Image object
        # if isinstance(img, Image):
        img.save(path)
     
class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            class_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            enable_cache = False
    ):
        super(ImageFolder, self).__init__(root, class_file, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          enable_cache=enable_cache)
        self.imgs = self.samples
