import os
import yaml
from typing import Any, Dict
import numpy as np
import torch
from torchvision import transforms
from torchgeo.datasets import BigEarthNet
from typing import Callable, Optional
from torch import Tensor
import rasterio
from rasterio.enums import Resampling

class DictTransforms:
    def __init__(self,dict_transform : dict,):
        self.dict_transform = dict_transform

    def __call__(self, sample):
        for key, function in self.dict_transform.items():
            sample[key] = function(sample[key])
        return sample

class SelectChannels:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels]

class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(dim=self.dim)

class ConvertType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype)
    
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def normalize(sample: Dict[str, Any], maxs: torch.Tensor, mins: torch.Tensor) -> Dict[str, Any]:
    """Normalize a single sample from the Dataset."""
    sample = sample.float()
    sample = (sample - mins) / (maxs - mins)
    sample = torch.clip(sample, min=0.0, max=1.0)
    return sample
    
def init_bigearthnet(cfg_path, bands, prefilter):
    """
    Initialize the BigEarthNet dataset with specified configuration.
    """
    # Load config file
    cfg = load_config(cfg_path)

    # Get dataset parameters
    split = cfg['dataset']['split']
    satellite = cfg['dataset']['satellite'] if 'satellite' in cfg['dataset'] else 's2'
    resize = cfg['dataset']['resize']

    # Get the directory for BigEarthNet data, defaulting to 'data' if not set
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    bigearthnet_dir = os.path.join(DATA_DIR, 'BigEarthNet')

    # Check if the BigEarthNet data directory exists
    assert os.path.isdir(os.path.join(bigearthnet_dir, BigEarthNet.metadata['s2']['directory'])), \
        "Download BigEarthNet with `sh datasets/bigearthnet_download.sh` or specify the DATA_DIR via an env variable."

    # Initialize image transformations
    image_transforms = [
        SelectChannels(bands),          # Select specified bands
        ConvertType(torch.float),       # Convert data type to float
        transforms.Resize(tuple(resize)), # Resize images to specified dimensions
    ]

    # If normalization is specified in the configuration, add it as preprocessing step
    if cfg['dataset']['normalize']:
        # Extract max and min values for the specified bands
        maxs = torch.tensor([cfg['preprocess']['maxs'][i] for i in bands]).unsqueeze(1).unsqueeze(1)
        mins = torch.tensor([cfg['preprocess']['mins'][i] for i in bands]).unsqueeze(1).unsqueeze(1)
        # image_transforms.append(lambda sample: filter_images_by_max_value(sample))
        image_transforms.append(lambda sample: normalize(sample, maxs, mins))

    # Wrap the transformations in a dictionary-based transform
    transforms_bigearth = DictTransforms({'image': transforms.Compose(image_transforms)})

    # Initialize the BigEarthNet dataset with the specified parameters and transformations
    dataset = BigEarthNetCustom(
        root=bigearthnet_dir,
        split=split,
        bands=satellite,
        num_classes=cfg['dataset']['num_classes'],
        transforms=transforms_bigearth,
        prefilter = prefilter
    )

    return dataset, cfg['preprocess']['max_val']

class BigEarthNetCustom(BigEarthNet):
    """
    Taken from torchgeo and adapted to extract metadata information and be able to perform filtering of the data."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
        prefilter: bool = False
    ) -> None:

        assert split in self.splits_metadata
        assert bands in ["s1", "s2", "all"]
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}
        self._verify()
        self.folders = self._load_folders()

        # Coordinates of Finland and Portugal
        self.country1_lat_range = (36.96, 42.15)
        self.country1_lon_range = (-9.50, -6.19)
        self.country2_lat_range = (59.81, 70.09)
        self.country2_lon_range = (19.08, 31.59)
        if prefilter:
            self.prefilter = True
            self._pre_filter_by_country()
        else:
            self.prefilter = False
    
    def _is_within_country(self, lat, lon):
        # Check if the lat, lon are within the bounds of either country 1 or country 2
        in_country1 = self.country1_lat_range[0] <= lat <= self.country1_lat_range[1] and self.country1_lon_range[0] <= lon <= self.country1_lon_range[1]
        in_country2 = self.country2_lat_range[0] <= lat <= self.country2_lat_range[1] and self.country2_lon_range[0] <= lon <= self.country2_lon_range[1]
        
        return in_country1 or in_country2

    def _pre_filter_by_country(self) -> None:
        """Pre-filter the dataset based on latitude and longitude ranges for two countries."""
        self.filtered_indices = []

        for index in range(len(self.folders)):
            if (29952 > index > 14709) or (120161 > index > 100071):
                # _, crs, _, _, _ = self._load_image(index)
                # lat, lon = crs[1].item(), crs[0].item()

                # if self._is_within_country(lat, lon):
                # print(index)
                self.filtered_indices.append(index)


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        if self.prefilter:
            index = self.filtered_indices[index]

        image, crs, date, time, index = self._load_image(index)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {"image": image, "label": label, 'crs': crs, 'date': date, 'time': time}

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        #Below is adapted and replaces return sample   
        # if 1000 <= sample['image'].max() <= 10000:
  
        sample = self.transforms(sample)
        return sample
        
        # else:
        #     return self.__getitem__(index+1)
            

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        if self.prefilter:
            return len(self.filtered_indices)
        else:
            return len(self.folders)


    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        import re
        from rasterio.errors import RasterioIOError
        
        def get_time(file_path):
            # Regular expression pattern to extract date and time
            pattern = r"_MSIL2A_(\d{8})T(\d{5,6})_"

            # Search for the pattern in the file path
            match = re.search(pattern, file_path)

            if match:
                # Extract date and time
                date_str = match.group(1)
                time_str = match.group(2)

                # Format the date and time
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"

                return formatted_date, formatted_time
            else:
                print("Date and time not found in the file path.")
                print(file_path)
    
        paths = self._load_paths(index)
        if len(paths) == 0:
            print('Skipped')
            if index > 2:
                index = index-1
                paths = self._load_paths(index)
            else:
                index = index+1
                paths = self._load_paths(index)
        images = []
        metadata = []
        try:
            for path in paths:
                # Bands are of different spatial resolutions
                # Resample to (120, 120)
                
                with rasterio.open(path) as dataset:
                    mask = dataset.dataset_mask()
                    for geom, val in rasterio.features.shapes(
                        mask, transform=dataset.transform):

                    # Transform shapes from the dataset's own coordinate
                    # reference system to CRS84 (EPSG:4326).
                        geom = rasterio.warp.transform_geom(
                            dataset.crs, 'EPSG:4326', geom, precision=6)
                    array = dataset.read(
                        indexes=1,
                        out_shape=self.image_size,
                        out_dtype="int32",
                        resampling=Resampling.bilinear,
                    )
                    images.append(array)
                    crs = torch.tensor([geom['coordinates'][0][0][0],geom['coordinates'][0][0][1]]) #int(dataset.crs.to_dict()['init'][5:])

            arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
            tensor = torch.from_numpy(arrays).float()
            date, time = get_time(path)
            return tensor, crs, date, time, index
        except RasterioIOError as e:
            print(f"Error reading {self.file_paths[index]}: {e}")
            # Return a default array filled with zeros
            pass 
            # default_array = np.zeros(self.default_shape, dtype=np.float32)
            # return default_array, None, None, None

