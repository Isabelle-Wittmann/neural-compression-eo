import os
from typing import Callable, Optional
import torch
import rasterio
import numpy as np
import rasterio.features
import rasterio.warp
from rasterio.enums import Resampling
import rasterio
from rasterio.enums import Resampling
from torch import Tensor
import re
from torchgeo.datasets.geo import NonGeoDataset
from typing import Optional, Callable, List, Dict, Tuple
from collections import defaultdict
from rasterio.errors import RasterioIOError
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pickle

def save_filtered_data_by_indices(dataset, save_path, indices):
    os.makedirs(save_path, exist_ok=True)
    filtered_data = [dataset[i] for i in indices]
    with open(os.path.join(save_path, 'filtered_data.pkl'), 'wb') as f:
        pickle.dump(filtered_data, f)


class Dataset_Sent2(NonGeoDataset):
    
    def __init__(
        self,
        base_directory: str = "data",
        transforms: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            base_directory: The directory containing all the .tiff files
            transforms: Optional transform function to apply to each sample
        """
        self.transforms = transforms
        self.grouped_files = self.group_files_by_common_identifier(base_directory)
    
    def group_files_by_common_identifier(self, base_directory: str) -> Dict[str, List[str]]:
        """Group files by their common identifiers.

        Args:
            base_directory: The directory containing all the .tiff files.

        Returns:
            A dictionary where the keys are common identifiers and the values are lists of file paths.
        """

        grouped_files = defaultdict(list)

        for file in os.listdir(base_directory):
            if file.endswith(".tiff"):
                # Extract the common part of the filename
                match = re.match(r"Sentinel2_B\d+_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{4})_(\d{2})_(\d{2})", file)
                if match:
                    # Create the common identifier from the captured groups
                    common_identifier = f"{match.group(1)}_{match.group(2)}_{match.group(3)}_{match.group(4)}_{match.group(5)}"
                    full_path = os.path.join(base_directory, file)
                    grouped_files[common_identifier].append(full_path)
        
        return grouped_files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
      
        image, crs, date = self._load_image(index)

        sample: dict[str, Tensor] = {"image": image, 'crs': crs, 'date': date}

        # if self.transforms is not None:
        #     sample = self.transforms(sample)

        #Below is adapted and replaces return sample   
        # if 1000 <= sample['image'].max() <= 10000:
  
        # sample = self.transforms(sample)
        return sample
        
        # else:
        #     return self.__getitem__(index+1)
            

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.grouped_files.keys())

    def _load_paths(self, index: int) -> List[str]:
        """Load paths to band files for the given index."""
        # Get the common identifier (key) from the index
        common_identifier = list(self.grouped_files.keys())[index]
        return sorted(self.grouped_files[common_identifier], key=lambda x: int(re.search(r"_B(\d+)", x).group(1)))

    def _load_image(self, index: int) -> torch.Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
    
        def extract_coordinates_and_date(file_path):
            # Regular expression to extract coordinates and date
            pattern = r"Sentinel2_B\d+_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{4})_(\d{2})_(\d{2})\.tiff"
            match = re.search(pattern, file_path)
            
            if match:
                lon, lat = float(match.group(1)), float(match.group(2))
                date = f"{match.group(3)}-{match.group(4)}-{match.group(5)}"
                return lon, lat, date
            else:
                raise ValueError(f"Cannot extract information from {file_path}")

        paths = self._load_paths(index)
        images = []

        try:
            for path in paths:
                with rasterio.open(path) as dataset:
                    # Read the data and resample it to a fixed size
                    array = dataset.read(
                        indexes=1,
                        out_dtype="int32",
                        resampling=Resampling.bilinear,
                    )
                    images.append(array)

            arrays = np.stack(images, axis=0)
            tensor = torch.from_numpy(arrays).float()

            # Extract the coordinate and date information from the first path
            lon, lat, date = extract_coordinates_and_date(paths[0])
            crs = torch.tensor([lon, lat])

            return tensor, crs, date

        except RasterioIOError as e:
            print(f"Error reading {paths}: {e}")
            # # Return a default tensor filled with zeros in case of an error
            # default_array = np.zeros((9, *self.image_size), dtype=np.float32)
            # return torch.from_numpy(default_array).float(), None, None


class Dataset_Sent2_TS(NonGeoDataset):
    
    def __init__(
        self,
        base_directory: str = "data",
        transforms: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            base_directory: The directory containing all the .tiff files
            transforms: Optional transform function to apply to each sample
        """
        self.transforms = transforms
        self.grouped_files, coord, dates = self.group_files_by_coordinate(base_directory)
        self.coordinates = coord
        self.dates = dates
 
    def save_zero_array_as_tiff(self, file_path: str):
        """Save a 256x256 zero array as a .tiff file."""
        zero_array = np.zeros((256, 256), dtype=np.uint8)
        image = Image.fromarray(zero_array)
        image.save(file_path)

    def group_files_by_coordinate(self, base_directory: str) -> Dict[Tuple[float, float], Dict[str, List[str]]]:
        """Group files by their coordinates and then by date.

        Args:
            base_directory: The directory containing all the .tiff files.

        Returns:
            A dictionary where the keys are tuples of coordinates and the values are dictionaries 
            with dates as keys and lists of file paths as values.
        """

        grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        coord = []
        dates = []

        for file in os.listdir(base_directory):
            if file.endswith(".tiff"):
                # Extract coordinates and date from the filename
                match = re.match(r"Sentinel2_B(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d{4})_(\d{2})_(\d{2})", file)
                # if match == None:
                #     print(file)
                if match:
                    band = int(match.group(1))
                    lon, lat = float(match.group(2)), float(match.group(3))
                    date = f"{match.group(4)}-{match.group(5)}-{match.group(6)}"
                    full_path = os.path.join(base_directory, file)
                    grouped_files[(lon, lat)][date][band].append(full_path)
                    coord += [(lon, lat)]
                    dates += [date]
        # path_zero_array = '/u/iwittmann/data/zeroes.tiff'
        # self.save_zero_array_as_tiff(path_zero_array)
        # # Sort and replace missing bands with None
        # grouped_files = {
        #         coordinate: {
        #             date: [band_dict.get(band, path_zero_array) for band in range(0, 13)]
        #             for date, band_dict in date_dict.items()
        #         }
        #         for coordinate, date_dict in grouped_files.items()
        #     }
        # print(grouped_files[(lon, lat)])

        return grouped_files, list(set(coord)), list(set(dates))
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        # Get the coordinate and date based on the index
        item_coordinates = self.coordinates[index]

        coordinate_dict = self.grouped_files[item_coordinates]

        images = []
        dates = []
        crs = item_coordinates
        for date in sorted(coordinate_dict.keys()):

            images.append(self._load_image(coordinate_dict[date]))
            dates.append(date)

        sample: dict[str, torch.Tensor] = {"images": images, 'crs': crs, 'dates': dates}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
        
    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.coordinates) 

    def _load_image(self, paths) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Load a single image.

        Args:
            coordinate: The coordinate (lon, lat) tuple to load the image for.
            date: The date string to load the image for.

        Returns:
            the raster image or target, coordinates as tensor, and date as string
        """

        images = []
        rgb_paths = [paths[1], paths[2], paths[3]]

        try:
            for path in rgb_paths:
                
                with rasterio.open(path[0]) as dataset:
                    # Read the data and resample it to a fixed size
                    array = dataset.read(
                        indexes=1,
                        out_dtype="int32",
                        resampling=Resampling.bilinear,
                    )
                    images.append(array)

            arrays = np.stack(images, axis=0)
            tensor = torch.from_numpy(arrays).float()

            return tensor
        
        except RasterioIOError as e:
            print(f"Error reading {paths}: {e}")
            raise

class PreloadedDataset(Dataset):
    def __init__(self, data_path):
        with open(os.path.join(data_path, 'filtered_data.pkl'), 'rb') as f:
            self.data_list = pickle.load(f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
def get_dataloaders():
    # # Specify the base directory
    # base_directory = "/dccstor/geofm-finetuning/isabellewittmann/data/sent2"

    # # Create an instance of your dataset
    # dataset = Dataset_Sent2_TS(base_directory=base_directory)

    # # Indices you want to include in the DataLoader
    # full_indices = [0,1,3,6,10,11,12,14,18,19,20,23,25,26,28,34,35,36,37,38,39,40,42,43,44,48,49,51,54,60,61,62,63,65,67,68,71,72,76,77,80,81,85,87,88,90,93,94,95,99]

    # # Create a Subset of your dataset with the specific indices
    # subset = Subset(dataset, full_indices)

    # # Create a DataLoader with batch size 1
    # data_loader = DataLoader(subset, batch_size=1, shuffle=False)

    # # Example of iterating through the DataLoader
    # for i, data in enumerate(data_loader):
    #     print(f"Batch {i}:")
    #     print(data['dates'][0])
    #     print(data['dates'][-1])

    # base_directory = "/dccstor/geofm-finetuning/isabellewittmann/data/sent2"
    # dataset = Dataset_Sent2_TS(base_directory=base_directory)
    save_path = "/u/iwittmann/data/timeseries"

    # Indices you want to save
    full_indices = [0,1,3,6,10,11,12,14,18,19,20,23,25,26,28,34,35,36,37,38,39,40,42,43,44,48,49,51,54,60,61,62,63,65,67,68,71,72,77,78,81,85,87,88,90,93,94,95,99]

    # Load the pre-saved data and use it in the DataLoader
    preloaded_dataset = PreloadedDataset(data_path=save_path)
    # Create a DataLoader with batch size 1
    data_loader_ts = DataLoader(preloaded_dataset, batch_size=1, shuffle=False)
   
    # all_images = []
    # for i in range(0, 49):
    #     if i == 40 or i == 24:
    #         continue
    #     all_images.extend(preloaded_dataset[i]['images'])

    # data_loader_all = DataLoader(all_images, batch_size=1, shuffle=False)

    return data_loader_ts


   