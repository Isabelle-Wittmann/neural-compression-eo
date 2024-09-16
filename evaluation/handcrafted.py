from evaluation.base_class import CodecTester
import sys
import os
import io
import torch
import tempfile
from osgeo import gdal
import rasterio
from rasterio.enums import Compression
from PIL import Image
from torchvision import transforms
import subprocess


class PillowCodecTester(CodecTester):
    def tensor_pillow_encode(self, img_tensor, quality_layers, fmt, bpp_per_channel):

        # Check if the input is a batch of tensors
        if not isinstance(img_tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        # Convert tensor to PIL image 
        img_pil = transforms.ToPILImage()(img_tensor)
        tmp = io.BytesIO()
        # value = openjpeg.utils.encode_array(np.swapaxes(np.array(img_tensor[i, :, :, :].squeeze().cpu(),dtype='uint8',  order='F'),0,2), signal_noise_ratios=[quality_layers], codec_format=0)
        # Encode PIL image to bytes
        # 
        img_pil.save(tmp,quality=quality_layers, format=fmt) #quality_mode = quality_mode, #, subsampling=0
        tmp.seek(0)
        
        # Calculate file size and bits per pixel
        filesize = tmp.getbuffer().nbytes

        bpp = filesize * 8.0 / (img_pil.size[0] * img_pil.size[1])
        bpp = [bpp/self.bands if bpp_per_channel else bpp][0]

        # bpp = sys.getsizeof(value) * 8 / (img_pil.size[0] * img_pil.size[1])
        
        # Reconstruct image from bytes
        rec_img = transforms.PILToTensor()(Image.open(tmp))/255

        return rec_img, bpp

    def get_metrics(self, fmt, quality):

        for count, data in enumerate(self.dataloader):

            input = (data['image'] if self.is_bigearth_data else data[0])

            decompressed, bpp_val = self.tensor_pillow_encode(input.squeeze(), quality_layers=quality, fmt=fmt, bpp_per_channel= self.bpp_per_channel)
            self.bpp_est_all += [0]
            self.bpp_all += [bpp_val]
            self.compute_distortion_metrics(input.squeeze(), decompressed.squeeze())

            if count % 100 == 0:
                print(f"Processed {count} images")
                self.img_stats(input)
                self.img_stats(decompressed)
                print()
            

class BinaryCodecTester(CodecTester):

    def filesize(self, filepath):
        return os.stat(filepath).st_size
    
    def set_fmt(self, fmt):
        self.fmt = fmt

    def run_command(self, cmd):
        try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            return result.decode("ascii")
        except subprocess.CalledProcessError as err:
            print("Command failed:", err.output.decode("utf-8"))
            sys.exit(1)

    def run_impl(self, in_filepath, quality):

        with tempfile.NamedTemporaryFile(suffix=self.fmt, dir= "u/iwittmann/data", delete=False) as out_file:
            out_filepath = out_file.name
        with tempfile.NamedTemporaryFile(suffix=".png", dir= "u/iwittmann/data", delete=False) as png_file: #png
            png_filepath = png_file.name
        
        self.run_command(self.get_encode_cmd(in_filepath, quality, out_filepath))
        self.run_command(self.get_decode_cmd(out_filepath, png_filepath))

        rec = transforms.PILToTensor()(self.read_image(png_filepath)).float()/255
        bpp_val = float(self.filesize(out_filepath)) * 8.0 / self.num_pixels

        os.remove(png_filepath)
        os.remove(out_filepath)

        return bpp_val, rec

    def get_metrics(self,fmt, quality):

        if  fmt == 'jpeg':
            self.set_fmt('.jpg')
        elif fmt == 'jpeg2000':
            self.set_fmt('.jp2')
        else:
            print("Unknown format")

        temp_dir = tempfile.mkdtemp(dir='./u/iwittmann/data/temp_images')

        for count, data in enumerate(self.dataloader):

            input = (data['image'] if self.is_bigearth_data else data[0])

            temp_image_path = os.path.join(temp_dir,  f'image_{count}.png')
    
            self.save_file(input.squeeze(), temp_image_path)

            bpp_val, decompressed = self.run_impl(temp_image_path, quality)
            
            self.bpp_est_all += [0]
            self.bpp_all += [bpp_val]
            self.compute_distortion_metrics(input.squeeze(), decompressed.squeeze())

            if count % 100 == 0:
                print(f"Processed {count} images")
                self.img_stats(input)
                self.img_stats(decompressed)
                print()

class BinaryCodecTesterMultispectral(BinaryCodecTester):

    def read_image(self, filepath, mode=None):
        return gdal.Open(filepath).ReadAsArray() 
    
    def save_file(self, image, temp_image_path):

        # Ensure the image is in the correct format
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Save the multi-band image as a GeoTIFF
        with rasterio.open(
            temp_image_path,
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=self.bands,  # Number of channels
            dtype='int16',
            compress=Compression.none
        ) as dst:
            for i in range(image.shape[1]):
                dst.write(image[0][i]*255, i + 1)  # Write each channel

class BinaryCodecTesterRGB(BinaryCodecTester):

    def read_image(self, filepath, mode="RGB"):
        return Image.open(filepath).convert(mode)

    def save_file(self, image, temp_image_path):
        image = transforms.ToPILImage()(image)
        image.save(temp_image_path, optimize = False, compress_level = 0)

class BCTRGBMagick(BinaryCodecTesterRGB):
    def get_encode_cmd(self, in_filepath, quality, out_filepath):
        return ["magick", "convert", in_filepath, "-quality", str(quality), "-strip", out_filepath]

    def get_decode_cmd(self, out_filepath, rec_filepath):
        return ["magick", "convert", out_filepath, rec_filepath]

