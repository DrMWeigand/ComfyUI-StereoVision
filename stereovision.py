import torch
from PIL import Image
import numpy as np
import tqdm
from comfy.utils import ProgressBar
import concurrent.futures
from numba import njit
import cv2
from scipy.ndimage import gaussian_filter

# Stereoscopic Generator Class
class StereoscopicGenerator:
    """
    Generates stereoscopic images/videos from a base image and depth map, allowing for a side-by-side 3D effect.
    Suitable for applications where depth-based 3D visuals are desired.
    """
    
    CATEGORY = "🌀 StereoVision"
    DESCRIPTION = "Generates stereoscopic images/videos from a base image and depth map for parallel or cross-eyed viewing."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),      # Original image for stereoscopic conversion
                "depth_map": ("IMAGE",),       # Depth map for 3D depth information
                "depth_scale": ("INT", {"default": 80}),  # Depth effect strength
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_stereoscopic_image"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_single_image(self, base_image_i, depth_map_i, depth_scale):
        # Convert tensor to numpy array and then to PIL Image
        image_np = base_image_i.cpu().numpy() 
        image = Image.fromarray((image_np * 255).astype(np.uint8))

        depth_map_np = depth_map_i.cpu().numpy()
        depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8))

        # Get dimensions and resize depth map to match base image
        width, height = image.size
        depth_map_img = depth_map_img.resize((width, height), Image.NEAREST)

        # Create an empty image for the side-by-side result
        sbs_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        depth_scaling = depth_scale / width 
        pbar = ProgressBar(height)

        # Fill the base images
        image_array = np.array(image)
        sbs_image[:, :width] = image_array
        sbs_image[:, width:] = image_array

        # generating the shifted image
        # Convert depth_map_img to numpy array
        depth_array = np.array(depth_map_img)[:,:,0]
        
        # Calculate pixel shifts
        pixel_shifts = (depth_array * depth_scaling).astype(int)
        
        # Create meshgrid for coordinates
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Calculate new x coordinates
        new_x_coords = np.clip(x_coords + pixel_shifts, 0, width - 1)
        
        # Create mask for valid shifts
        valid_mask = new_x_coords < width
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Create shifted image
        for shift in range(11):  # 0 to 10, inclusive
            shifted_coords = np.clip(new_x_coords + shift, 0, width - 1)
            sbs_image[y_coords[valid_mask], shifted_coords[valid_mask]] = image_array[y_coords[valid_mask], x_coords[valid_mask]]
        
        # Update progress bar
        pbar.update(height)

        # Convert back to tensor if needed
        sbs_image_tensor = torch.tensor(sbs_image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        return sbs_image_tensor

    def generate_stereoscopic_image(self, base_image, depth_map, depth_scale):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of future tasks with their indices
            futures = {
                executor.submit(
                    self.process_single_image, 
                    base_image[i], 
                    depth_map[i], 
                    depth_scale, 
                ): i for i in range(len(base_image))
            }
            
            # Collect results and sort by original index
            processed_images = [None] * len(base_image)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                processed_images[idx] = future.result()
        
        new_ims = torch.cat(processed_images, dim=1)
        return new_ims


# Autostereogram Generator Class
class AutostereogramGenerator:
    """
    Creates autostereograms (Magic Eye images) from depth maps, with optional texture overlay and depth scaling.
    Ideal for hidden 3D effects where images appear when viewed with special focus techniques.
    """
    
    CATEGORY = "🌀 StereoVision"
    DESCRIPTION = "Generates autostereograms (Magic Eye visuals) from depth maps, with options for custom patterns, depth scaling, and texture."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "pattern_div": ("INT", {"default": 8, "min": 4}),
                "invert": ("BOOLEAN", {"default": False}),
                "depth_multiplier": ("FLOAT", {"default": 2, "min": 0.1, "max": 10.0, "step": 0.1}),
                "x_tiles": ("INT", {"default": 8, "min": 1}),
                "y_tiles": ("INT", {"default": 8, "min": 1}),
                "pattern_type": (["random", "perlin", "dots", "lines", "checkers", "waves"], {"default": "random"}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "noise_octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "color_mode": (["grayscale", "rgb", "complementary"], {"default": "rgb"}),
                # Remove clamp_noise_to_first, keep other clamping options
                "clamp_depth_to_first": ("BOOLEAN", {"default": False}),
                "clamp_texture_to_first": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "texture": ("IMAGE",),
                "output_width": ("INT", {"default": 0, "min": 0}),
                "output_height": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_autostereogram"

    def resize_uniformly(self, img, target_width, target_height):
        # Calculate the scaling factor while preserving the aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(aspect_ratio * target_height)
        new_height = int(target_width / aspect_ratio)

        # Choose dimensions based on which scaling factor fits within the target dimensions
        if new_width > target_width:
            new_width = target_width
        else:
            new_height = target_height

        # Resize the image with BICUBIC filtering
        resized_img = img.resize((new_width, new_height), Image.BICUBIC)

        # Calculate the average depth of the edge pixels
        left_edge = np.array(resized_img.crop((0, 0, 1, resized_img.height)))
        right_edge = np.array(resized_img.crop((resized_img.width - 1, 0, resized_img.width, resized_img.height)))
        top_edge = np.array(resized_img.crop((0, 0, resized_img.width, 1)))
        bottom_edge = np.array(resized_img.crop((0, resized_img.height - 1, resized_img.width, resized_img.height)))
        avg_value = int((np.mean(left_edge) + np.mean(right_edge) + np.mean(top_edge) + np.mean(bottom_edge)) / 4)

        # Create a new image with the target resolution, filled with the average depth value
        padded_img = Image.new("L", (target_width, target_height), avg_value)

        # Paste the resized image onto the center of the new image
        padding_left = (target_width - new_width) // 2
        padding_top = (target_height - new_height) // 2
        padded_img.paste(resized_img, (padding_left, padding_top))

        return padded_img

    def process_autostereogram(self, depth_map_i, pattern_div, invert, depth_multiplier, 
                              x_tiles, y_tiles, pattern_type, noise_scale, noise_octaves, 
                              color_mode, texture_i=None, output_width=0, output_height=0,
                              seed=None):  # Changed from cached_noise to seed
        # Convert depth_map_i to a numpy array and then to a PIL Image
        depth_map_np = depth_map_i.cpu().numpy()
        depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8))

        # Handle custom resolution if specified
        if output_width > 0 and output_height > 0:
            depth_map_img = self.resize_uniformly(depth_map_img.convert('L'), output_width, output_height)
        
        # Get dimensions (either original or resized)
        width, height = depth_map_img.size

        # Decide between using texture or noise
        pattern_width = width // pattern_div
        if texture_i is not None:
            texture_np = texture_i.cpu().numpy()
            texture_img = Image.fromarray((texture_np * 255).astype(np.uint8))
            pattern = self.load_texture(texture_img, width, height, x_tiles, y_tiles)
        else:
            pattern = self.generate_noise(
                pattern_width, height, pattern_type, noise_scale, noise_octaves, color_mode)

        # Convert depth map to single channel if not already
        depth_data = np.array(depth_map_img.convert('L'))
        out_data = np.empty((height, width, 3), dtype=np.uint8)

        invert_value = -1 if invert else 1

        # Call the optimized function
        create_autostereogram(depth_data, pattern, invert_value, pattern_width, depth_multiplier, pattern_div, out_data)

        # Convert back to tensor
        out_image_np = out_data.astype(np.float32) / 255.0
        out_image_tensor = torch.from_numpy(out_image_np).unsqueeze(0)
        return out_image_tensor

    def generate_autostereogram(self, depth_map, pattern_div=8, invert=False, 
                                depth_multiplier=1.0, x_tiles=8, y_tiles=8, 
                                pattern_type="random", noise_scale=1.0, 
                                noise_octaves=4, color_mode="rgb", 
                                clamp_depth_to_first=False,
                                clamp_texture_to_first=False,
                                texture=None, output_width=0, output_height=0):
        # Get batch sizes
        depth_batch_size = depth_map.shape[0]
        texture_batch_size = texture.shape[0] if texture is not None else depth_batch_size
        
        # Use the larger batch size
        batch_size = max(depth_batch_size, texture_batch_size)
        results = [None] * batch_size

        # Handle depth map batching
        if clamp_depth_to_first:
            # Clamp to first frame
            depth_map = depth_map[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        elif depth_batch_size < batch_size:
            # Repeat the depth map to match batch size
            repeats = batch_size // depth_batch_size + (1 if batch_size % depth_batch_size else 0)
            depth_map = depth_map.repeat(repeats, 1, 1, 1)[:batch_size]

        # Handle texture batching
        if texture is not None:
            if clamp_texture_to_first:
                # Clamp to first frame
                texture = texture[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            elif texture_batch_size < batch_size:
                # Repeat the texture to match batch size
                repeats = batch_size // texture_batch_size + (1 if batch_size % texture_batch_size else 0)
                texture = texture.repeat(repeats, 1, 1, 1)[:batch_size]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_with_index = []
            
            for i in range(batch_size):
                depth_map_i = depth_map[i]
                texture_i = texture[i] if texture is not None else None
                
                future = executor.submit(
                    self.process_autostereogram,
                    depth_map_i,
                    pattern_div,
                    invert,
                    depth_multiplier,
                    x_tiles,
                    y_tiles,
                    pattern_type,
                    noise_scale,
                    noise_octaves,
                    color_mode,
                    texture_i,
                    output_width,
                    output_height,
                )
                future_with_index.append((future, i))

            for future, index in future_with_index:
                result = future.result()
                results[index] = result

        assert all(r is not None for r in results), "Some frames were not processed"
        
        output_images = torch.cat(results, dim=0)
        return (output_images,)

    def load_texture(self, texture_img, width, height, x_tiles, y_tiles):
        # Adjust the texture size
        new_texture_width = width // x_tiles
        new_texture_height = height // y_tiles
        resized_texture = texture_img.resize((new_texture_width, new_texture_height))

        # Tile the texture
        extended_texture = Image.new("RGB", (width, height))
        for x in range(x_tiles):
            for y in range(y_tiles):
                extended_texture.paste(resized_texture, (new_texture_width * x, new_texture_height * y))

        return np.array(extended_texture)

    def generate_noise(self, width, height, pattern_type="random", noise_scale=1.0, 
                      noise_octaves=4, color_mode="rgb"):
        """
        Generate various types of noise patterns for autostereograms.
        
        Args:
            width (int): Width of the pattern
            height (int): Height of the pattern
            pattern_type (str): Type of pattern to generate
            noise_scale (float): Scale factor for the noise
            noise_octaves (int): Number of octaves for Perlin noise
            color_mode (str): Color mode for the pattern
        """
        # Ensure minimum dimensions
        width = max(1, int(width))
        height = max(1, int(height))
        
        def generate_perlin_noise(shape, scale, octaves):
            if shape[0] == 0 or shape[1] == 0:
                raise ValueError(f"Invalid shape for Perlin noise: {shape}")
            
            noise = np.zeros(shape)
            frequency = 1
            amplitude = 1
            for _ in range(octaves):
                noise += amplitude * gaussian_filter(
                    np.random.randn(*shape),
                    sigma=max(0.1, scale/frequency)
                )
                frequency *= 2
                amplitude *= 0.5
            
            noise_range = noise.max() - noise.min()
            if noise_range == 0:
                return np.zeros_like(noise)
            return (noise - noise.min()) / noise_range

        def create_color_pattern(base_pattern):
            if color_mode == "grayscale":
                return np.stack([base_pattern] * 3, axis=-1)
            elif color_mode == "rgb":
                return np.stack([
                    generate_perlin_noise((height, width), max(0.1, noise_scale), max(1, noise_octaves)),
                    generate_perlin_noise((height, width), max(0.1, noise_scale), max(1, noise_octaves)),
                    generate_perlin_noise((height, width), max(0.1, noise_scale), max(1, noise_octaves))
                ], axis=-1)
            else:  # complementary
                hue = generate_perlin_noise((height, width), max(0.1, noise_scale), max(1, noise_octaves))
                hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
                rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
                return rgb

        base_shape = (height, width)
        
        try:
            if pattern_type == "random":
                pattern = np.random.rand(*base_shape)
            
            elif pattern_type == "perlin":
                pattern = generate_perlin_noise(base_shape, max(0.1, noise_scale), max(1, noise_octaves))
            
            elif pattern_type == "dots":
                pattern = np.zeros(base_shape)
                dot_size = max(1, int(min(width, height) * noise_scale / 20))
                for _ in range(max(1, noise_octaves * 100)):
                    x = np.random.randint(0, max(1, width))
                    y = np.random.randint(0, max(1, height))
                    cv2.circle(pattern, (x, y), dot_size, 1.0, -1)
            
            elif pattern_type == "lines":
                pattern = np.zeros(base_shape)
                line_thickness = max(1, int(min(width, height) * noise_scale / 50))
                for _ in range(max(1, noise_octaves * 10)):
                    x1, y1 = np.random.randint(0, max(1, width)), np.random.randint(0, max(1, height))
                    x2, y2 = np.random.randint(0, max(1, width)), np.random.randint(0, max(1, height))
                    cv2.line(pattern, (x1, y1), (x2, y2), 1.0, line_thickness)
            
            elif pattern_type == "checkers":
                cell_size = max(1, int(min(width, height) * noise_scale / 10))
                pattern = np.indices(base_shape).sum(axis=0) % (cell_size * 2)
                pattern = (pattern < cell_size).astype(float)
            
            elif pattern_type == "waves":
                x = np.linspace(0, max(0.1, noise_scale * 10), max(1, width))
                y = np.linspace(0, max(0.1, noise_scale * 10), max(1, height))
                X, Y = np.meshgrid(x, y)
                pattern = np.sin(X) * np.cos(Y)
                for i in range(2, max(1, noise_octaves + 1)):
                    pattern += np.sin(X * i) * np.cos(Y * i) / i

            # Normalize pattern safely
            pattern_range = pattern.max() - pattern.min()
            if pattern_range == 0:
                pattern = np.zeros_like(pattern)
            else:
                pattern = (pattern - pattern.min()) / pattern_range

            # Convert to color pattern
            colored_pattern = create_color_pattern(pattern)
            
            return (colored_pattern * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: Failed to generate {pattern_type} pattern: {str(e)}")
            # Fallback to random noise
            fallback = np.random.rand(*base_shape, 3)
            return (fallback * 255).astype(np.uint8)

# Optimize the loop with Numba
@njit
def create_autostereogram(depth_data, pattern, invert_value, pattern_width, depth_multiplier, pattern_div, out_data):
    height, width = depth_data.shape
    pattern_height, pattern_width_pattern, _ = pattern.shape
    for y in range(height):
        for x in range(width):
            if x < pattern_width:
                out_data[y, x, :] = pattern[y % pattern_height, x % pattern_width_pattern, :]
            else:
                shift = int(depth_data[y, x] * depth_multiplier) // pattern_div
                shifted_x = x - pattern_width + (shift * invert_value)
                if 0 <= shifted_x < width:
                    out_data[y, x, :] = out_data[y, shifted_x, :]
                else:
                    out_data[y, x, :] = 0







