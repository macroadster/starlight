import onnxruntime
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import os
import struct
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import json
from scripts.starlight_extractor import extraction_functions
from scripts.starlight_utils import load_unified_input # Import from trainer.py




# --- Worker-specific globals ---
SESSION = None
MODEL_TYPE = None  # 'onnx' or 'pytorch'
DEVICE = None  # 'cpu', 'cuda', or 'mps'
METHOD_MAP = {0: "alpha", 1: "palette", 2: "lsb.rgb", 3: "exif", 4: "raw"}
NO_HEURISTICS = False # Global flag to disable heuristics for benchmarking

# Import PyTorch model class
class BalancedStarlightDetector(nn.Module):
    """Balanced model with weighted metadata processing to reduce EXIF dominance"""

    def __init__(self, meta_weight=0.1):
        super(BalancedStarlightDetector, self).__init__()

        self.meta_weight = meta_weight  # Weight to reduce metadata dominance

        # Metadata stream (now 2048 features instead of 1024)
        self.meta_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        # Alpha stream
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        # LSB stream
        self.lsb_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        # Palette stream (fully connected)
        self.palette_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # Format features stream
        self.format_features_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Content features stream
        self.content_features_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Fusion and classification
        self.fusion_dim = 128 * 16 + 64 * 8 * 8 + 64 * 8 * 8 + 64 + 16 + 16 + 3  # +3 for bit_order features
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128 + 64)  # 128 for embedding, 64 for classification
        )

        # Heads
        self.stego_head = nn.Linear(64, 1)
        self.method_head = nn.Linear(64, 5)  # alpha, palette, lsb.rgb, exif, raw
        self.embedding_head = nn.Linear(64, 64)

    def forward(self, meta, alpha, lsb, palette, format_features, content_features, bit_order):
        # Metadata stream with weighting
        meta = meta.unsqueeze(1)  # Add channel dimension
        meta = self.meta_conv(meta)
        meta = meta.reshape(meta.size(0), -1)
        meta = meta * self.meta_weight  # Apply weighting to reduce dominance

        # Alpha stream
        alpha = self.alpha_conv(alpha)
        alpha = alpha.reshape(alpha.size(0), -1)

        # LSB stream - should already be in CHW format when it reaches here
        if lsb.dim() == 4:  # Already in (N, C, H, W) format
            pass  # Already correct
        elif lsb.dim() == 3:  # (C, H, W) format
            lsb = lsb.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected LSB tensor shape: {lsb.shape}")
        
        lsb = self.lsb_conv(lsb)
        lsb = lsb.reshape(lsb.size(0), -1)

        # Palette stream
        palette = self.palette_fc(palette)

        # Format features stream
        format_features = self.format_features_fc(format_features)

        # Content features stream
        content_features = self.content_features_fc(content_features)

        # Fusion with bit_order features
        fused = torch.cat([meta, alpha, lsb, palette, format_features, content_features, bit_order], dim=1)
        fused = self.fusion(fused)

        # Split into embedding and classification features
        embedding = fused[:, :128]
        cls_features = fused[:, 128:]

        # Outputs
        stego_logits = self.stego_head(cls_features)
        method_logits = self.method_head(cls_features)
        embedding = self.embedding_head(cls_features)

        # Get method predictions
        method_probs = torch.softmax(method_logits, dim=1)
        method_id = torch.argmax(method_probs, dim=1)

        return stego_logits, method_logits, method_id, method_probs, embedding

def init_worker(model_path):
    """Initializer for each worker process."""
    global SESSION, MODEL_TYPE, DEVICE
    import piexif # Re-import piexif in the worker process
    
    # Preload commonly used modules to reduce import overhead
    import PIL.Image
    import numpy as np
    
    # Determine model type and device
    if model_path.endswith('.pth'):
        MODEL_TYPE = 'pytorch'
        # Check for MPS availability on Mac
        if torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
        elif torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = torch.device('cpu')
        
        # Load PyTorch model with optimizations
        model = BalancedStarlightDetector()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Optimize for inference
        if DEVICE.type == 'mps':
            # MPS-specific optimizations (disable compile due to compatibility issues)
            pass  # torch.compile has issues with MPS conv2d
        elif DEVICE.type == 'cuda':
            # CUDA-specific optimizations
            with torch.no_grad():
                for param in model.parameters():
                    param.requires_grad = False
            model = torch.jit.script(model)  # TorchScript optimization
        
        SESSION = model
    else:
        MODEL_TYPE = 'onnx'
        DEVICE = None
        # ONNX Runtime optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        SESSION = onnxruntime.InferenceSession(model_path, sess_options)

def _scan_logic(image_path, session, extract_message=False, fast_mode=False):
    """The actual scanning logic, independent of the execution context."""
    global METHOD_MAP, MODEL_TYPE, DEVICE
    try:

        pixel_tensor, meta, alpha, lsb, palette, format_features, content_features = load_unified_input(image_path, fast_mode=fast_mode)
        
        # Create bit_order feature (default to msb-first for inference)
        bit_order = torch.tensor([[0.0, 1.0, 0.0]])  # [lsb-first, msb-first, none]
        
        # Move tensors to GPU/MPS if using PyTorch to avoid CPU bottlenecks
        if MODEL_TYPE == 'pytorch' and DEVICE is not None:
            meta = meta.to(DEVICE)
            alpha = alpha.to(DEVICE)
            # Convert LSB from HWC to CHW before moving to device
            if lsb.dim() == 3 and lsb.shape == torch.Size([256, 256, 3]):  # HWC format
                lsb = lsb.permute(2, 0, 1)  # HWC -> CHW
            lsb = lsb.to(DEVICE)
            palette = palette.to(DEVICE)
            format_features = format_features.to(DEVICE)
            content_features = content_features.to(DEVICE)
            bit_order = bit_order.to(DEVICE)

        # Add batch dimension for ONNX model
        meta = meta.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        lsb = lsb.unsqueeze(0)
        palette = palette.unsqueeze(0)
        format_features = format_features.unsqueeze(0)
        content_features = content_features.unsqueeze(0)
        
        # Run inference
        if MODEL_TYPE == 'pytorch':
            # PyTorch inference
            with torch.no_grad():
                meta = meta.to(DEVICE)
                alpha = alpha.to(DEVICE)
                lsb = lsb.to(DEVICE)
                palette = palette.to(DEVICE)
                format_features = format_features.to(DEVICE)
                content_features = content_features.to(DEVICE)
                
                stego_logits, method_logits, method_id, method_probs, embedding = session(meta, alpha, lsb, palette, format_features, content_features, bit_order)
                
                stego_logits = stego_logits.cpu().numpy()
                method_id = method_id.cpu().numpy()
                method_probs = method_probs.cpu().numpy()
        else:
            # ONNX inference - need to permute LSB to match ONNX expected format
            if lsb.dim() == 3:  # HWC format
                lsb_onnx = lsb.permute(2, 0, 1)  # HWC -> CHW
            else:
                lsb_onnx = lsb  # Already correct format
            
            # Prepare inputs for ONNX (add batch dimension where needed)
            input_feed = {
                'meta': meta.numpy(),
                'alpha': alpha.numpy(),
                'lsb': lsb_onnx.numpy(),
                'palette': palette.numpy(),
                'format_features': format_features.numpy(),
                'content_features': content_features.numpy(),
                'bit_order': bit_order.numpy()
            }
            stego_logits, _, method_id, method_probs, _ = session.run(None, input_feed)

        # Process results
        # Use a numerically stable sigmoid to avoid overflow warnings
        logit = stego_logits[0][0]
        if logit >= 0:
            stego_prob = 1 / (1 + np.exp(-logit))
        else:
            # Use the equivalent alternative form to avoid overflow for large negative logits
            stego_prob = np.exp(logit) / (1 + np.exp(logit))

        if not NO_HEURISTICS:
            # Method-specific thresholds tuned to reduce false positives based on analysis
            thresholds = {0: 0.7, 1: 0.98, 2: 0.95, 3: 0.5, 4: 0.95}  # Raised alpha and palette thresholds
            threshold = thresholds.get(method_id[0], 0.8)
            is_stego = stego_prob > threshold
            
            # Strong validation for high-confidence LSB predictions to prevent systematic false positives
            if method_id[0] == 2 and is_stego and stego_prob > 0.9:  # High confidence LSB
                try:
                    lsb_message, _ = extraction_functions['lsb'](image_path)
                    # If extraction returns None or empty, it's definitely a false positive
                    if not lsb_message:
                        is_stego = False
                        stego_prob = 0.1  # Very low confidence
                    else:
                        # Check for patterns indicating false positive
                        total_chars = len(lsb_message)
                        unique_chars = len(set(lsb_message))
                        
                        # If message is all hex characters and repetitive pattern, likely false positive
                        hex_chars = sum(1 for c in lsb_message if c.lower() in '0123456789abcdef')
                        if hex_chars == total_chars and unique_chars <= 16:
                            # Check for repetitive hex patterns (like ffffff, 000000)
                            if 'ffff' in lsb_message or '0000' in lsb_message:
                                is_stego = False
                                stego_prob = 0.2
                except:
                    is_stego = False
                    stego_prob = 0.1
            
            # Strong validation for alpha steganography to prevent model errors
            if method_id[0] == 0 and is_stego:  # Predicted as alpha steganography
                img = Image.open(image_path)
                # CRITICAL FIX: If image has no alpha channel, it cannot be alpha steganography
                if img.mode != 'RGBA':
                    is_stego = False
                    stego_prob = 0.01  # Very low confidence for non-alpha images
                else:
                    alpha_channel = img.split()[-1]
                    alpha_data = np.array(alpha_channel)
                    # If alpha is uniform (all 255), it cannot be alpha steganography
                    if alpha_data.std() == 0 or np.sum(alpha_data != 255) == 0:
                        is_stego = False
                        stego_prob = min(stego_prob, 0.1)  # Very low confidence
            
            # Special case: Override model decision for alpha steganography when clearly present
            # This fixes the metadata bias issue where PNG files with metadata are incorrectly classified as clean
            # Also handles cases where alpha steganography signal is weak but contains valid content
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                alpha_channel = img.split()[-1]
                alpha_data = np.array(alpha_channel)
                # Only check for alpha steganography if there's actual alpha variation
                if alpha_data.std() > 0 and np.sum(alpha_data != 255) > 100:
                    # Try to extract and validate alpha content
                    try:
                        alpha_message, _ = extraction_functions['alpha'](image_path)
                        # If alpha contains readable text content (not just binary), it's alpha stego
                        if alpha_message and len(alpha_message) > 10 and any(c.isalpha() and c.islower() for c in alpha_message):
                            # Override to alpha method if it wasn't detected as alpha
                            if method_id[0] != 0:
                                method_id[0] = 0
                                stego_prob = max(stego_prob, 0.8)
                    except:
                        pass
            
            # Special case: Reduce palette false positives by validating extracted content
            if method_id[0] == 1 and is_stego:  # Detected as palette steganography
                try:
                    palette_message, _ = extraction_functions['palette'](image_path)
                    # If palette extraction returns None or empty, it's likely a false positive
                    if not palette_message:
                        is_stego = False
                        stego_prob = 0.1  # Very low confidence
                    else:
                        # Check for patterns indicating false positive in palette data
                        total_chars = len(palette_message)
                        unique_chars = len(set(palette_message))
                        
                        # If message is all same character or very repetitive, likely false positive
                        if unique_chars <= 2 and total_chars > 10:
                            is_stego = False
                            stego_prob = min(stego_prob, 0.3)
                        # If message is mostly null/control characters, likely false positive
                        elif sum(1 for c in palette_message if ord(c) < 32 or ord(c) > 126) / total_chars > 0.8:
                            is_stego = False
                            stego_prob = min(stego_prob, 0.2)
                except:
                    is_stego = False
                    stego_prob = 0.1

            # Special case: Reduce LSB false positives by validating extracted content
            if method_id[0] == 2 and is_stego:  # Detected as LSB steganography
                try:
                    lsb_message, _ = extraction_functions['lsb'](image_path)
                    # If LSB message is mostly repeated characters or meaningless, it's likely false positive
                    if lsb_message:
                        # Check for patterns indicating false positive
                        unique_chars = set(lsb_message)
                        total_chars = len(lsb_message)
                        
                        # If message is mostly repeated characters (>80% same char) or very short, likely false positive
                        if len(unique_chars) <= 2 and total_chars > 20:
                            # Calculate ratio of most common character
                            most_common_ratio = max(lsb_message.count(c) for c in unique_chars) / total_chars
                            if most_common_ratio > 0.8:
                                is_stego = False
                                stego_prob = min(stego_prob, 0.4)  # Reduce confidence
                        # If message is all same character repeated, definitely false positive
                        elif len(unique_chars) == 1 and total_chars > 10:
                            is_stego = False
                            stego_prob = min(stego_prob, 0.2)
                        # If message is mostly null bytes or control characters, likely false positive
                        elif sum(1 for c in lsb_message if ord(c) < 32) / total_chars > 0.7:
                            is_stego = False
                            stego_prob = min(stego_prob, 0.3)
                except:
                    pass
        else:
            # When heuristics are disabled, use a simple 0.5 threshold for benchmarking
            threshold = 0.5
            is_stego = stego_prob > threshold
        result = {
            "file_path": str(image_path),
            "is_stego": bool(is_stego),
            "stego_probability": float(stego_prob),
            "method_id": int(method_id[0]),
        }

        if is_stego:
            stego_type = METHOD_MAP.get(method_id[0], "unknown")
            result["stego_type"] = stego_type
            result["confidence"] = float(np.max(method_probs))

            if extract_message:
                # Handle method name mapping for extractor
                extractor_method_name = stego_type
                if stego_type == "lsb.rgb":
                    extractor_method_name = "lsb"
                elif stego_type == "raw":
                    extractor_method_name = "eoi"

                if extractor_method_name in extraction_functions:
                    try:
                        extractor = extraction_functions[extractor_method_name]
                        message, _ = extractor(image_path)
                        if message:
                            result["extracted_message"] = message
                    except Exception as e:
                        result["extraction_error"] = str(e)

        return result
    except Exception as e:
        error_message = f"Could not process {image_path}: {e}"
        return {"file_path": str(image_path), "error": error_message}

def scan_image_worker(image_path):
    """The actual scanning logic that runs in each worker process."""
    global SESSION
    # Extraction is disabled for directory scans for performance
    return _scan_logic(image_path, SESSION, extract_message=False)

def scan_batch_worker(image_paths):
    """Process a batch of images for better GPU utilization."""
    global SESSION, MODEL_TYPE, DEVICE
    results = []
    
    # Smart fast mode: detect LSB-heavy workloads
    lsb_files = sum(1 for path in image_paths if 'lsb' in path.lower())
    lsb_ratio = lsb_files / len(image_paths) if image_paths else 0
    
    # Disable fast mode completely to ensure 100% detection accuracy
    fast_mode = False
    
    # For now, use individual processing but with optimizations
    # True batch processing requires more complex tensor handling
    for path in image_paths:
        try:
            result = _scan_logic(path, SESSION, extract_message=False, fast_mode=fast_mode)
            results.append(result)
        except Exception as e:
            # Handle errors gracefully without crashing the entire batch
            error_msg = str(e)
            if "cannot identify image file" in error_msg:
                # Skip non-image files silently
                continue
            results.append({"file_path": path, "error": f"Could not process {path}: {error_msg}"})
    
    return results

def _process_inference_result(image_path, stego_logits, method_id, method_probs, extract_message=False):
    """Process inference results into final result format."""
    global METHOD_MAP
    
    # Use a numerically stable sigmoid to avoid overflow warnings
    logit = stego_logits[0]
    if logit >= 0:
        stego_prob = 1 / (1 + np.exp(-logit))
    else:
        stego_prob = np.exp(logit) / (1 + np.exp(logit))

    # Method-specific thresholds tuned to reduce false positives based on analysis
    thresholds = {0: 0.7, 1: 0.98, 2: 0.95, 3: 0.5, 4: 0.95}
    threshold = thresholds.get(method_id[0], 0.8)
    is_stego = stego_prob > threshold
    
    result = {
        "file_path": str(image_path),
        "is_stego": bool(is_stego),
        "stego_probability": float(stego_prob),
        "method_id": int(method_id[0]),
    }

    if is_stego:
        stego_type = METHOD_MAP.get(method_id[0], "unknown")
        result["stego_type"] = stego_type
        result["confidence"] = float(np.max(method_probs))

        if extract_message:
            # Handle method name mapping for extractor
            extractor_method_name = stego_type
            if stego_type == "lsb.rgb":
                extractor_method_name = "lsb"
            elif stego_type == "raw":
                extractor_method_name = "eoi"

            if extractor_method_name in extraction_functions:
                try:
                    extractor = extraction_functions[extractor_method_name]
                    message, _ = extractor(image_path)
                    if message:
                        result["extracted_message"] = message
                except Exception as e:
                    result["extraction_error"] = str(e)

    return result

class StarlightScanner:
    def __init__(self, model_path, num_workers=4, quiet=False):
        self.model_path = model_path
        self.num_workers = num_workers
        if not quiet:
            print(f"[INIT] Model path set to: {model_path}")
            print(f"[INIT] Fast scanner configured (workers={num_workers})")

    def scan_file(self, file_path):
        """Scans a single image file."""
        global MODEL_TYPE, DEVICE
        
        # Initialize model for single file scan
        if self.model_path.endswith('.pth'):
            MODEL_TYPE = 'pytorch'
            # Check for MPS availability on Mac
            if torch.backends.mps.is_available():
                DEVICE = torch.device('mps')
            elif torch.cuda.is_available():
                DEVICE = torch.device('cuda')
            else:
                DEVICE = torch.device('cpu')
            
            # Load PyTorch model
            model = BalancedStarlightDetector()
            model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            session = model
        else:
            MODEL_TYPE = 'onnx'
            DEVICE = None
            session = onnxruntime.InferenceSession(self.model_path)
        
        # Extraction is enabled for single file scans
        return _scan_logic(file_path, session, extract_message=True)

    def scan_directory(self, path, quiet=False):
        # Efficient file discovery with generator to avoid memory issues
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        image_paths = []
        
        if not quiet:
            print(f"\n[SCANNER] Discovering images in {path}...")
        
        # Use os.walk but collect paths efficiently with progress indication
        import time
        start_time = time.time()
        last_report = start_time
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_paths.append(os.path.join(root, file))
                    
                    # Report progress every 2 seconds for large directories
                    current_time = time.time()
                    if not quiet and current_time - last_report > 2:
                        print(f"[SCANNER] Discovered {len(image_paths)} images so far...")
                        last_report = current_time
        
        if not image_paths:
            if not quiet:
                print("[SCANNER] No images found")
            return []
        
        results = []
        
        # Use optimized batching based on model type and device
        # For large datasets, use larger batches to reduce overhead
        if len(image_paths) > 5000:
            if MODEL_TYPE == 'pytorch' and DEVICE and DEVICE.type in ['mps', 'cuda']:
                optimal_batch_size = 50  # Larger batches for big datasets
            else:
                optimal_batch_size = 150
        else:
            if MODEL_TYPE == 'pytorch' and DEVICE and DEVICE.type in ['mps', 'cuda']:
                optimal_batch_size = 25
            else:
                optimal_batch_size = 100
            
        batches = [image_paths[i:i + optimal_batch_size] for i in range(0, len(image_paths), optimal_batch_size)]
        
        if not quiet:
            print(f"[SCANNER] Found {len(image_paths)} images to scan")
            if len(image_paths) > 5000:
                print(f"[SCANNER] Large dataset detected - using optimized batch size: {optimal_batch_size}")
            elif MODEL_TYPE == 'pytorch' and DEVICE and DEVICE.type in ['mps', 'cuda']:
                print(f"[SCANNER] Using GPU/MPS optimized batching ({optimal_batch_size} images per batch)")
            print(f"[SCANNER] Initializing {self.num_workers} parallel workers...")
            print(f"[SCANNER] Estimated time: ~{len(image_paths) / 20:.0f} seconds (based on ~20 images/sec)")
        
        if not quiet:
            print(f"[SCANNER] Processing {len(batches)} batches (~{optimal_batch_size} images per batch)")
        
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=init_worker, initargs=(self.model_path,)) as executor:
            progress_bar = tqdm(total=len(image_paths), desc="Scanning images", disable=quiet)
            with progress_bar:
                # Submit all batches at once for better throughput
                futures = [executor.submit(scan_batch_worker, batch) for batch in batches]
                
                for future in as_completed(futures):
                    try:
                        batch_results = future.result(timeout=60)  # 60 second timeout per batch
                        for res in batch_results:
                            if 'error' in res and not quiet:
                                print(f"Warning: {res['error']}")
                            results.append(res)
                        progress_bar.update(len(batch_results))
                    except TimeoutError:
                        if not quiet:
                            print(f"Warning: Batch processing timed out")
                        progress_bar.update(optimal_batch_size)  # Update progress even on timeout
        
        return results

def main():
    global NO_HEURISTICS
    parser = argparse.ArgumentParser(description="Scan a directory or a single file for steganography.")
    parser.add_argument("path", help="The directory or file to scan.")
    parser.add_argument("--model", default="models/detector_balanced.onnx", help="Path to the ONNX or PyTorch model file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for scanning.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument("--no-heuristics", action="store_true", help="Disable post-processing heuristics and special cases for benchmarking.")
    args = parser.parse_args()
    NO_HEURISTICS = args.no_heuristics
    
    # Auto-detect and use PyTorch model if available on Mac
    if args.model.endswith('.onnx') and torch.backends.mps.is_available():
        pth_model = args.model.replace('.onnx', '.pth')
        if os.path.exists(pth_model):
            args.model = pth_model
            if not args.json:
                print(f"[AUTO] Using PyTorch model with MPS acceleration: {args.model}")
    
    # Initialize global variables for model type and device
    global MODEL_TYPE, DEVICE
    if args.model.endswith('.pth'):
        MODEL_TYPE = 'pytorch'
        if torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
            if not args.json:
                print(f"[DEVICE] Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            if not args.json:
                print(f"[DEVICE] Using CUDA for acceleration")
        else:
            DEVICE = torch.device('cpu')
            if not args.json:
                print(f"[DEVICE] Using CPU for inference")
    else:
        MODEL_TYPE = 'onnx'
        DEVICE = None
        if not args.json:
            print(f"[DEVICE] Using ONNX Runtime for inference")

    scanner = StarlightScanner(args.model, num_workers=args.workers, quiet=args.json)
    start_time = time.time()
    
    if os.path.isfile(args.path):
        results = [scanner.scan_file(args.path)]
    else:
        results = scanner.scan_directory(args.path, quiet=args.json)
        
    end_time = time.time()

    if args.json:
        print(json.dumps(results, indent=2))
        return

    detected_files = [r for r in results if r.get("is_stego")]
    clean_files = [r for r in results if not r.get("is_stego") and "error" not in r]
    errors = [r for r in results if "error" in r]

    print("\n" + "="*60)
    print("SCAN COMPLETE")
    print("="*60)
    total_scanned = len(results) - len(errors)
    print(f"Total images scanned: {total_scanned}")
    print(f"Steganography detected: {len(detected_files)}")

    if detected_files:
        print("\nDetected files:")
        for i, r in enumerate(detected_files):
            if i < 20:
                message_info = f", Message: '{r['extracted_message'][:50]}...'" if r.get('extracted_message') else ""
                print(f"  - {os.path.basename(r['file_path'])} (Predicted: {r['stego_type']}, Confidence: {r.get('confidence', 0):.1%}{message_info})")
        if len(detected_files) > 20:
            print(f"  ... and {len(detected_files) - 20} more.")

    print(f"\nClean images: {len(clean_files)}")
    if errors:
        print(f"Errors on {len(errors)} images.")

    elapsed_time = end_time - start_time
    images_per_sec = total_scanned / elapsed_time if elapsed_time > 0 else 0
    print(f"Scan time: {elapsed_time:.2f} seconds ({images_per_sec:.1f} images/sec)")

    if detected_files:
        print("\nSteganography types found:")
        type_counts = {}
        for r in detected_files:
            stype = r['stego_type']
            type_counts[stype] = type_counts.get(stype, 0) + 1
        for stype, count in type_counts.items():
            print(f"  {stype}: {count}")

if __name__ == "__main__":
    main()
