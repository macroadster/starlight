# import onnxruntime  # Disabled - no Python 3.14 support
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import json
import multiprocessing as mp
from scripts.starlight_extractor import extraction_functions
from scripts.starlight_utils import load_unified_input # Import from trainer.py

# Set multiprocessing start method to avoid CUDA issues
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set




# --- Worker-specific globals ---
SESSION = None
MODEL_TYPE = 'pytorch'  # Only PyTorch supported now
DEVICE = None  # 'cpu', 'cuda', or 'mps'
METHOD_MAP = {0: "alpha", 1: "palette", 2: "lsb.rgb", 3: "exif", 4: "raw"}
BIT_ORDER_MAP = {0: "lsb-first", 1: "msb-first", 2: "none"}
NO_HEURISTICS = False # Global flag to disable heuristics for benchmarking

# Legacy model class for older PyTorch models (REMOVED)
# ONNX support removed due to Python 3.14 compatibility issues

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
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
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

        # Palette stream
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

        # Pixel tensor processing (new stream)
        self.pixel_conv = nn.Sequential(
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

        # Palette LSB processing (new stream)
        self.palette_lsb_conv = nn.Sequential(
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

        # Attention Mechanisms - Enabled for EOI/EXIF Detection
        self.eoi_attention = nn.Sequential(
            nn.Linear(2048, 512),  # Metadata stream features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Attention weight for EOI
            nn.Sigmoid()
        )
        
        self.exif_attention = nn.Sequential(
            nn.Linear(2048, 512),  # Metadata stream features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Attention weight for EXIF
            nn.Sigmoid()
        )

        # Fusion and classification
        # Original fusion dim + 512 for attention features (256 * 2)
        # self.fusion_dim = 128 * 16 + 64 * 8 * 8 + 64 * 8 * 8 + 64 * 8 * 8 + 64 * 8 * 8 + 64 + 16 + 16  # Original: 18528
        self.fusion_dim = 18528 + 512  # Adding attention features

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
        self.bit_order_head = nn.Linear(64, 3) # lsb-first, msb-first, none
        self.embedding_head = nn.Linear(64, 64)

    def forward(self, pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features):
        # Pixel tensor stream (new)
        pixel_tensor = self.pixel_conv(pixel_tensor)
        pixel_tensor = pixel_tensor.reshape(pixel_tensor.size(0), -1)

        # Metadata stream with weighting
        meta_features = meta # Keep original features for attention
        meta = meta.unsqueeze(1)  # Add channel dimension
        meta = self.meta_conv(meta)
        meta = meta.reshape(meta.size(0), -1)
        meta = meta * self.meta_weight  # Apply weighting to reduce dominance

        # Calculate attention scores
        eoi_attn = self.eoi_attention(meta_features)
        exif_attn = self.exif_attention(meta_features)
        
        # Create attention features
        # Expand attention weights to feature dimensions (simple scaling for now)
        eoi_feat = torch.ones(meta.size(0), 256).to(meta.device) * eoi_attn
        exif_feat = torch.ones(meta.size(0), 256).to(meta.device) * exif_attn
        
        attention_features = torch.cat([eoi_feat, exif_feat], dim=1)

        # Alpha stream
        alpha = self.alpha_conv(alpha)
        alpha = alpha.reshape(alpha.size(0), -1)

        # LSB stream - ensure CHW format
        if lsb.dim() == 4 and lsb.shape[1] == 256:  # (N, H, W, C) format
            lsb = lsb.permute(0, 3, 1, 2)  # NHWC -> NCHW
        elif lsb.dim() == 4:  # Already in (N, C, H, W) format
            pass  # Already correct
        elif lsb.dim() == 3 and lsb.shape[0] == 256:  # (H, W, C) format
            lsb = lsb.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> NCHW
        elif lsb.dim() == 3:  # (C, H, W) format
            lsb = lsb.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected LSB tensor shape: {lsb.shape}")
        
        lsb = self.lsb_conv(lsb)
        lsb = lsb.reshape(lsb.size(0), -1)

        # Palette stream
        palette = self.palette_fc(palette)

        # Palette LSB stream (new)
        palette_lsb = self.palette_lsb_conv(palette_lsb)
        palette_lsb = palette_lsb.reshape(palette_lsb.size(0), -1)

        # Format features stream
        format_features = self.format_features_fc(format_features)

        # Content features stream
        content_features = self.content_features_fc(content_features)

        # Fusion of all 8 streams + attention features
        fused = torch.cat([pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features, attention_features], dim=1)
        fused = self.fusion(fused)

        # Split into embedding and classification features
        embedding = fused[:, :128]
        cls_features = fused[:, 128:]

        # Outputs
        stego_logits = self.stego_head(cls_features)
        method_logits = self.method_head(cls_features)
        bit_order_logits = self.bit_order_head(cls_features)
        embedding = self.embedding_head(cls_features)

        # Get method predictions
        method_probs = F.softmax(method_logits, dim=1)
        method_id = torch.argmax(method_probs, dim=1)
        
        # Get bit order predictions
        bit_order_probs = F.softmax(bit_order_logits, dim=1)
        bit_order_id = torch.argmax(bit_order_probs, dim=1)

        return stego_logits, method_logits, bit_order_logits, method_id, method_probs, bit_order_id, embedding

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
        # Use spawn start method to avoid CUDA issues - workers can use GPU safely
        if torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
        elif torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = torch.device('cpu')
        
        # Load PyTorch model with optimizations
        model = BalancedStarlightDetector()
        # Load with strict=False to handle missing keys for new attention layers
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
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
            # model = torch.jit.script(model)  # TorchScript optimization - disabled for dynamic heads
        
        SESSION = model
    else:
        # Fallback to PyTorch
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
        # Load with strict=False to handle missing keys for new attention layers
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        
        SESSION = model

def _scan_logic(image_path, session, extract_message=False):

    """The actual scanning logic, independent of the execution context."""

    global METHOD_MAP, BIT_ORDER_MAP, MODEL_TYPE, DEVICE

    try:
        # Check if image needs patch-based scanning
        from PIL import Image
        img = Image.open(image_path)
        img_size = img.size  # (width, height)
        
        # If image is larger than 256x256, use patch-based scanning
        if img_size[0] > 256 or img_size[1] > 256:
            # For now, center crop large images to avoid dependency issues
            pass

        pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(image_path)

        # LSB is returned as HWC (256, 256, 3) from load_unified_input.
        # Permute to CHW (3, 256, 256) to match model input expectation.
        lsb = lsb.permute(2, 0, 1) # HWC -> CHW

        # Move tensors to GPU/MPS if using PyTorch to avoid CPU bottlenecks
        if MODEL_TYPE == 'pytorch' and DEVICE is not None:
            pixel_tensor = pixel_tensor.to(DEVICE)
            meta = meta.to(DEVICE)
            alpha = alpha.to(DEVICE)
            lsb = lsb.to(DEVICE) 
            palette = palette.to(DEVICE)
            palette_lsb = palette_lsb.to(DEVICE)
            format_features = format_features.to(DEVICE)
            content_features = content_features.to(DEVICE)

        # Add batch dimension
        pixel_tensor = pixel_tensor.unsqueeze(0)
        meta = meta.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        lsb = lsb.unsqueeze(0)
        palette = palette.unsqueeze(0)
        palette_lsb = palette_lsb.unsqueeze(0)
        format_features = format_features.unsqueeze(0)
        content_features = content_features.unsqueeze(0)
        
        # Run inference
        if MODEL_TYPE == 'pytorch':
            # PyTorch inference
            with torch.no_grad():
                # Forward call matching trainer.py enhanced model
                stego_logits, method_logits, bit_order_logits, method_id, method_probs, bit_order_id, embedding = session(
                    pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features
                )
                
                stego_logits = stego_logits.cpu().numpy()
                method_id = method_id.cpu().numpy()
                method_probs = method_probs.cpu().numpy()
                bit_order_id = bit_order_id.cpu().numpy()
        else:
            # Fallback
            with torch.no_grad():
                stego_logits, method_logits, bit_order_logits, method_id, method_probs, bit_order_id, embedding = session(
                    pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features
                )
                
                stego_logits = stego_logits.cpu().numpy()
                method_id = method_id.cpu().numpy()
                method_probs = method_probs.cpu().numpy()
                bit_order_id = bit_order_id.cpu().numpy()

        # Process results
        # Use a numerically stable sigmoid to avoid overflow warnings
        logit = stego_logits[0][0]
        if logit >= 0:
            stego_prob = 1 / (1 + np.exp(-logit))
        else:
            stego_prob = np.exp(logit) / (1 + np.exp(logit))

        # When heuristics are disabled, use a simple 0.5 threshold for benchmarking
        threshold = 0.5
        is_stego = stego_prob > threshold
        method_val = int(method_id[0])
        stego_type = METHOD_MAP.get(method_val, "unknown")
        bit_order_val = int(bit_order_id[0])
        bit_order_type = BIT_ORDER_MAP.get(bit_order_val, "none")
        
        confidence = float(np.max(method_probs)) if method_probs is not None else None
        extracted_message = None
        extraction_error = None

        # Heuristic safety net: if the model says clean but we can decode an AI42 alpha payload, flag it.
        if (not is_stego) and (not NO_HEURISTICS):
            try:
                alpha_extractor = extraction_functions.get("alpha")
                if alpha_extractor:
                    msg, _ = alpha_extractor(image_path)
                    if msg:
                        is_stego = True
                        stego_prob = max(stego_prob, 0.99)
                        method_val = 0
                        stego_type = "alpha"
                        confidence = 1.0
                        if extract_message:
                            extracted_message = msg
            except Exception as e:
                extraction_error = str(e)

        # Heuristic safety net: if model says clean but we can extract EOI payload, flag it.
        if (not is_stego) and (not NO_HEURISTICS):
            try:
                eoi_extractor = extraction_functions.get("eoi")
                if eoi_extractor:
                    msg, _ = eoi_extractor(image_path)
                    if msg:
                        is_stego = True
                        stego_prob = max(stego_prob, 0.95)
                        method_val = 4
                        stego_type = "raw"
                        confidence = 1.0
                        if extract_message:
                            extracted_message = msg
            except Exception as e:
                extraction_error = str(e)

        result = {
            "file_path": str(image_path),
            "is_stego": bool(is_stego),
            "stego_probability": float(stego_prob),
            "method_id": method_val,
            "bit_order": bit_order_type
        }

        if is_stego:
            result["stego_type"] = stego_type
            if confidence is not None:
                result["confidence"] = confidence

        if extract_message and extracted_message is None:
            try:
                extracted_message = _extract_message(image_path, stego_type, bit_order_type)
            except Exception as e:
                extraction_error = str(e)

        if extracted_message:
            result["extracted_message"] = extracted_message
            # Only override clean classification if extracted message is substantial
            meaningful_message = (
                len(extracted_message.strip()) > 3 and
                len(set(extracted_message.strip())) > 2 and
                not extracted_message.strip().startswith('\x00') and
                not extracted_message.strip().isspace() and
                not extracted_message.strip().endswith('?')
            )
            
            if not is_stego and meaningful_message:
                result["is_stego"] = True
                result["stego_probability"] = max(result["stego_probability"], 0.99)
                result["stego_type"] = stego_type
                if confidence is not None:
                    result["confidence"] = confidence
        if extraction_error:
            result["extraction_error"] = extraction_error

        return result

    except Exception as e:
        error_message = f"Could not process {image_path}: {e}"
        return {"file_path": str(image_path), "error": error_message}

def scan_image_worker(image_path):
    """The actual scanning logic that runs in each worker process."""
    global SESSION
    return _scan_logic(image_path, SESSION, extract_message=False)

def scan_batch_worker(image_paths):
    """Process a batch of images for better GPU utilization."""
    global SESSION, MODEL_TYPE, DEVICE
    results = []
    
    for path in image_paths:
        try:
            result = _scan_logic(path, SESSION, extract_message=False)
            results.append(result)
        except Exception as e:
            error_msg = str(e)
            if "cannot identify image file" in error_msg:
                continue
            results.append({"file_path": path, "error": f"Could not process {path}: {error_msg}"})
    
    return results

def _extract_message(image_path, stego_type, predicted_bit_order="none"):
    extractor_method_name = stego_type
    if stego_type == "lsb.rgb":
        extractor_method_name = "lsb"
    elif stego_type == "raw":
        extractor_method_name = "eoi"

    # Prioritize predicted method
    candidates = []
    if extractor_method_name:
        candidates.append(extractor_method_name)
    
    # Fallbacks
    for fallback in ("lsb", "alpha", "exif", "eoi", "palette"):
        if fallback not in candidates:
            candidates.append(fallback)

    for method in candidates:
        extractor = extraction_functions.get(method)
        if extractor is None:
            continue
            
        # Try with predicted bit order if it matches the method's capabilities
        # (currently only implemented in starlight_extractor.py functions if we update them, 
        # but most extractors there try multiple internal strategies)
        message, _ = extractor(image_path)
        if message:
            return message
    return None

def _process_inference_result(image_path, stego_logits, method_id, method_probs, bit_order_id, extract_message=False):
    """Process inference results into final result format."""
    global METHOD_MAP, BIT_ORDER_MAP
    
    # Use a numerically stable sigmoid to avoid overflow warnings
    logit = stego_logits[0]
    if logit >= 0:
        stego_prob = 1 / (1 + np.exp(-logit))
    else:
        stego_prob = np.exp(logit) / (1 + np.exp(logit))

    # Method-specific thresholds
    thresholds = {0: 0.7, 1: 0.98, 2: 0.95, 3: 0.5, 4: 0.65}
    threshold = thresholds.get(method_id[0], 0.8)
    is_stego = stego_prob > threshold
    
    bit_order_type = BIT_ORDER_MAP.get(bit_order_id[0], "none")
    
    result = {
        "file_path": str(image_path),
        "is_stego": bool(is_stego),
        "stego_probability": float(stego_prob),
        "method_id": int(method_id[0]),
        "bit_order": bit_order_type
    }

    stego_type = METHOD_MAP.get(method_id[0], "unknown")
    if is_stego:
        result["stego_type"] = stego_type
        result["confidence"] = float(np.max(method_probs))

    if extract_message:
        try:
            message = _extract_message(image_path, stego_type, bit_order_type)
            if message:
                result["extracted_message"] = message
                if not is_stego:
                    result["is_stego"] = True
                    result["stego_probability"] = max(result["stego_probability"], 0.99)
                    result["stego_type"] = stego_type
                    result["confidence"] = float(np.max(method_probs))
        except Exception as e:
            result["extraction_error"] = str(e)

    return result

class StarlightScanner:
    def __init__(self, model_path, num_workers=4, quiet=False):
        self.model_path = model_path
        self.num_workers = num_workers
        if not quiet:
            print(f"[INIT] Model path set to: {self.model_path}")
            print(f"[INIT] Fast scanner configured (workers={self.num_workers})")

    def scan_file(self, file_path):
        """Scans a single image file."""
        global MODEL_TYPE, DEVICE
        
        # Initialize model for single file scan
        if self.model_path.endswith('.pth'):
            MODEL_TYPE = 'pytorch'
            if torch.backends.mps.is_available():
                DEVICE = torch.device('mps')
            elif torch.cuda.is_available():
                DEVICE = torch.device('cuda')
            else:
                DEVICE = torch.device('cpu')
            
            model = BalancedStarlightDetector()
            state_dict = torch.load(self.model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            session = model
        else:
            # Fallback
            MODEL_TYPE = 'pytorch'
            if torch.backends.mps.is_available():
                DEVICE = torch.device('mps')
            elif torch.cuda.is_available():
                DEVICE = torch.device('cuda')
            else:
                DEVICE = torch.device('cpu')
            
            model = BalancedStarlightDetector()
            state_dict = torch.load(self.model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            session = model
        
        return _scan_logic(file_path, session, extract_message=True)

    def scan_directory(self, path, quiet=False):
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        image_paths = []
        
        if not quiet:
            print(f"\n[SCANNER] Discovering images in {path}...")
        
        import time
        start_time = time.time()
        last_report = start_time
        
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_paths.append(os.path.join(root, file))
                    
                    current_time = time.time()
                    if not quiet and current_time - last_report > 2:
                        print(f"[SCANNER] Discovered {len(image_paths)} images so far...")
                        last_report = current_time
        
        if not image_paths:
            if not quiet:
                print("[SCANNER] No images found")
            return []
        
        results = []
        
        if len(image_paths) > 5000:
            optimal_batch_size = 150
        else:
            optimal_batch_size = 100
            
        batches = [image_paths[i:i + optimal_batch_size] for i in range(0, len(image_paths), optimal_batch_size)]
        
        if not quiet:
            print(f"[SCANNER] Found {len(image_paths)} images to scan")
            print(f"[SCANNER] Initializing {self.num_workers} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=init_worker, initargs=(self.model_path,)) as executor:
            progress_bar = tqdm(total=len(image_paths), desc="Scanning images", disable=quiet)
            with progress_bar:
                futures = [executor.submit(scan_batch_worker, batch) for batch in batches]
                
                for future in as_completed(futures):
                    try:
                        batch_results = future.result(timeout=60)
                        for res in batch_results:
                            results.append(res)
                        progress_bar.update(len(batch_results))
                    except TimeoutError:
                        progress_bar.update(optimal_batch_size)
        
        return results

def main():
    global NO_HEURISTICS
    parser = argparse.ArgumentParser(description="Scan a directory or a single file for steganography.")
    parser.add_argument("path", help="The directory or file to scan.")
    parser.add_argument("--model", default="models/detector_v3_auto_attn.pth", help="Path to PyTorch model file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers for scanning.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument("--no-heuristics", action="store_true", help="Disable post-processing heuristics and special cases for benchmarking.")
    args = parser.parse_args()
    NO_HEURISTICS = args.no_heuristics
    
    global MODEL_TYPE, DEVICE
    MODEL_TYPE = 'pytorch'
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

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
                print(f"  - {os.path.basename(r['file_path'])} (Type: {r['stego_type']}, Order: {r['bit_order']}, Confidence: {r.get('confidence', 0):.1%}{message_info})")
        if len(detected_files) > 20:
            print(f"  ... and {len(detected_files) - 20} more.")

    elapsed_time = end_time - start_time
    images_per_sec = total_scanned / elapsed_time if elapsed_time > 0 else 0
    print(f"\nScan time: {elapsed_time:.2f} seconds ({images_per_sec:.1f} images/sec)")

if __name__ == "__main__":
    main()
