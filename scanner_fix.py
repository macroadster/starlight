# Quick fix for scanner.py - replace the problematic section around lines 720-760
# This section fixes the circular dependency and broken indentation

def _scan_logic(image_path, session, extract_message=False):
    """The actual scanning logic, independent of the execution context."""
    global METHOD_MAP, MODEL_TYPE, DEVICE

    try:
        # Check if image needs patch-based scanning
        from PIL import Image
        img = Image.open(image_path)
        img_size = img.size  # (width, height)
        
        # If image is larger than 256x256, use patch-based scanning
        if img_size[0] > 256 or img_size[1] > 256:
            print(f"Large image detected ({img_size[0]}x{img_size[1]}), falling back to center crop scanning...")
            # For large images, just use center crop for now to avoid circular dependencies
            # TODO: Implement proper patch-based scanning
            pass

        pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features = load_unified_input(image_path)

        # LSB is returned as HWC (256, 256, 3) from load_unified_input.
        # Permute to CHW (3, 256, 256) to match ONNX model input expectation.
        lsb = lsb.permute(2, 0, 1) # HWC -> CHW

        
        # Create bit_order feature (default to msb-first for inference)
        bit_order = torch.tensor([[0.0, 1.0, 0.0]])  # [lsb-first, msb-first, none]

        
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

        # Add batch dimension for ONNX model
        pixel_tensor = pixel_tensor.unsqueeze(0)
        meta = meta.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        lsb = lsb.unsqueeze(0)
        palette = palette.unsqueeze(0)
        palette_lsb = palette_lsb.unsqueeze(0)
        format_features = format_features.unsqueeze(0)
        content_features = content_features.unsqueeze(0)

        # LSB handling for ONNX model input - it's already CHW, just add batch dim
        lsb_onnx = lsb.unsqueeze(0)
        
        palette = palette.unsqueeze(0)
        palette_lsb = palette_lsb.unsqueeze(0)
        format_features = format_features.unsqueeze(0)
        content_features = content_features.unsqueeze(0)

        # Run inference
        if MODEL_TYPE == 'pytorch':
            # PyTorch inference
            with torch.no_grad():
                pixel_tensor = pixel_tensor.to(DEVICE)
                meta = meta.to(DEVICE)
                alpha = alpha.to(DEVICE)
                lsb = lsb.to(DEVICE)
                palette = palette.to(DEVICE)
                palette_lsb = palette_lsb.to(DEVICE)
                format_features = format_features.to(DEVICE)
                content_features = content_features.to(DEVICE)
                bit_order = bit_order.to(DEVICE)
                
                # Current model expects 8 inputs
                stego_logits, method_logits, method_id, method_probs, embedding = session(pixel_tensor, meta, alpha, lsb, palette, palette_lsb, format_features, content_features, bit_order)

                stego_logits = stego_logits.cpu().numpy()
                method_id = method_id.cpu().numpy()
                method_probs = method_probs.cpu().numpy()
        else:
            # ONNX inference
            model_inputs = {inp.name for inp in session.get_inputs()}
            
            input_feed = {}
            if 'meta' in model_inputs:
                input_feed['meta'] = meta.numpy()
            if 'alpha' in model_inputs:
                input_feed['alpha'] = alpha.numpy()
            if 'lsb' in model_inputs:
                # lsb_onnx is already (1, 3, 256, 256) NCHW
                input_feed['lsb'] = lsb_onnx.numpy()
            if 'palette' in model_inputs:
                input_feed['palette'] = palette.numpy()
            if 'format_features' in model_inputs:
                input_feed['format_features'] = format_features.numpy()
            if 'content_features' in model_inputs:
                input_feed['content_features'] = content_features.numpy()
            if 'pixel_tensor' in model_inputs:
                input_feed['pixel_tensor'] = pixel_tensor.numpy()
            if 'palette_lsb' in model_inputs:
                input_feed['palette_lsb'] = palette_lsb.numpy()
            
            # Legacy inputs handling (for older models if any)
            if 'bit_order' in model_inputs:
                bit_order = torch.tensor([[0.0, 1.0, 0.0]])
                input_feed['bit_order'] = bit_order.numpy()
            if 'pixel' in model_inputs:
                pixel_4ch = torch.cat([pixel_tensor.squeeze(0), alpha.squeeze(0)], dim=0)
                input_feed['pixel'] = pixel_4ch.unsqueeze(0).numpy()
            if 'metadata' in model_inputs:
                input_feed['metadata'] = meta.numpy()
            
            stego_logits, _, method_id, method_probs, _ = session.run(None, input_feed)

        # Process results
        # Use a numerically stable sigmoid to avoid overflow warnings
        logit = stego_logits[0][0]
        if logit >= 0:
            stego_prob = 1 / (1 + np.exp(-logit))
        else:
            # Use the equivalent alternative form to avoid overflow for large negative logits
            stego_prob = np.exp(logit) / (1 + np.exp(logit))

        # When heuristics are disabled, use a simple 0.5 threshold for benchmarking
        threshold = 0.5
        is_stego = stego_prob > threshold
        method_val = int(method_id[0])
        stego_type = METHOD_MAP.get(method_val, "unknown")
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
                        stego_prob = max(stego_prob, 0.99)  # Boost probability to reflect detection
                        method_val = 0
                        stego_type = "alpha"
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
        }

        if is_stego:
            result["stego_type"] = stego_type
            if confidence is not None:
                result["confidence"] = confidence

            if extract_message and extracted_message is None:
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
                            extracted_message = message
                    except Exception as e:
                        extraction_error = str(e)

            if extracted_message:
                result["extracted_message"] = extracted_message
            if extraction_error:
                result["extraction_error"] = extraction_error

        return result

    except Exception as e:

        error_message = f"Could not process {image_path}: {e}"

        return {"file_path": str(image_path), "error": error_message}