import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# Limit to physical cores to prevent thread thrashing
torch.set_num_threads(os.cpu_count() or 4)

class VLMClassifier:
    def __init__(self, model_id="openai/clip-vit-large-patch14-336"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "System CPU"
        
        print("\n" + "=" * 60)
        print(f"[VLM INIT] Initializing zero-shot verifier...")
        print(f"  • Model Checkpoint : {model_id}")
        print(f"  • Compute Target   : {self.device.upper()} ({device_name})")
        print("=" * 60)

        print("[1/3] Loading CLIP processor...", end=" ", flush=True)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        print("Done.")

        print("[2/3] Loading Vision Transformer weights into memory...", end=" ", flush=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        
        # Compile model for faster execution on PyTorch 2.0+
        if hasattr(torch, "compile"):
            print("[INFO] Compiling model for faster CPU inference...", end=" ", flush=True)
            self.model = torch.compile(self.model)
            print("Done.")
        else:
            print("Done.")

        print("[3/3] Pre-computing text embeddings for target classes...", end=" ", flush=True)
        self.candidate_labels = [
            # [0] Target Luminaire (Physical active lighting fixtures)
            "a close-up photo of an active glowing light fixture, outdoor garden pathway light, hanging patio lamp, landscape spotlight, or illuminated luminaire",

            # [1] Environmental Specular Reflections, Plant Glare & Gloss
            "a photo of specular reflection on glossy plant leaves, wet brick pathway glare, shiny chrome appliance glare, wet floor reflection, or lens flare",

            # [2] Human Skin, Face & Body Highlights (Eliminates forehead/face false positives)
            "a photo of human skin, a person's face, forehead highlight, oily skin reflection, nose, cheeks, facial hair, or human body",

            # [3] Natural Daylight, Windows & Blinds (Eliminates louver/window false positives)
            "a photo of bright outdoor daylight, a sunlit window, glass louvers, horizontal window blinds, window slats, or outdoor skylight",

            # [4] Architectural Surfaces, Foliage & Inert Hardware
            "a photo of a blank white ceiling, plain drywall, dark night sky, tree branches, unlit foliage, door frame, or electrical switchboard"
        ]

        text_inputs = self.processor(
            text=self.candidate_labels,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            # Process through the text tower and apply the projection layer manually
            text_outputs = self.model.text_model(**text_inputs)
            text_feats = self.model.text_projection(text_outputs.pooler_output)
            self.text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
        print("Done.")
        print("[STATUS] VLM Engine is active and ready for inference.\n")

    def verify_boxes_single_pass(self, frame_bgr, candidates):
        """
        O(1) Vision Pass: Processes the full image once and extracts candidate features 
        by pooling spatial patch tokens corresponding to their bounding boxes.
        """
        if not candidates:
            return []

        total = len(candidates)
        print(f"[VLM INFERENCE] Single-Pass ViT slicing for {total} candidate(s) on {self.device.upper()}...", end=" ", flush=True)

        # 1. Preprocess the full frame
        from PIL import Image
        pil_image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Extract model grid dimensions (e.g., 336 / 14 = 24x24 grid)
        vision_cfg = self.model.config.vision_config
        grid_dim = vision_cfg.image_size // vision_cfg.patch_size

        results = []
        with torch.no_grad():
            # 2. Single forward pass for the entire image
            vision_outputs = self.model.vision_model(**inputs)
            hidden_states = vision_outputs.last_hidden_state[0] # Shape: (seq_len, hidden_dim)
            
            # 3. Reshape the spatial tokens into a 2D grid (excluding the CLS token at index 0)
            patch_tokens = hidden_states[1:].view(grid_dim, grid_dim, -1)
            
            # 4. Map bounding boxes to the token grid and extract features
            for cand in candidates:
                x1, y1, x2, y2 = cand["bbox"]
                
                # Map pixel coordinates to grid coordinates
                px1 = max(0, int((x1 / orig_w) * grid_dim))
                py1 = max(0, int((y1 / orig_h) * grid_dim))
                px2 = min(grid_dim, int((x2 / orig_w) * grid_dim) + 1)
                py2 = min(grid_dim, int((y2 / orig_h) * grid_dim) + 1)
                
                roi_tokens = patch_tokens[py1:py2, px1:px2]
                
                # Average pool the tokens in the bounding box (Fallback to CLS if box is too small)
                if roi_tokens.numel() == 0:
                    pooled_feat = hidden_states[0]
                else:
                    pooled_feat = roi_tokens.mean(dim=(0, 1))
                
                # Project, normalize, and score
                image_feats = self.model.visual_projection(pooled_feat.unsqueeze(0))
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
                
                logit_scale = self.model.logit_scale.exp()
                logits = (image_feats @ self.text_features.T) * logit_scale
                p = logits.softmax(dim=-1).tolist()[0]
                
                bulb_conf, reflection_conf, daylight_conf = p[0], p[1], p[3]
                
                area = (x2 - x1) * (y2 - y1)
                required_conf = 0.40 if area < 2500 else 0.55
                
                is_bulb = (
                    bulb_conf > reflection_conf 
                    and bulb_conf > daylight_conf 
                    and bulb_conf >= required_conf
                )
                
                pred_label = "bulb" if is_bulb else "artifact"
                results.append((is_bulb, bulb_conf, pred_label))

        print("Done.")
        return results