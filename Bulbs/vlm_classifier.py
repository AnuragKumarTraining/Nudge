import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class VLMClassifier:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        print(f"Loading lightweight VLM classifier ({model_id}) on CPU...")
        self.device = "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # Labels for zero-shot candidate validation
        self.candidate_labels = [
            "a photo of a glowing light bulb or lamp fixture",
            "a photo of a glare, reflection, screen, or bright wall surface"
        ]
        print("VLM classifier ready.")

    def verify_crop(self, crop_bgr):
        """
        Takes an OpenCV BGR crop image and determines if it is a true bulb/lamp.
        Returns: (is_bulb: bool, confidence: float, predicted_label: str)
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return False, 0.0, "invalid_crop"

        # Convert OpenCV BGR to RGB PIL Image
        rgb_img = Image.fromarray(crop_bgr[:, :, ::-1])

        inputs = self.processor(
            text=self.candidate_labels,
            images=rgb_img,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits per image -> softmax probabilities
            probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

        bulb_conf = probs[0]
        glare_conf = probs[1]
        
        is_bulb = bulb_conf > glare_conf
        pred_label = "bulb" if is_bulb else "glare/artifact"

        return is_bulb, bulb_conf, pred_label