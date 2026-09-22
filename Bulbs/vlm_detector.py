import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class VLMBulbDetector:
    def __init__(self, model_id="google/owlvit-base-patch32"):
        print(f"Loading Open-Vocabulary VLM ({model_id}) for CPU execution...")
        self.device = "cpu"
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("VLM loaded successfully.")

    def detect_bulbs(self, image_path, score_threshold=0.10):
        """
        Detects light fixtures using open-vocabulary text queries.
        Returns coordinates in absolute pixel scale: [[x1, y1, x2, y2], ...]
        """
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Open-vocabulary text queries
        text_queries = [["a light bulb", "a ceiling lamp", "a hanging light fixture", "a wall lamp"]]

        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Target image sizes tensor for rescaling normalized boxes to pixel coordinates
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold
        )[0]

        boxes = []
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = box.tolist()
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return boxes