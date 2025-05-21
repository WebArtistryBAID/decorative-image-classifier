from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

# === Load Model ===
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Classifier Function ===
def classify_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
        return "content" if prob > 0.5 else "decorative"
    except Exception as e:
        return f"error: {str(e)}"

# === HTTP Server ===
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        img_path = query.get("image", [None])[0]

        if img_path is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing ?image=path parameter.")
            return

        if not os.path.exists(img_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Image not found.")
            return

        result = classify_image(img_path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(result.encode())

# === Run Server ===
if __name__ == "__main__":
    print("Server listening on http://localhost:59192")
    server = HTTPServer(("localhost", 59192), RequestHandler)
    server.serve_forever()