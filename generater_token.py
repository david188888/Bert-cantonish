tokens = ["香港", "原本", "有", "一個", "人煙", "稀少", "嘅", "漁港"]

# Simulating confidence scores (on a scale from 0 to 1)
import random
random.seed(42)  # For reproducibility
confidence_scores = [random.uniform(0.5, 1.0) for _ in tokens]

print("Confidence scores:", confidence_scores)