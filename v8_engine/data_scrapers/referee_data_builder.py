import pandas as pd
import numpy as np
import random

def generate_referee_data():
    print("Generating simulated referee strictness profiles...")
    
    # Mocking referees for the 2026 World Cup
    referees = [
        {"ref_id": 1, "name": "Mateu Lahoz (Mock)", "cards_per_game": 5.8, "penalties_per_game": 0.45, "strictness_index": 0.95},
        {"ref_id": 2, "name": "Michael Oliver", "cards_per_game": 3.2, "penalties_per_game": 0.20, "strictness_index": 0.40},
        {"ref_id": 3, "name": "Wilton Sampaio", "cards_per_game": 4.5, "penalties_per_game": 0.35, "strictness_index": 0.70},
        {"ref_id": 4, "name": "Daniele Orsato", "cards_per_game": 4.1, "penalties_per_game": 0.25, "strictness_index": 0.60},
        {"ref_id": 5, "name": "Szymon Marciniak", "cards_per_game": 3.8, "penalties_per_game": 0.30, "strictness_index": 0.55}
    ]
    
    df = pd.DataFrame(referees)
    df.to_csv("../referee_stats.csv", index=False)
    print("Generated referee_stats.csv")

if __name__ == "__main__":
    generate_referee_data()
