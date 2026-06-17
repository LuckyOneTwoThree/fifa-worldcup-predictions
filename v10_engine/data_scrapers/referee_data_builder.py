import pandas as pd
import numpy as np
import os

def generate_referee_data():
    print("Generating comprehensive realistic referee profiles...")
    
    # 20+ realistic top international referees for World Cup
    referees = [
        {"ref_id": 1, "name": "Daniele Orsato", "cards_per_game": 5.1, "penalties_per_game": 0.25, "strictness_index": 0.75},
        {"ref_id": 2, "name": "Szymon Marciniak", "cards_per_game": 4.2, "penalties_per_game": 0.35, "strictness_index": 0.60},
        {"ref_id": 3, "name": "Anthony Taylor", "cards_per_game": 3.8, "penalties_per_game": 0.22, "strictness_index": 0.50},
        {"ref_id": 4, "name": "Michael Oliver", "cards_per_game": 3.4, "penalties_per_game": 0.30, "strictness_index": 0.45},
        {"ref_id": 5, "name": "Clement Turpin", "cards_per_game": 3.1, "penalties_per_game": 0.28, "strictness_index": 0.40},
        {"ref_id": 6, "name": "Danny Makkelie", "cards_per_game": 3.3, "penalties_per_game": 0.26, "strictness_index": 0.42},
        {"ref_id": 7, "name": "Wilton Sampaio", "cards_per_game": 5.5, "penalties_per_game": 0.38, "strictness_index": 0.85},
        {"ref_id": 8, "name": "Fernando Rapallini", "cards_per_game": 5.2, "penalties_per_game": 0.32, "strictness_index": 0.80},
        {"ref_id": 9, "name": "Facundo Tello", "cards_per_game": 5.4, "penalties_per_game": 0.34, "strictness_index": 0.82},
        {"ref_id": 10, "name": "Cesar Ramos", "cards_per_game": 4.0, "penalties_per_game": 0.20, "strictness_index": 0.55},
        {"ref_id": 11, "name": "Ismail Elfath", "cards_per_game": 4.1, "penalties_per_game": 0.33, "strictness_index": 0.58},
        {"ref_id": 12, "name": "Ivan Barton", "cards_per_game": 4.3, "penalties_per_game": 0.35, "strictness_index": 0.62},
        {"ref_id": 13, "name": "Slavko Vincic", "cards_per_game": 3.9, "penalties_per_game": 0.21, "strictness_index": 0.52},
        {"ref_id": 14, "name": "Victor Gomes", "cards_per_game": 3.7, "penalties_per_game": 0.29, "strictness_index": 0.48},
        {"ref_id": 15, "name": "Mustapha Ghorbal", "cards_per_game": 4.2, "penalties_per_game": 0.27, "strictness_index": 0.61},
        {"ref_id": 16, "name": "Abdulrahman Al-Jassim", "cards_per_game": 3.5, "penalties_per_game": 0.25, "strictness_index": 0.45},
        {"ref_id": 17, "name": "Chris Beath", "cards_per_game": 3.6, "penalties_per_game": 0.24, "strictness_index": 0.47},
        {"ref_id": 18, "name": "Jesus Valenzuela", "cards_per_game": 5.0, "penalties_per_game": 0.31, "strictness_index": 0.72},
        {"ref_id": 19, "name": "Andres Matonte", "cards_per_game": 4.9, "penalties_per_game": 0.28, "strictness_index": 0.70},
        {"ref_id": 20, "name": "Alireza Faghani", "cards_per_game": 4.4, "penalties_per_game": 0.26, "strictness_index": 0.65}
    ]
    
    df = pd.DataFrame(referees)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(base_dir, "referee_stats.csv"), index=False)
    print("Generated referee_stats.csv with 20 realistic referees")

if __name__ == "__main__":
    generate_referee_data()
