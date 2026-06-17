import os
import pandas as pd
import json
import numpy as np
import hashlib
from scipy.stats import poisson
from functools import lru_cache

# Centralized translation dictionary
zh_translation = {
    'Mexico': '墨西哥', 'South Africa': '南非', 'South Korea': '韩国', 'Czech Republic': '捷克',
    'Canada': '加拿大', 'Bosnia and Herzegovina': '波黑', 'USA': '美国', 'Paraguay': '巴拉圭',
    'Germany': '德国', 'Curaçao': '库拉索', 'Ivory Coast': '科特迪瓦', 'Ecuador': '厄瓜多尔',
    'Netherlands': '荷兰', 'Japan': '日本', 'Sweden': '瑞典', 'Tunisia': '突尼斯',
    'Qatar': '卡塔尔', 'Switzerland': '瑞士', 'Brazil': '巴西', 'Morocco': '摩洛哥',
    'Haiti': '海地', 'Scotland': '苏格兰', 'Australia': '澳大利亚', 'Turkey': '土耳其',
    'Belgium': '比利时', 'Egypt': '埃及', 'Iran': '伊朗', 'New Zealand': '新西兰',
    'Spain': '西班牙', 'Cape Verde': '佛得角', 'Saudi Arabia': '沙特阿拉伯', 'Uruguay': '乌拉圭',
    'Argentina': '阿根廷', 'Portugal': '葡萄牙', 'Croatia': '克罗地亚', 'Senegal': '塞内加尔',
    'France': '法国', 'England': '英格兰', 'Korea Republic': '韩国', 'United States': '美国'
}

def get_zh_name(en_name):
    return zh_translation.get(en_name, en_name)

@lru_cache(maxsize=1)
def load_v10_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'v10_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tier(elo):
    if elo >= 1900: return "T0-世界顶尖"
    elif elo >= 1750: return "T1-洲际强队"
    elif elo >= 1600: return "T2-中坚力量"
    elif elo >= 1450: return "T3-边缘球队"
    else: return "T4-送分鱼腩"

def dixon_coles_prob(l1, l2, k1, k2, rho=0.0):
    """
    Compute Poisson probability for score (k1, k2) with Dixon-Coles adjustment
    """
    prob = poisson.pmf(k1, l1) * poisson.pmf(k2, l2)
    if k1 == 0 and k2 == 0:
        prob *= max(0.0, 1 - l1 * l2 * rho)
    elif k1 == 0 and k2 == 1:
        prob *= max(0.0, 1 + l1 * rho)
    elif k1 == 1 and k2 == 0:
        prob *= max(0.0, 1 + l2 * rho)
    elif k1 == 1 and k2 == 1:
        prob *= max(0.0, 1 - rho)
    return prob

@lru_cache(maxsize=1)
def load_results_csv():
    """Load results.csv only once and cache it in memory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '../results.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@lru_cache(maxsize=1)
def get_cached_models(cutoff_date="2026-06-11"):
    """Train models only once per session and cache them."""
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
        
    from core_model import train_v8_models
    
    df = load_results_csv()
    train_df = df[df['date'] < cutoff_date].copy() # Train on data before cutoff
    
    squad_vals = pd.read_csv(os.path.join(base_dir, 'data_scrapers/squad_values.csv'))
    tac_df = pd.read_csv(os.path.join(base_dir, 'data_scrapers/tactical_styles.csv'))
    squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
    tac_dict = tac_df.set_index('team').to_dict('index')
    
    print("[V10 Shared] Training models... (This should only happen once per session)")
    models = train_v8_models(train_df, squad_dict, tac_dict)
    return models

def get_assigned_referee(t1, t2, date_str, referees_list):
    # Use MD5 hash of match details to stably pick a referee from the pool
    seed_str = f'{t1}_{t2}_{date_str}'
    hash_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    return referees_list[hash_int % len(referees_list)]


name_mapping = {'South Korea': 'South Korea', 'Korea Republic': 'South Korea', 'USA': 'USA', 'United States': 'USA'}
def map_name(name):
    return name_mapping.get(name, name)
