import os
import pandas as pd
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
def load_results_csv():
    """Load results.csv only once and cache it in memory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '../results.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@lru_cache(maxsize=1)
def get_cached_models():
    """Train models only once per session and cache them."""
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
        
    from core_model import train_v8_models
    
    df = load_results_csv()
    train_df = df[df['date'] < '2026-06-11'].copy() # Train on data before WC 2026
    
    squad_vals = pd.read_csv(os.path.join(base_dir, 'data_scrapers/squad_values.csv'))
    tac_df = pd.read_csv(os.path.join(base_dir, 'data_scrapers/tactical_styles.csv'))
    squad_dict = dict(zip(squad_vals['team'], squad_vals['squad_value_m']))
    tac_dict = tac_df.set_index('team').to_dict('index')
    
    print("[V8 Shared] Training models... (This should only happen once per session)")
    models = train_v8_models(train_df, squad_dict, tac_dict)
    return models