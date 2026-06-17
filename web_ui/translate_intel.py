import json
import os

filepath = 'd:/HelloWorld/Git_Project/worldcup/web_ui/public/latest_intel.json'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Dictionary of translations
team_map = {
    "葡萄牙 vs DR Congo": "葡萄牙 vs 刚果民主共和国",
    "Uzbekistan vs Colombia": "乌兹别克斯坦 vs 哥伦比亚",
    "英格兰 vs 克罗地亚": "英格兰 vs 克罗地亚",
    "Ghana vs Panama": "加纳 vs 巴拿马"
}

news_map = {
    "Cristiano Ronaldo is 'just a football player' in Portugal team, says Roberto Martinez": "马丁内斯: C罗在葡萄牙队中'只是名普通球员'",
    "Roberto Martinez has explained how Cristiano Ronaldo's superstar status changes once he enters Portugal's national team environment. The comment matters because Ronaldo is still one of the biggest names in world football, even at a stage where every ...": "罗伯托·马丁内斯解释了当C罗进入国家队环境后，他的巨星地位会发生怎样的变化。这番评论意义重大，因为即使在现阶段，C罗仍是世界足坛最大的招牌之一...",
    "Portugal Announce Cristiano Ronaldo News Amid Injury Concerns Before USMNT Game": "葡萄牙在对阵美国网友谊赛前宣布关于C罗的伤情更新",
    "Cristiano Ronaldo is not match fit before the March friendly matches for the Portuguese national team.": "C罗在葡萄牙国家队三月份的友谊赛前尚未达到比赛体能状态。",
    "Inside Camp: DR Congo Update": "内部情报：刚果（金）动态",
    "Tactical leak suggests DR Congo might switch to a 3-at-the-back formation for this crucial tie.": "战术泄露表明，刚果（金）可能会在这场关键战役中改用三后卫阵型。",
    "Tactical Radar: DR Congo": "战术雷达：刚果（金）",
    "DR Congo squad seen doing intensive set-piece drills behind closed doors.": "刚果（金）全队正在进行封闭的定位球高强度演练。",
    "Inside Camp: Uzbekistan Update": "内部情报：乌兹别克斯坦动态",
    "Medical staff clears Uzbekistan starting striker, but will likely be limited to 60 minutes.": "医疗团队批准乌兹别克斯坦首发前锋出战，但上场时间可能被限制在60分钟内。",
    "Tactical Radar: Uzbekistan": "战术雷达：乌兹别克斯坦",
    "Tactical leak suggests Uzbekistan might switch to a 3-at-the-back formation for this crucial tie.": "战术泄露表明，乌兹别克斯坦可能会在这场关键战役中改用三后卫阵型。",
    "Inside Camp: Colombia Update": "内部情报：哥伦比亚动态",
    "Captain of Colombia gives rallying speech, team looks highly motivated and united.": "哥伦比亚队长发表动员演讲，全队士气高涨且空前团结。",
    "Tactical Radar: Colombia": "战术雷达：哥伦比亚",
    "Rumors of food poisoning in Colombia camp have been officially denied by the FA.": "关于哥伦比亚阵营食物中毒的传闻已被足协官方辟谣。",
    "Inside Camp: England Update": "内部情报：英格兰动态",
    "Rumors of food poisoning in England camp have been officially denied by the FA.": "关于英格兰阵营食物中毒的传闻已被足协官方辟谣。",
    "Tactical Radar: England": "战术雷达：英格兰",
    "England squad seen doing intensive set-piece drills behind closed doors.": "英格兰全队正在进行封闭的定位球高强度演练。",
    "Inside Camp: Croatia Update": "内部情报：克罗地亚动态",
    "Tactical leak suggests Croatia might switch to a 3-at-the-back formation for this crucial tie.": "战术泄露表明，克罗地亚可能会在这场关键战役中改用三后卫阵型。",
    "Tactical Radar: Croatia": "战术雷达：克罗地亚",
    "Rumors of food poisoning in Croatia camp have been officially denied by the FA.": "关于克罗地亚阵营食物中毒的传闻已被足协官方辟谣。",
    "Inside Camp: Ghana Update": "内部情报：加纳动态",
    "Local media criticizing Ghana's defensive vulnerabilities shown in the previous matches.": "当地媒体猛烈批评加纳在上一场比赛中暴露出的防守漏洞。",
    "Tactical Radar: Ghana": "战术雷达：加纳",
    "Ghana key midfielder missed today's training session with reported hamstring tightness.": "加纳核心中场因腿筋紧绷缺席了今天的训练课。",
    "Inside Camp: Panama Update": "内部情报：巴拿马动态",
    "Rumors of food poisoning in Panama camp have been officially denied by the FA.": "关于巴拿马阵营食物中毒的传闻已被足协官方辟谣。",
    "Tactical Radar: Panama": "战术雷达：巴拿马",
    "Panama squad seen doing intensive set-piece drills behind closed doors.": "巴拿马全队正在进行封闭的定位球高强度演练。"
}

weather_map = {
    "Clear/Partly Cloudy, 25.6°C, Wind 15.3km/h": "晴/多云，25.6°C，风速 15.3km/h",
    "Clear/Partly Cloudy, 16.9°C, Wind 2.2km/h": "晴/多云，16.9°C，风速 2.2km/h",
    "Clear/Partly Cloudy, 26.9°C, Wind 24.1km/h": "晴/多云，26.9°C，风速 24.1km/h",
    "Clear/Partly Cloudy, 16.8°C, Wind 10.1km/h": "晴/多云，16.8°C，风速 10.1km/h"
}

# Translate keys
new_data = {}
for k, v in data.items():
    new_key = team_map.get(k, k)
    
    # Translate weather
    if "impacts" in v:
        w = v["impacts"].get("weather", "")
        v["impacts"]["weather"] = weather_map.get(w, w)
        
        if "_raw_news" in v["impacts"]:
            raw = v["impacts"]["_raw_news"]
            rw = raw.get("weather", "")
            raw["weather"] = weather_map.get(rw, rw)
            
            for list_key in ["home_news", "away_news"]:
                if list_key in raw:
                    for item in raw[list_key]:
                        title = item.get("title", "")
                        body = item.get("body", "")
                        item["title"] = news_map.get(title, title)
                        item["body"] = news_map.get(body, body)
    
    new_data[new_key] = v

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("Translation completed.")
