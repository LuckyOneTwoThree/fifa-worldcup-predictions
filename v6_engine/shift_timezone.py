import os
import codecs

pdata_dir = r"d:\HelloWorld\Git_Project\worldcup\pdata"

# The original files
p_13 = os.path.join(pdata_dir, "2026-06-13_prediction.md")
r_13 = os.path.join(pdata_dir, "2026-06-13_review.md")
p_14 = os.path.join(pdata_dir, "2026-06-14_prediction.md")

# We will rename them to Beijing time (+1 day)
new_p_14 = os.path.join(pdata_dir, "2026-06-14_prediction.md") # The old 13th becomes 14th
new_r_14 = os.path.join(pdata_dir, "2026-06-14_review.md")     # The old 13th review becomes 14th review
new_p_15 = os.path.join(pdata_dir, "2026-06-15_prediction.md") # The old 14th becomes 15th

# Read old contents
try:
    with codecs.open(p_13, 'r', 'utf-8') as f: c_p_13 = f.read()
    with codecs.open(r_13, 'r', 'utf-8') as f: c_r_13 = f.read()
except FileNotFoundError:
    # already renamed perhaps? Let's just handle it
    pass

try:
    with codecs.open(p_14, 'r', 'utf-8') as f: c_p_14 = f.read()
except FileNotFoundError:
    pass

# We will just write new files with updated headers
if 'c_p_13' in locals():
    # Replace header for 13th prediction -> 14th Beijing Time
    new_content = c_p_13.replace("# 📅 日期：2026-06-13 (赛前预测)", "# 📅 北京时间：2026-06-14 (美洲当地 6-13)\n## 📊 赛前量化预测")
    # if not replaced because of missing, just prepend
    if new_content == c_p_13:
        new_content = "# 📅 北京时间：2026-06-14 (美洲当地 6-13)\n## 📊 赛前量化预测\n\n" + c_p_13
    with codecs.open(os.path.join(pdata_dir, "temp_14_pred.md"), 'w', 'utf-8') as f:
        f.write(new_content)

if 'c_r_13' in locals():
    new_content = c_r_13.replace("# 📅 日期：2026-06-13 (赛后量化复盘)", "# 📅 北京时间：2026-06-14 (美洲当地 6-13)\n## 📈 赛后量化复盘")
    if new_content == c_r_13:
        new_content = "# 📅 北京时间：2026-06-14 (美洲当地 6-13)\n## 📈 赛后量化复盘\n\n" + c_r_13
    with codecs.open(new_r_14, 'w', 'utf-8') as f:
        f.write(new_content)

if 'c_p_14' in locals():
    new_content = c_p_14.replace("# 📅 日期：2026-06-14 (赛前预测)", "# 📅 北京时间：2026-06-15 (美洲当地 6-14)\n## 📊 赛前量化预测")
    if new_content == c_p_14:
        new_content = "# 📅 北京时间：2026-06-15 (美洲当地 6-14)\n## 📊 赛前量化预测\n\n" + c_p_14
    with codecs.open(new_p_15, 'w', 'utf-8') as f:
        f.write(new_content)

# Cleanup old files
try: os.remove(p_13)
except: pass
try: os.remove(r_13)
except: pass
# Note: we wrote p_13 to temp_14_pred to avoid overwriting p_14 before reading it
try: os.rename(os.path.join(pdata_dir, "temp_14_pred.md"), new_p_14)
except: pass

print("Timezone shift applied to all reports.")
