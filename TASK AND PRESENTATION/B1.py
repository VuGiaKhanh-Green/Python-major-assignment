import pip
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import os

# Danh sách bảng cần cào
table_links = {
    'Standard Stats': ('https://fbref.com/en/comps/9/stats/Premier-League-Stats', 'stats_standard'),
    'Shooting': ('https://fbref.com/en/comps/9/shooting/Premier-League-Stats', 'stats_shooting'),
    'Passing': ('https://fbref.com/en/comps/9/passing/Premier-League-Stats', 'stats_passing'),
    'Goal and Shot Creation': ('https://fbref.com/en/comps/9/gca/Premier-League-Stats', 'stats_gca'),
    'Defense': ('https://fbref.com/en/comps/9/defense/Premier-League-Stats', 'stats_defense'),
    'Possession': ('https://fbref.com/en/comps/9/possession/Premier-League-Stats', 'stats_possession'),
    'Miscellaneous': ('https://fbref.com/en/comps/9/misc/Premier-League-Stats', 'stats_misc'),
    'Goalkeeping': ('https://fbref.com/en/comps/9/keepers/Premier-League-Stats', 'stats_keeper')
}

# Khởi tạo WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
merged_df = pd.DataFrame()
goalkeeping_df = pd.DataFrame()

for name, (url, table_id) in table_links.items():
    print(f'\n Fetching data: {name}')
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, table_id)))
    except Exception as e:
        print(f'Loading table error {name}: {e}')
        continue

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', id=table_id)

    if not table:
        print(f'Unable to find table {name}')
        continue

    # Đọc bảng
    df = pd.read_html(StringIO(str(table)))[0]

    # Xử lý MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            group = col[0].strip() if col[0].strip() and 'Unnamed' not in col[0] else ''
            subgroup = col[1].strip() if col[1].strip() else col[0].strip()
            col_name = f"{group} {subgroup}" if group and group != subgroup else subgroup
            new_cols.append(col_name.strip())
        df.columns = new_cols
    else:
        df.columns = [col.strip() for col in df.columns]

    # In cột gốc
    print(f"Original column {name}: {list(df.columns)}")

    # Tìm cột 'Player'
    player_col = next((col for col in df.columns if 'player' in col.lower() and 'rk' not in col.lower()), None)
    if not player_col:
        print(f"ERROR: Can't find column 'Player' in {name}")
        continue

    # Làm sạch dữ liệu
    df = df.loc[df[player_col].notna() & (df[player_col] != player_col)].drop_duplicates(subset=player_col)
    df = df.rename(columns={player_col: 'Player'})

    # Chọn cột cần thiết cho từng bảng
    required_cols = []
    if name == 'Standard Stats':
        required_cols = ['Player', 'Nation', 'Squad', 'Pos', 'Age', 'Playing Time Min', 'Playing Time MP' , 'Playing Time Starts',
                        'Performance Gls', 'Performance Ast', 'Performance CrdY', 'Performance CrdR',
                        'Expected xG',  'Expected xAG','Progression PrgC', 'Progression PrgP', 'Progression PrgR',
                        'Per 90 Minutes Gls', 'Per 90 Minutes Ast', 'Per 90 Minutes xG', 'Per 90 Minutes xAG']
    elif name == 'Shooting':
        required_cols = ['Player', 'Standard SoT%', 'Standard SoT/90', 'Standard G/Sh', 'Standard Dist']
    elif name == 'Passing':
        df = df.rename(columns={'1/3': 'Passing 1/3'})
        required_cols = ['Player', 'Total Cmp', 'Total Cmp%', 'Total TotDist', 'Short Cmp', 'Medium Cmp', 'Long Cmp',
                        'KP', 'Passing 1/3', 'PPA', 'CrsPA', 'PrgP']
    elif name == 'Goal and Shot Creation':
        required_cols = ['Player', 'SCA', 'SCA SCA90', 'GCA', 'GCA GCA90']
    elif name == 'Defense':
        required_cols = ['Player', 'Tackles Tkl', 'Tackles TklW', 'Challenges Att', 'Challenges Lost',
                        'Blocks', 'Blocks Sh', 'Blocks Pass', 'Int']
    elif name == 'Possession':
        required_cols = ['Player', 'Touches', 'Touches Def Pen', 'Touches Def 3rd', 'Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen',
                        'Take-Ons Att', 'Take-Ons Succ%', 'Take-Ons Tkld%',
                        'Carries', 'Carries PrgDist', 'Carries PrgC', 'Carries 1/3', 'Carries CPA', 'Carries Mis', 'Carries Dis',
                        'Receiving Rec', 'Receiving PrgR']
    elif name == 'Miscellaneous':
        required_cols = ['Player', 'Performance Fls', 'Performance Fld', 'Performance Off', 'Performance Crs', 'Performance Recov' , 'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%']
    elif name == 'Goalkeeping':
        required_cols = ['Player', 'Performance GA90', 'Performance Save%', 'Performance CS%', 'Penalty Kicks Save%']

    # Lọc cột có sẵn
    selected_cols = [col for col in required_cols if col in df.columns]
    if len(selected_cols) <= 1:  # Chỉ có 'Player'
        print(f"Can't find the required column for {name}, pass!")
        continue
    df = df[selected_cols]

    # In cột đã chọn
    print(f"Selected column {name}: {list(df.columns)}")

    # Lưu dữ liệu
    if name == 'Goalkeeping':
        goalkeeping_df = df.copy()
    else:
        if merged_df.empty:
            merged_df = df
        else:
            df_cols_to_merge = [col for col in df.columns if col != 'Player']
            merged_df = pd.merge(merged_df, df[['Player'] + df_cols_to_merge], on='Player', how='outer')

driver.quit()

# Lọc cầu thủ > 90 phút
min_col = next((col for col in merged_df.columns if 'playing time min' in col.lower()), None)
if min_col:
    merged_df = merged_df[merged_df[min_col].notna()]
    try:
        merged_df[min_col] = merged_df[min_col].str.replace(',', '', regex=False).astype(float)
        merged_df = merged_df[merged_df[min_col] > 90]
        print(f"Found {len(merged_df)} players who played over 90 minutes")
    except Exception as e:
        print(f"Filtering Error 'Playing Time Min': {e}")
else:
    print("ERROR: Can't find column 'Playing Time Min'")

# Xử lý dữ liệu thủ môn
if not goalkeeping_df.empty and not merged_df.empty:
    goalkeeping_df = goalkeeping_df[goalkeeping_df['Player'].isin(merged_df['Player'])]
    pos_col = next((col for col in merged_df.columns if 'pos' in col.lower()), None)
    if pos_col:
        goalkeeping_df['Pos'] = goalkeeping_df['Player'].map(merged_df.set_index('Player')[pos_col])
        for col in goalkeeping_df.columns:
            if col not in ['Player', 'Pos']:
                goalkeeping_df[col] = goalkeeping_df.apply(
                    lambda row: row[col] if pd.notna(row['Pos']) and 'GK' in row['Pos'] else 'N/A', axis=1
                )
        goalkeeping_df.drop(columns='Pos', inplace=True)
        merged_df = pd.merge(merged_df, goalkeeping_df, on='Player', how='left')
    else:
        print("ERROR: Can't find column 'Pos'")


# Cột mong muốn cuối cùng
required_columns = [
    'Player', 'Nation', 'Squad', 'Pos', 'Age', 'Playing Time Min', 'Playing Time MP' , 'Playing Time Starts',
                        'Performance Gls', 'Performance Ast', 'Performance CrdY', 'Performance CrdR',
                        'Expected xG',  'Expected xAG','Progression PrgC', 'Progression PrgP', 'Progression PrgR',
                        'Per 90 Minutes Gls', 'Per 90 Minutes Ast', 'Per 90 Minutes xG', 'Per 90 Minutes xAG',
    'Performance GA90', 'Performance Save%', 'Performance CS%', 'Penalty Kicks Save%',
    'Standard SoT%', 'Standard SoT/90', 'Standard G/Sh', 'Standard Dist',
    'Total Cmp', 'Total Cmp%', 'Total TotDist', 'Short Cmp', 'Medium Cmp', 'Long Cmp',
                        'KP', 'Passing 1/3', 'PPA', 'CrsPA', 'PrgP',
    'SCA', 'SCA SCA90', 'GCA', 'GCA GCA90',
    'Tackles Tkl', 'Tackles TklW', 'Challenges Att', 'Challenges Lost',
    'Blocks', 'Blocks Sh', 'Blocks Pass', 'Int',
    'Touches', 'Touches Def Pen', 'Touches Def 3rd', 'Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen',
    'Take-Ons Att', 'Take-Ons Succ%', 'Take-Ons Tkld%',
    'Carries', 'Carries PrgDist', 'Carries PrgC', 'Carries 1/3', 'Carries CPA', 'Carries Mis', 'Carries Dis',
    'Receiving Rec', 'Receiving PrgR',
    'Performance Fls', 'Performance Fld', 'Performance Off', 'Performance Crs', 'Performance Recov', 'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%'
]

# Lọc cột có sẵn và đảm bảo thứ tự
available_columns = [col for col in required_columns if col in merged_df.columns]
final_columns = []
must_have = ['Player', 'Nation', 'Squad', 'Pos', 'Age']
for col in must_have:
    if col in available_columns:
        final_columns.append(col)
        available_columns.remove(col)
final_columns.extend(available_columns)
merged_df = merged_df[final_columns]
print(f"Final column: {final_columns}")

merged_df.fillna('N/a', inplace=True)

# Xuất file
output_file = 'results.csv'
try:
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'Extracted file {output_file} successfully')
except Exception as e:
    output_file = 'results_backup.csv'
    print(f'Error when saving to results.csv: {e}, try saving to {output_file}')
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'Extracted file {output_file} successfully')