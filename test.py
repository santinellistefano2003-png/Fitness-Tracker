import os
import pandas as pd
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # ← backend non interattivo, necessario su server
# ── URLs grafici ──────────────────────────────────────────────────────────────
pie     = "https://raw.githubusercontent.com/santinellistefano2003-png/Fitness-Tracker/refs/heads/main/assets/volume_per_type.svg"
heatmap = "https://raw.githubusercontent.com/santinellistefano2003-png/Fitness-Tracker/refs/heads/main/assets/volume_giornaliero.svg"
hist    = "https://raw.githubusercontent.com/santinellistefano2003-png/Fitness-Tracker/refs/heads/main/assets/volume_muscle_group.svg"

# ── Variabili d'ambiente ──────────────────────────────────────────────────────
load_dotenv(override=True)
token   = os.getenv("NOTION_TOKEN").strip()
Musc    = os.getenv("Musc").strip()
Wei     = os.getenv("Wei").strip()
Exce    = os.getenv("Exce").strip()
PAGE_ID = os.getenv("PAGE_ID").strip()

HEADERS = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json"
}

# ── Funzioni Notion → DataFrame ───────────────────────────────────────────────
def extract_property(prop: dict, relation_map: dict = {}):
    ptype = prop["type"]
    extractors = {
        "title":        lambda p: p["title"][0]["plain_text"] if p["title"] else None,
        "rich_text":    lambda p: p["rich_text"][0]["plain_text"] if p["rich_text"] else None,
        "number":       lambda p: p["number"],
        "select":       lambda p: p["select"]["name"] if p["select"] else None,
        "multi_select": lambda p: [s["name"] for s in p["multi_select"]],
        "date":         lambda p: p["date"]["start"] if p["date"] else None,
        "checkbox":     lambda p: p["checkbox"],
        "url":          lambda p: p["url"],
        "formula":      lambda p: p["formula"].get("number") or p["formula"].get("string"),
        "relation":     lambda p: [relation_map.get(r["id"], r["id"]) for r in p["relation"]] if p["relation"] else None,
    }
    return extractors.get(ptype, lambda p: None)(prop)


def notion_datasource_to_df(data_source_id: str, relation_maps: dict = {}) -> pd.DataFrame:
    url = f"https://api.notion.com/v1/data_sources/{data_source_id}/query"
    rows = []
    payload = {}
    while True:
        response = requests.post(url, headers=HEADERS, json=payload)
        data = response.json()
        for page in data["results"]:
            row = {"page_id": page["id"]}
            for col_name, prop in page["properties"].items():
                rel_map = relation_maps.get(col_name, {})
                row[col_name] = extract_property(prop, rel_map)
            rows.append(row)
        if data["has_more"]:
            payload["start_cursor"] = data["next_cursor"]
        else:
            break
    return pd.DataFrame(rows)

# ── Caricamento dati ──────────────────────────────────────────────────────────
muscles    = notion_datasource_to_df(Musc)
excercices = notion_datasource_to_df(Exce)
weight     = notion_datasource_to_df(Wei)
weight.drop(columns=['page_id', '0'], inplace=True)

# ── Merge e pulizia ───────────────────────────────────────────────────────────
excercices['Exercise'] = excercices['Exercise'].apply(lambda x: x[0] if isinstance(x, list) else x)
df = pd.merge(excercices, muscles, left_on='Exercise', right_on='page_id', how='left')
df.drop(columns=['page_id_y', 'page_id_x', 'Exercise_x'], inplace=True)
df.rename(columns={'Exercise_y': 'Exercise'}, inplace=True)

if 'Type' in df.columns:
    df.drop(columns=['Type'], inplace=True)
if 'Date1' in df.columns:
    df.rename(columns={'Date': 'Split', 'Date1': 'Date'}, inplace=True)
if 'Exercise' in df.columns:
    df['Exercise'] = df['Exercise'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
if 'Date' in df.columns:
    df['Date'] = df['Date'].apply(lambda x: x.split('T')[0] if isinstance(x, str) else x)
if 'Split' in df.columns:
    df['Split'] = df['Split'].str.strip()

# ── Volume ────────────────────────────────────────────────────────────────────
weight['Date'] = pd.to_datetime(weight['Date'])
df['Date']     = pd.to_datetime(df['Date'])

def get_body_weight(date):
    weights_before_date = weight[weight['Date'] <= date]
    return weights_before_date.iloc[-1]['Weight'] if not weights_before_date.empty else None

def adjust_weight_for_bodyweight(exercise, w, date):
    body_weight_exercises = ['Dips', 'Chin Up', 'Pull Up', 'Hammer Pullups', 'Push Up', 'Leg Raise']
    if exercise in body_weight_exercises:
        body_weight = get_body_weight(date)
        if body_weight is not None:
            w += body_weight * 0.30 if exercise == 'Leg Raise' else body_weight
    return w

df['Weight'] = df.apply(lambda row: adjust_weight_for_bodyweight(row['Exercise'], row['Weight'], row['Date']), axis=1)
df['Volume'] = df['Sets'] * df['Reps'] * df['Weight']

# ── Grafici ───────────────────────────────────────────────────────────────────
def draw_grafico1(ax, df, equal_aspect=True):
    df['Date'] = pd.to_datetime(df['Date'])
    train_2026 = df[df['Date'].dt.year == 2026].copy()
    daily_counts = train_2026.groupby('Date')['Volume'].sum().reset_index(name='volume')
    calendar_df = pd.DataFrame({'Date': pd.date_range('2026-01-01', '2026-12-31', freq='D')})
    calendar_df = calendar_df.merge(daily_counts, on='Date', how='left')
    calendar_df['volume'] = calendar_df['volume'].fillna(0).astype(float)
    start_weekday = pd.Timestamp('2026-01-01').weekday()
    calendar_df['week'] = ((calendar_df['Date'] - pd.Timestamp('2026-01-01')).dt.days + start_weekday) // 7
    calendar_df['weekday'] = calendar_df['Date'].dt.weekday
    ax.set_facecolor('#191919')
    max_val = calendar_df['volume'].max()
    cmap = plt.cm.PuBuGn
    for _, row in calendar_df.iterrows():
        x, y, val = row['week'], 6 - row['weekday'], row['volume']
        if val == 0:
            color = '#161b22'
        else:
            norm = val / max_val if max_val > 0 else 0
            color = cmap(0.85 - 0.65 * norm)
        rect = mpatches.FancyBboxPatch(
            (x - 0.38, y - 0.38), 0.76, 0.76,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor='none'
        )
        ax.add_patch(rect)
    ax.set_xlim(-1.5, 54)
    ax.set_ylim(-1.2, 8.2)
    if equal_aspect:
        ax.set_aspect('equal')
    ax.axis('off')
    days = ['Dom', 'Sab', 'Ven', 'Gio', 'Mer', 'Mar', 'Lun']
    for i, day in enumerate(days):
        ax.text(-1.6, i, day, color='#8b949e', va='center', ha='right', fontsize=8, fontfamily='monospace')
    months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    for month_num, month_name in enumerate(months, start=1):
        first_day = pd.Timestamp(f'2026-{month_num:02d}-01')
        week_num = ((first_day - pd.Timestamp('2026-01-01')).days + start_weekday) // 7
        ax.text(week_num, 7.5, month_name, color='#8b949e', fontsize=8, ha='left', fontfamily='monospace')
    ax.set_title("Volume giornaliero 2026", color='white', pad=10, fontsize=13, fontweight='bold')


def draw_grafico2(ax, df):
    muscle_to_group = {
        'Dorsali': 'Schiena', 'Trapezio': 'Schiena', 'Romboidi': 'Schiena',
        'Deltoidi': 'Spalle', 'Deltoide Anteriore': 'Spalle', 'Deltoidi Laterali': 'Spalle', 'Deltoidi Posteriori': 'Spalle',
        'Pettorali': 'Petto',
        'Bicipiti': 'Bicipiti', 'Brachiale': 'Bicipiti', 'Flessori': 'Gambe',
        'Tricipiti': 'Tricipiti',
        'Addominali': 'Core',
        'Quadricipiti': 'Gambe', 'Femorali': 'Gambe', 'Glutei': 'Gambe', 'Polpacci': 'Gambe',
    }
    df1 = df[df['Date'] >= pd.Timestamp.today() - pd.Timedelta(days=7)].copy()
    df1['Secondary Muscle'] = df1['Secondary Muscle'].replace('—', None)
    df1['Vol_Primary']   = df1['Volume'] * 0.70
    df1['Vol_Secondary'] = df1['Volume'] * 0.30
    primary_grouped = df1.copy()
    primary_grouped['Group'] = df1['Primary Muscle'].map(muscle_to_group)
    vol_primary = primary_grouped.groupby(['Group', 'Primary Muscle'])['Vol_Primary'].sum()
    secondary_grouped = df1[df1['Secondary Muscle'].notna()].copy()
    secondary_grouped['Group'] = secondary_grouped['Secondary Muscle'].map(muscle_to_group)
    vol_secondary = secondary_grouped.groupby(['Group', 'Secondary Muscle'])['Vol_Secondary'].sum()
    vol_primary.index.names   = ['Group', 'Muscle']
    vol_secondary.index.names = ['Group', 'Muscle']
    combined = pd.concat([vol_primary, vol_secondary]).reset_index()
    combined.columns = ['Group', 'Muscle', 'Volume']
    combined = combined.groupby(['Group', 'Muscle'])['Volume'].sum().reset_index()
    pivot     = combined.pivot(index='Group', columns='Muscle', values='Volume').fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    base_color = "#0a8af3"
    ax.set_facecolor('#191919')
    max_n = max(len(pivot_pct.loc[g][pivot_pct.loc[g] > 0]) for g in pivot.index)
    for group_idx, group in enumerate(pivot.index):
        row     = pivot.loc[group]
        row_pct = pivot_pct.loc[group]
        sorted_muscles = row_pct[row_pct > 0].sort_values(ascending=False)
        b = 0.0
        for i, (muscle, pct) in enumerate(sorted_muscles.items()):
            val       = row[muscle]
            lightness = 0.15 + 0.55 * (i / max(max_n - 1, 1))
            color     = tuple(min(1, c + lightness * (1 - c)) for c in mcolors.to_rgb(base_color))
            ax.bar(group_idx, val, bottom=b, color=color, width=0.6)
            if pct > 5:
                text_color = 'white' if lightness < 0.45 else '#191919'
                ax.text(group_idx, b + val / 2, f'{muscle}\n{pct:.0f}%',
                        ha='center', va='center', fontsize=7.5, color=text_color, fontweight='bold')
            b += val
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, color='#8b949e')
    ax.set_title('Volume per Muscle Group', color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Muscle Group', color='#8b949e')
    ax.set_ylabel('Volume', color='#8b949e')
    ax.tick_params(colors='#8b949e', which='both')
    ax.yaxis.label.set_color('#8b949e')
    ax.xaxis.label.set_color('#191919')
    for spine in ax.spines.values():
        spine.set_edgecolor('#191919')


def draw_grafico3(ax, df):
    base_color = "#0a8af3"
    df = df[df['Date'] >= pd.Timestamp.today() - pd.Timedelta(days=100)].copy()
    df_type  = df[df['Split'].notna()].copy()
    vol_type = df_type.groupby('Split')['Volume'].sum()
    n = len(vol_type)
    colors = [
        tuple(min(1, c + (0.15 + 0.55 * (i / max(n - 1, 1))) * (1 - c)) for c in mcolors.to_rgb(base_color))
        for i in range(n)
    ]
    ax.set_facecolor('#191919')
    wedges, texts, autotexts = ax.pie(
        vol_type, labels=vol_type.index, autopct='%1.0f%%',
        colors=colors, pctdistance=0.84,
        wedgeprops=dict(width=0.3), startangle=110
    )
    for t in texts:
        t.set_color('#8b949e')
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(9)
    ax.set_title('Volume per Type', color='white', fontsize=13, fontweight='bold')

# ── Salvataggio SVG singoli ───────────────────────────────────────────────────
fig1, ax_s = plt.subplots(figsize=(20, 3.8))
fig1.patch.set_facecolor('#191919')
draw_grafico1(ax_s, df)
plt.savefig('assets/volume_giornaliero.svg', bbox_inches='tight', facecolor='#191919')
plt.close(fig1)

fig2, ax_s = plt.subplots(figsize=(16, 6))
fig2.patch.set_facecolor('#191919')
draw_grafico2(ax_s, df)
plt.savefig('assets/volume_muscle_group.svg', bbox_inches='tight', facecolor='#191919')
plt.close(fig2)

fig3, ax_s = plt.subplots(figsize=(3, 3))
fig3.patch.set_facecolor('#191919')
draw_grafico3(ax_s, df)
plt.savefig('assets/volume_per_type.svg', bbox_inches='tight', facecolor='#191919')
plt.close(fig3)

# ── Funzioni Notion API ───────────────────────────────────────────────────────
def get_image_url(image_base_url: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return f"{image_base_url}?v={now}"

def get_page_blocks(page_id: str) -> list:
    response = requests.get(f"https://api.notion.com/v1/blocks/{page_id}/children", headers=HEADERS)
    return response.json().get("results", [])

def find_image_block(blocks: list, image_url: str) -> str | None:
    base_url = image_url.split("?v")[0]
    for block in blocks:
        if block["type"] == "image":
            image_data = block["image"]
            url = image_data.get("external", {}).get("url") or image_data.get("file", {}).get("url")
            if url and base_url in url:
                print(f"✅ Blocco trovato: {block['id']}")
                return block["id"]
    print("❌ Nessun blocco immagine trovato con quell'URL")
    return None

def delete_block(block_id: str) -> bool:
    response = requests.delete(f"https://api.notion.com/v1/blocks/{block_id}", headers=HEADERS)
    if response.status_code == 200:
        print("🗑️ Blocco eliminato con successo!")
        return True
    print(f"❌ Errore {response.status_code}: {response.json()}")
    return False

def create_image_block(page_id: str, image_url: str) -> bool:
    payload = {"children": [{"type": "image", "image": {"type": "external", "external": {"url": image_url}}}]}
    response = requests.patch(f"https://api.notion.com/v1/blocks/{page_id}/children", headers=HEADERS, json=payload)
    if response.status_code == 200:
        print("✅ Blocco immagine creato con successo!")
        print(f"Block ID: {response.json()['results'][0]['id']}")
        return True
    print(f"❌ Errore {response.status_code}: {response.json()}")
    return False

def update_image_block(image_base_url: str) -> None:
    image_url = get_image_url(image_base_url)
    blocks    = get_page_blocks(PAGE_ID)
    block_id  = find_image_block(blocks, image_url)
    if block_id:
        delete_block(block_id)
    create_image_block(PAGE_ID, image_url)
    print(image_url)

# ── Aggiornamento Notion ──────────────────────────────────────────────────────
update_image_block(pie)
update_image_block(heatmap)
update_image_block(hist)