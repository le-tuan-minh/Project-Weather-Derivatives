import os
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Daily, units

STATION_FILE = "Station List Meteostat.xlsx"
OUTPUT_FOLDER = "Weather Database Meteostat 2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Đọc station ID dạng text
stations_df = pd.read_excel(STATION_FILE, dtype={"Station ID": str})
stations_df["Station ID"] = stations_df["Station ID"].str.strip()

if 'Country' in stations_df.columns:
    def is_usa(x):
        try:
            s = str(x).strip().lower()
        except Exception:
            return False
        return s in ('USA')

    stations_df = stations_df[stations_df['Country'].apply(is_usa)].copy()
else:
    print("Warning: file does not contain 'Country' column — no filtering applied.")

start_date = datetime(2004, 1, 1)
yesterday = datetime.today() - timedelta(days=1)  


def fetch_and_save_station(station_id: str):
    station_id = str(station_id).strip()
    file_path = os.path.join(OUTPUT_FOLDER, f"{station_id}.csv")

    # ==== Cập nhật dữ liệu ====
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path, parse_dates=["time"])  
        existing.set_index("time", inplace=True)

        update_start = yesterday - timedelta(days=30)
        if update_start < start_date:
            update_start = start_date
        update_end = yesterday
        old_data = existing.loc[existing.index < update_start]
    else:
        update_start = start_date
        update_end = yesterday
        old_data = pd.DataFrame()

    try:
        ts = Daily(station_id, update_start, update_end, model=True)
        ts = ts.convert(units.imperial)
        data = ts.fetch()
    except Exception as e:
        print(f"Error fetching data for {station_id}:", e)
        return

    if data.empty:
        print(f"No data returned for {station_id} in {update_start.date()}..{update_end.date()}")
        return

    # Giữ lại cột time, tmin, tmax
    data = data.reset_index()[["time", "tmin", "tmax"]]
    data.set_index("time", inplace=True)

    if not old_data.empty:
        df = pd.concat([old_data, data])
    else:
        df = data

    # Loại bỏ dòng trùng theo thời gian, giữ bản ghi cuối
    df = df[~df.index.duplicated(keep="last")]

    # Lưu file CSV với định dạng ngày chuẩn
    df.to_csv(file_path, index=True, date_format="%Y-%m-%d")
    print(f"Saved {len(df)} rows to {file_path}")


def main():
    if stations_df.empty:
        print("No stations to process after filtering. Check the 'Country' column or station list file.")
        return

    for _, row in stations_df.iterrows():
        station_id = row["Station ID"]
        if pd.notna(station_id):
            fetch_and_save_station(station_id)

    print("All stations updated.")


if __name__ == "__main__":
    main()
