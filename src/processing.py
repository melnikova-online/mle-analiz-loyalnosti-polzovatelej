import pandas as pd
from pathlib import Path
import os

class DataProcessor:
    def __init__(self, base_path: str = "."):
        # Преобразуем строку в объект Path
        self.base_path = Path(base_path).resolve()
        # Папка data должна быть рядом с base_path
        self.data_dir = self.base_path / "data"
        
        print(f"📁 Базовый путь: {self.base_path}")
        print(f"📁 Ищем данные в: {self.data_dir}")

    def load_and_process(self) -> pd.DataFrame:
        """Выполняет загрузку, подготовку курсов и предобработку."""
        
        # --- ШАГ 1: Загрузка данных о покупках ---
        purchases_file = self.data_dir / "afisha_raw.csv"
        if not purchases_file.exists():
            raise FileNotFoundError(f"❌ ФАЙЛ НЕ НАЙДЕН: {purchases_file}\n"
                                    f"Проверьте, лежит ли файл afisha_raw.csv в папке {self.data_dir}")
        
        print(f"✅ Загружаем: {purchases_file.name}")
        df = pd.read_csv(purchases_file)
        print(f"   Загружено строк: {len(df):,}, колонок: {len(df.columns)}")

        # --- ШАГ 2: Загрузка и подготовка курсов валют ---
        rates_file = self.data_dir / "final_tickets_tenge_df.csv"
        if not rates_file.exists():
            raise FileNotFoundError(f"❌ ФАЙЛ НЕ НАЙДЕН: {rates_file}\n"
                                    f"Проверьте, лежит ли файл final_tickets_tenge_df.csv в папке {self.data_dir}")

        print(f"✅ Загружаем курсы: {rates_file.name}")
        rates = pd.read_csv(rates_file, parse_dates=['data'])
        
        # Расчет курса
        rates['rate_per_kzt'] = rates['curs'] / rates['nominal']
        rates = rates[['data', 'rate_per_kzt']].rename(columns={'data': 'order_dt'})

        # Полный календарь для заполнения пропусков
        calendar = pd.DataFrame({
            'order_dt': pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        })
        
        rates_full = calendar.merge(rates, on='order_dt', how='left').sort_values('order_dt')
        rates_full['rate_per_kzt'] = rates_full['rate_per_kzt'].ffill().bfill()
        
        na_count = rates_full['rate_per_kzt'].isna().sum()
        if na_count > 0:
            print(f"⚠️ Внимание: осталось {na_count} пропусков в курсе валют!")
        else:
            print("✅ Курсы валют подготовлены, пропусков нет.")

        # --- ШАГ 3: Предобработка (Типы, Конвертация, Чистка) ---
        
        # 3.1. Даты
        df['order_dt'] = pd.to_datetime(df['order_dt']).dt.normalize()
        if 'order_ts' in df.columns:
            df['order_ts'] = pd.to_datetime(df['order_ts'])

        # 3.2. Конвертация валюты
        df = df.merge(rates_full, on='order_dt', how='left')
        
        # Нормализация валюты
        currency = df['currency_code'].str.strip().str.lower()
        
        # Конвертация
        df['revenue_rub'] = df['revenue'].where(
            currency == 'rub',
            df['revenue'] * df['rate_per_kzt']
        ).round(2)

        # 3.3. Оптимизация типов (Downcast)
        if 'tickets_count' in df.columns:
            df['tickets_count'] = pd.to_numeric(df['tickets_count'], downcast='integer')
        if 'days_since_prev' in df.columns:
            df['days_since_prev'] = df['days_since_prev'].astype('Int16')
            
        for col in ['revenue', 'revenue_rub']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

        # 3.4. Чистка категорий
        cat_cols = ['device_type_canonical', 'currency_code', 'event_type_main',
                    'service_name', 'region_name', 'city_name']
        
        existing_cat_cols = [c for c in cat_cols if c in df.columns]
        
        for col in existing_cat_cols:
            df[col] = (df[col].astype(str)
                              .str.strip()
                              .str.replace(r'\s+', ' ', regex=True)
                              .astype('category'))

        # 3.5. Удаление выбросов (Top 1%)
        if 'revenue_rub' in df.columns:
            p99 = df['revenue_rub'].quantile(0.99)
            initial_count = len(df)
            df = df[df['revenue_rub'] <= p99].copy()
            removed_count = initial_count - len(df)
            print(f"🗑️ Удалено выбросов (top 1%): {removed_count} строк.")

        print(f"✅ Обработка завершена. Итого строк: {len(df):,}")
        return df

# --- ТОЧКА ВХОДА ---
if __name__ == "__main__":
    # ВАЖНО: Если ты запускаешь из папки src, используй "..", если из корня - "."
    # Ниже логика автоматически определяет, откуда запущен скрипт, чтобы найти папку data
    
    current_dir = Path(__file__).parent.resolve()
    
    # Если скрипт лежит в src, а data лежит в корне проекта, то base_path = ".."
    # Если скрипт лежит в корне, то base_path = "."
    # Мы проверяем, есть ли папка data внутри текущей директории. Если нет, идем вверх.
    
    if (current_dir / "data").exists():
        base_path = current_dir
    else:
        base_path = current_dir.parent
        
    print(f"🔍 Автоматически определен базовый путь: {base_path}")
    
    try:
        processor = DataProcessor(base_path=str(base_path))
        final_df = processor.load_and_process()
        
        print("\n--- Результат (первые 5 строк) ---")
        print(final_df.head())
        print("\n--- Типы данных ---")
        print(final_df.info())
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("\n💡 ПОДСКАЗКА: Убедитесь, что файлы afisha_raw.csv и final_tickets_tenge_df.csv лежат в папке 'data' рядом с проектом.")