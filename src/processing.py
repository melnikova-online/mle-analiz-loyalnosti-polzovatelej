import pandas as pd
from pathlib import Path
# Настройки отображения Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


class DataProcessor:
    """Готовит витрину заказов для анализа лояльности пользователей."""

    PURCHASES_FILE = "afisha_raw.csv"
    RATES_FILE = "final_tickets_tenge_df.csv"

    CAT_COLS = ["device_type_canonical", "currency_code", "event_type_main",
                "service_name", "region_name", "city_name"]
    REQUIRED_COLS = ["user_id", "order_id", "order_dt", "order_ts", "revenue"]

    def __init__(self, base_path=None, revenue_quantile=0.99, verbose=True):
        # Логика определения пути: если скрипт запущен из src, берем родителя
        if base_path is None:
            # __file__ - это путь к этому файлу. .parents[1] - это папка проекта (на 2 уровня выше src)
            base_path = Path(__file__).resolve().parents[1]
        
        self.base_path = Path(base_path).resolve()
        self.data_dir = self.base_path / "data"
        self.revenue_quantile = revenue_quantile
        self.verbose = verbose

        self.rates = None
        self.data = None
        self.stats = {}

        self._log(f"Базовый путь:  {self.base_path}")
        self._log(f"Папка данных:  {self.data_dir}")

    def __repr__(self):
        rows = len(self.data) if self.data is not None else 0
        return f"DataProcessor(base_path='{self.base_path}', rows={rows:,})"

    # ------------------------------------------------------------------ load
    def load_data(self):
        """Загружает CSV с заказами."""
        path = self._file(self.PURCHASES_FILE)
        df = pd.read_csv(path, parse_dates=["order_dt", "order_ts"])

        missing = set(self.REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Нет обязательных колонок: {sorted(missing)}")

        self._log(f"Заказы загружены: {len(df):,} строк, {df.shape[1]} колонок")
        return df

    def load_rates(self):
        """Загружает курс тенге и строит непрерывный календарь курсов."""
        path = self._file(self.RATES_FILE)
        rates = pd.read_csv(path, parse_dates=["data"])

        missing = {"data", "curs", "nominal"} - set(rates.columns)
        if missing:
            raise ValueError(f"В файле курсов нет колонок: {sorted(missing)}")

        rates["rate_per_kzt"] = rates["curs"] / rates["nominal"]
        rates = (rates[["data", "rate_per_kzt"]]
                 .rename(columns={"data": "order_dt"})
                 .assign(order_dt=lambda x: x["order_dt"].dt.normalize())
                 .drop_duplicates(subset="order_dt"))

        calendar = pd.DataFrame(
            {"order_dt": pd.date_range("2024-01-01", "2024-12-31", freq="D")})
        rates = (calendar.merge(rates, on="order_dt", how="left")
                 .sort_values("order_dt"))
        rates["rate_per_kzt"] = rates["rate_per_kzt"].ffill().bfill()

        if rates["rate_per_kzt"].isna().any():
            raise ValueError("В календаре курсов остались пропуски.")

        self._log(f"Курсы валют: {len(rates)} дней, пропусков нет")
        self.rates = rates.reset_index(drop=True)
        return self.rates

    # ------------------------------------------------------------ preprocess
    def preprocess(self, df, rates):
        """Приведение типов, конвертация валют и базовая чистка."""
        df = df.copy()
        rows_before = len(df)
        mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2

        # 1. даты
        df["order_dt"] = pd.to_datetime(df["order_dt"], errors="coerce").dt.normalize()
        df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")

        # 2. чистка: дубли, пропуски, аномалии
        n_dupes = int(df.duplicated().sum())
        df = df.drop_duplicates().drop_duplicates(subset="order_id")

        n_na = int(df[self.REQUIRED_COLS].isna().any(axis=1).sum())
        df = df.dropna(subset=self.REQUIRED_COLS)

        n_bad = int(((df["revenue"] <= 0) | (df["tickets_count"] <= 0)).sum())
        df = df[(df["revenue"] > 0) & (df["tickets_count"] > 0)]

        # 3. конвертация валюты
        df = df.drop(columns=["rate_per_kzt", "revenue_rub"], errors="ignore")
        df = df.merge(rates, on="order_dt", how="left")

        currency = df["currency_code"].astype(str).str.strip().str.lower()
        df["revenue_rub"] = (df["revenue"]
                             .where(currency == "rub",
                                    df["revenue"] * df["rate_per_kzt"])
                             .round(2))

        self.stats["currency_check"] = (
            df.assign(_cur=currency)
            .groupby("_cur", observed=True)
            .agg(orders=("revenue", "size"),
                 revenue_orig=("revenue", "sum"),
                 revenue_rub=("revenue_rub", "sum"),
                 na_rub=("revenue_rub", lambda s: int(s.isna().sum())))
            .round(2))

        # 4. типы данных
        df["tickets_count"] = pd.to_numeric(
            df["tickets_count"], errors="coerce", downcast="integer")
        
        if "days_since_prev" in df.columns:
            df["days_since_prev"] = pd.to_numeric(
                df["days_since_prev"], errors="coerce").astype("Int16")

        for col in ["revenue", "revenue_rub", "rate_per_kzt"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

        for col in [c for c in self.CAT_COLS if c in df.columns]:
            df[col] = (df[col].astype(str)
                       .str.strip()
                       .str.replace(r"\s+", " ", regex=True)
                       .astype("category"))

        # 5. выбросы по 99-му перцентилю revenue_rub
        n_out = 0
        if self.revenue_quantile is not None:
            p = df["revenue_rub"].quantile(self.revenue_quantile)
            n_out = int((df["revenue_rub"] > p).sum())
            df = df[df["revenue_rub"] <= p]
            self._log(f"Порог выбросов (p{self.revenue_quantile:.0%}): "
                      f"{p:.2f} руб., удалено {n_out:,} строк")

        df = df.sort_values(["user_id", "order_ts"]).reset_index(drop=True)
        mem_after = df.memory_usage(deep=True).sum() / 1024 ** 2

        self.stats.update({
            "rows_before": rows_before,
            "rows_after": len(df),
            "duplicates": n_dupes,
            "rows_with_na": n_na,
            "anomalies": n_bad,
            "outliers": n_out,
            "users": int(df["user_id"].nunique()),
            "memory_mb": f"{mem_before:.1f} -> {mem_after:.1f}",
        })

        self._log(f"Чистка: {rows_before:,} -> {len(df):,} строк "
                  f"(дубли {n_dupes}, пропуски {n_na}, аномалии {n_bad})")
        self._log(f"Память: {mem_before:.1f} MB -> {mem_after:.1f} MB")
        return df

    # -------------------------------------------------------------- pipeline
    def run(self):
        """Полный цикл: загрузка заказов и курсов + предобработка."""
        df = self.load_data()
        rates = self.load_rates()
        self.data = self.preprocess(df, rates)
        self._log("Данные готовы к анализу")
        return self.data

    def report(self):
        """Сводка по качеству итоговой витрины."""
        if self.data is None:
            raise RuntimeError("Сначала вызовите run().")

        df = self.data

        print("\n--- Итоговая витрина ---")
        print(f"Заказов:        {len(df):,}")
        print(f"Пользователей:  {df['user_id'].nunique():,}")
        print(f"Период:         {df['order_dt'].min().date()} — "
              f"{df['order_dt'].max().date()}")
        print(f"Средний чек:    {df['revenue_rub'].mean():.2f} руб.")
        print(f"Медианный чек:  {df['revenue_rub'].median():.2f} руб.")
        print(f"Полных дублей:  {df.duplicated().sum()} | "
              f"дублей order_id: {df['order_id'].duplicated().sum()}")

        # пропуски в days_since_prev должны быть только у первых заказов
        if "days_since_prev" in df.columns:
            first_ts = df.groupby("user_id", observed=True)["order_ts"].transform("min")
            is_first = df["order_ts"] == first_ts
            na_total = int(df["days_since_prev"].isna().sum())
            na_first = int((df["days_since_prev"].isna() & is_first).sum())
            status = "совпадает" if na_total == na_first else "РАСХОЖДЕНИЕ"
            print(f"days_since_prev: пропусков {na_total:,}, "
                  f"из них первые заказы {na_first:,} — {status}")

        na = pd.DataFrame({
            "na_count": df.isna().sum(),
            "na_pct": (100 * df.isna().mean()).round(2),
            "dtype": df.dtypes.astype(str),
        })
        na = na[na["na_count"] > 0]

        print("\nПропуски по колонкам:")
        print(na if not na.empty else "нет")

        print("\nПроверка конвертации валют:")
        print(self.stats.get("currency_check", "нет данных"))

        return na

    def save(self, filename="afisha_processed.csv"):
        """Сохраняет готовую витрину в папку data/."""
        if self.data is None:
            raise RuntimeError("Нет данных: сначала вызовите run().")

        path = self.data_dir / filename
        self.data.to_csv(path, index=False)
        self._log(f"Сохранено {len(self.data):,} строк в {path}")
        return path

    # --------------------------------------------------------------- helpers
    def _file(self, name):
        """Возвращает путь к файлу в data/, проверяя его наличие."""
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {path}\n"
                f"Проверьте, что '{name}' лежит в папке {self.data_dir}")
        return path

    def _log(self, message):
        """Печатает сообщение, если включён режим verbose."""
        if self.verbose:
            print(message)


# --- ТОЧКА ВХОДА (ИСПРАВЛЕНО) ---
# Здесь НЕЛЬЗЯ использовать return. Только вызовы функций.
if __name__ == "__main__":
    try:
        # Создаем экземпляр класса. Путь определится автоматически (родительская папка)
        processor = DataProcessor()
        
        # Запускаем полный пайплайн
        data = processor.run()
        
        # Выводим отчет
        processor.report()

        print("\n--- Первые 5 строк ---")
        print(data.head())
        
        # Опционально: сохранить результат
        # processor.save("afisha_processed.csv")
        
    except FileNotFoundError as e:
        print(f"\n❌ ОШИБКА ФАЙЛА: {e}")
        print("\n💡 ПОДСКАЗКА:")
        print("1. Убедитесь, что файлы лежат в папке: mle-analiz-loyalnosti-polzovatelej/data/")
        print("2. Проверьте названия файлов: afisha_raw.csv и final_tickets_tenge_df.csv")
    except Exception as e:
        print(f"\n❌ ПРОИЗОШЛА ОШИБКА: {e}")