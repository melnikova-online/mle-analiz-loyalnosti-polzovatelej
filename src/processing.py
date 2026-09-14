import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik
from phik.report import plot_correlation_matrix
from scipy import stats as st
from statsmodels.stats.proportion import proportions_ztest

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
        if base_path is None:
            base_path = Path(__file__).resolve().parents[1]
        
        self.base_path = Path(base_path).resolve()
        self.data_dir = self.base_path / "data"
        self.revenue_quantile = revenue_quantile
        self.verbose = verbose

        self.rates = None
        self.data = None
        self.stats = {}
        self.user_profile = None
        self.orders_clean = None

        self._log(f"Базовый путь:  {self.base_path}")
        self._log(f"Папка данных:  {self.data_dir}")

    def __repr__(self):
        rows = len(self.data) if self.data is not None else 0
        return f"DataProcessor(base_path='{self.base_path}', rows={rows:,})"

    # ------------------------------------------------------------------ load & preprocess
    def load_data(self):
        path = self._file(self.PURCHASES_FILE)
        df = pd.read_csv(path, parse_dates=["order_dt", "order_ts"])

        missing = set(self.REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Нет обязательных колонок: {sorted(missing)}")

        self._log(f"Заказы загружены: {len(df):,} строк, {df.shape[1]} колонок")
        return df

    def load_rates(self):
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

    def preprocess(self, df, rates):
        df = df.copy()
        rows_before = len(df)
        mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2

        df["order_dt"] = pd.to_datetime(df["order_dt"], errors="coerce").dt.normalize()
        df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")

        n_dupes = int(df.duplicated().sum())
        df = df.drop_duplicates().drop_duplicates(subset="order_id")

        n_na = int(df[self.REQUIRED_COLS].isna().any(axis=1).sum())
        df = df.dropna(subset=self.REQUIRED_COLS)

        n_bad = 0
        if "tickets_count" in df.columns:
            n_bad = int(((df["revenue"] <= 0) | (df["tickets_count"] <= 0)).sum())
            df = df[(df["revenue"] > 0) & (df["tickets_count"] > 0)]
        else:
            n_bad = int((df["revenue"] <= 0).sum())
            df = df[df["revenue"] > 0]

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

        if "tickets_count" in df.columns:
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

        n_out = 0
        if self.revenue_quantile is not None and "revenue_rub" in df.columns:
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

    def run(self):
        df = self.load_data()
        rates = self.load_rates()
        self.data = self.preprocess(df, rates)
        self._log("Данные готовы к анализу")
        return self.data

    def report(self):
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
        if self.data is None:
            raise RuntimeError("Нет данных: сначала вызовите run().")
        path = self.data_dir / filename
        self.data.to_csv(path, index=False)
        self._log(f"Сохранено {len(self.data):,} строк в {path}")
        return path

    def _file(self, name):
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Файл не найден: {path}\n"
                f"Проверьте, что '{name}' лежит в папке {self.data_dir}")
        return path

    def _log(self, message):
        if self.verbose:
            print(message)

    # ==============================================================================
    # НОВЫЕ МЕТОДЫ: Анализ и Моделирование
    # ==============================================================================

    def create_user_profile(self):
        if self.data is None:
            raise RuntimeError("Сначала вызовите run().")

        df = self.data.copy()
        df = df.sort_values(['user_id', 'order_ts']).reset_index(drop=True)

        df['days_gap'] = df.groupby('user_id')['order_dt'].diff().dt.days

        user_profile = df.groupby('user_id', observed=True).agg(
            first_order_dt   = ('order_dt', 'min'),
            last_order_dt    = ('order_dt', 'max'),
            first_device     = ('device_type_canonical', 'first'),
            first_region     = ('region_name', 'first'),
            first_event_type = ('event_type_main', 'first'),
            orders_total     = ('order_id', 'nunique'),
            avg_revenue_rub  = ('revenue_rub', 'mean'),
            avg_days_between = ('days_gap', 'mean'),
            total_revenue    = ('revenue_rub', 'sum'),
            total_tickets    = ('tickets_count', 'sum')
        ).reset_index()

        user_profile['avg_revenue_rub'] = user_profile['avg_revenue_rub'].round(2)
        user_profile['avg_days_between'] = user_profile['avg_days_between'].round(2)
        user_profile['is_two'] = (user_profile['orders_total'] >= 2).astype(int)
        
        days_active = df.groupby('user_id')['order_dt'].nunique()
        user_profile['active_days'] = user_profile['user_id'].map(days_active)
        user_profile['is_returning'] = (user_profile['active_days'] >= 2).astype(int)

        activity = df.groupby('user_id', observed=True).agg(
            orders  = ('order_id', 'size'),
            days    = ('order_dt', 'nunique'),
            regions = ('region_name', 'nunique'),
            events  = ('event_id', 'nunique')
        )
        activity['per_day'] = activity['orders'] / activity['days']

        ORDERS_MAX = 152
        REGIONS_MAX = 20
        PER_DAY_MAX = 20

        flag_orders  = activity['orders'] > ORDERS_MAX
        flag_regions = activity['regions'] >= REGIONS_MAX
        flag_rate    = activity['per_day'] > PER_DAY_MAX

        activity['is_bot'] = flag_orders | flag_regions | flag_rate
        bot_ids = set(activity[activity['is_bot']].index)

        self._log(f"Обнаружено и отфильтровано {len(bot_ids)} технических аккаунтов.")

        profile_clean = user_profile[~user_profile['user_id'].isin(bot_ids)].copy()
        orders_clean = df[~df['user_id'].isin(bot_ids)].copy()

        recalc = orders_clean.groupby('user_id', observed=True).agg(
            orders_total     = ('order_id', 'size'),
            avg_revenue_rub  = ('revenue_rub', 'mean'),
            avg_days_between = ('days_gap', 'mean'),
            active_days      = ('order_dt', 'nunique')
        )
        
        profile_clean = profile_clean.set_index('user_id')
        profile_clean.update(recalc[['orders_total', 'avg_revenue_rub']])
        profile_clean['avg_days_between'] = recalc['avg_days_between']
        profile_clean['active_days'] = recalc['active_days']
        profile_clean = profile_clean.reset_index()

        profile_clean['orders_total'] = profile_clean['orders_total'].astype(int)
        profile_clean['is_two'] = (profile_clean['orders_total'] >= 2).astype(int)
        profile_clean['is_returning'] = (profile_clean['active_days'] >= 2).astype(int)
        profile_clean['avg_revenue_rub'] = profile_clean['avg_revenue_rub'].round(2)
        profile_clean['avg_days_between'] = profile_clean['avg_days_between'].round(2)

        self.user_profile = profile_clean
        self.orders_clean = orders_clean
        
        self._log(f"Профиль пользователя создан: {len(profile_clean):,} пользователей.")
        return profile_clean

    def calculate_retention(self, cohort_size='month', metric='is_returning'):
        if self.user_profile is None:
            raise RuntimeError("Сначала вызовите create_user_profile().")

        df = self.user_profile.copy()
        
        if cohort_size == 'month':
            df['cohort'] = df['first_order_dt'].dt.to_period('M').astype(str)
        elif cohort_size == 'week':
            df['cohort'] = df['first_order_dt'].dt.to_period('W').astype(str)
        else:
            raise ValueError("cohort_size должен быть 'month' или 'week'")

        segments = ['first_region', 'first_event_type', 'first_device']
        retention_report = {}

        for seg_col in segments:
            if seg_col not in df.columns:
                continue
                
            agg = df.groupby(['cohort', seg_col]).agg(
                total_users=(seg_col, 'size'),
                returned_users=(metric, 'sum')
            ).reset_index()
            
            agg['retention_rate'] = (agg['returned_users'] / agg['total_users']).round(4)
            
            top_segments = agg.groupby(seg_col)['total_users'].sum().nlargest(10).index
            seg_report = agg[agg[seg_col].isin(top_segments)].sort_values(['cohort', seg_col])
            
            retention_report[seg_col] = seg_report

        total_agg = df.groupby('cohort').agg(
            total_users=('user_id', 'size'),
            returned_users=(metric, 'sum')
        ).reset_index()
        total_agg['retention_rate'] = (total_agg['returned_users'] / total_agg['total_users']).round(4)
        retention_report['overall'] = total_agg

        self._log(f"Retention рассчитан для сегментов: {list(retention_report.keys())}")
        return retention_report

    def test_hypotheses(self):
        if self.user_profile is None:
            raise RuntimeError("Сначала вызовите create_user_profile().")

        df = self.user_profile
        results = {}

        # --- Гипотеза 1: Спорт vs Концерты ---
        sport_mask = df['first_event_type'] == 'спорт'
        concert_mask = df['first_event_type'] == 'концерты'
        
        if sport_mask.sum() > 0 and concert_mask.sum() > 0:
            count = np.array([df.loc[sport_mask, 'is_two'].sum(), 
                              df.loc[concert_mask, 'is_two'].sum()])
            nobs = np.array([sport_mask.sum(), concert_mask.sum()])
            
            stat, p_one = proportions_ztest(count, nobs, alternative='larger')
            stat2, p_two = proportions_ztest(count, nobs, alternative='two-sided')
            
            results['hypothesis_1'] = {
                'sport_rate': (count[0]/nobs[0]).round(3),
                'concert_rate': (count[1]/nobs[1]).round(3),
                'p_value_one_sided': p_one,
                'p_value_two_sided': p_two,
                'conclusion': 'Подтверждена' if p_one < 0.05 else 'Отвергнута'
            }
            self._log(f"Гипотеза 1 (Спорт > Концерты): p-value={p_one:.4f}. Вывод: {results['hypothesis_1']['conclusion']}")

        # --- Гипотеза 2: Активные регионы vs Остальные ---
        reg_stats = df.groupby('first_region')['user_id'].size().sort_values(ascending=False)
        top_regions = reg_stats.head(5).index.tolist()
        
        is_top = df['first_region'].isin(top_regions)
        
        count = np.array([df.loc[is_top, 'is_two'].sum(), 
                          df.loc[~is_top, 'is_two'].sum()])
        nobs = np.array([is_top.sum(), (~is_top).sum()])
        
        stat, p_one = proportions_ztest(count, nobs, alternative='larger')
        
        results['hypothesis_2'] = {
            'top_regions': top_regions,
            'top_rate': (count[0]/nobs[0]).round(3),
            'other_rate': (count[1]/nobs[1]).round(3),
            'p_value': p_one,
            'conclusion': 'Подтверждена' if p_one < 0.05 else 'Отвергнута'
        }
        self._log(f"Гипотеза 2 (Топ регионы > Остальные): p-value={p_one:.4f}. Вывод: {results['hypothesis_2']['conclusion']}")

        # --- Проверка корреляции Спирмена (ИСПРАВЛЕНО) ---
        big_regs = reg_stats[reg_stats >= 100]
        
        if len(big_regs) > 1:
            # Ряд 1: Размер региона (число пользователей)
            x_values = big_regs.values.astype(float)
            
            # Ряд 2: Средняя доля вернувшихся в этом регионе
            loyalty_by_region = df.groupby('first_region')['is_two'].mean()
            y_values = loyalty_by_region.loc[big_regs.index].values.astype(float)
            
            try:
                rho, p_rho = st.spearmanr(x_values, y_values)
                
                results['spearman_corr'] = {
                    'rho': rho,
                    'p_value': p_rho,
                    'n_samples': len(x_values),
                    'conclusion': 'Значимая монотонная связь' if p_rho < 0.05 else 'Связи нет'
                }
                self._log(f"Корреляция Спирмена (размер региона vs лояльность): rho={rho:.3f}, p={p_rho:.4f}")
            except Exception as e:
                self._log(f"Ошибка расчета корреляции Спирмена: {e}")
                results['spearman_corr'] = {'error': str(e)}
        else:
            self._log("Недостаточно крупных регионов для расчета корреляции Спирмена.")
            results['spearman_corr'] = {'error': 'Мало данных'}

        return results

    def correlation_analysis(self, save_plot=False):
        if self.user_profile is None:
            raise RuntimeError("Сначала вызовите create_user_profile().")

        df = self.user_profile.copy()
        
        def order_segment(n):
            if n == 1: return '1 заказ'
            if n == 2: return '2 заказа'
            if n <= 4: return '3-4 заказа'
            return '5+ заказов'

        df['orders_segment'] = df['orders_total'].apply(order_segment).astype('category')
        
        cols = ['orders_segment', 'first_event_type', 'first_device', 
                'first_region', 'first_dow', 'avg_revenue_rub', 'is_two']
        
        if 'first_dow' not in df.columns:
            df['first_dow'] = df['first_order_dt'].dt.day_name().astype('category')

        interval_cols = ['avg_revenue_rub']
        
        try:
            phik_matrix = df[cols].phik_matrix(interval_cols=interval_cols)
            significance_matrix = df[cols].significance_matrix(interval_cols=interval_cols)
            
            target_corr = phik_matrix['orders_segment'].drop('orders_segment').sort_values(ascending=False)
            target_sig = significance_matrix['orders_segment'].drop('orders_segment').sort_values(ascending=False)

            self._log("Корреляционный анализ (phi_k) завершен.")
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(phik_matrix, dtype=bool), k=1)
            sns.heatmap(phik_matrix, mask=mask, annot=True, fmt='.2f', cmap='YlOrRd', 
                        vmin=0, vmax=1, linewidths=.5, linecolor='white')
            plt.title('Матрица корреляции Phi-K: Признаки профиля и число заказов')
            
            if save_plot:
                plot_path = self.data_dir / "correlation_heatmap.png"
                plt.savefig(plot_path, dpi=300)
                self._log(f"Тепловая карта сохранена в {plot_path}")
            
            plt.show()

            return {
                'correlation': target_corr,
                'significance': target_sig,
                'matrix': phik_matrix
            }

        except Exception as e:
            self._log(f"Ошибка при расчете phi_k: {e}")
            num_cols = ['orders_total', 'avg_revenue_rub', 'avg_days_between']
            corr = df[num_cols].corr()
            return {'correlation': corr}

    def generate_full_report(self):
        print("\n=== НАЧАЛО ГЕНЕРАЦИИ ПОЛНОГО ОТЧЕТА ===\n")
        
        self._log("Шаг 1: Создание профиля пользователя и фильтрация ботов...")
        self.create_user_profile()
        print("\n--- Статистика профиля (после очистки) ---")
        print(self.user_profile.describe(include='all').round(2))
        print(f"Доля вернувшихся (is_returning): {self.user_profile['is_returning'].mean():.2%}")
        print(f"Доля с >=2 заказами (is_two): {self.user_profile['is_two'].mean():.2%}")

        self._log("Шаг 2: Расчет Retention по сегментам...")
        ret_report = self.calculate_retention(cohort_size='month')
        print("\n--- Пример Retention по регионам (Топ-5) ---")
        top_regs = self.user_profile.groupby('first_region')['user_id'].size().nlargest(5).index
        if 'first_region' in ret_report:
            print(ret_report['first_region'][ret_report['first_region']['first_region'].isin(top_regs)].head(15))

        self._log("Шаг 3: Проверка гипотез...")
        hyp_results = self.test_hypotheses()
        print("\n--- Результаты проверки гипотез ---")
        for k, v in hyp_results.items():
            print(f"{k}: {v}")

        self._log("Шаг 4: Корреляционный анализ...")
        corr_results = self.correlation_analysis(save_plot=True)
        print("\n--- Топ-5 признаков, связанных с числом заказов (phi_k) ---")
        if 'correlation' in corr_results and isinstance(corr_results['correlation'], pd.Series):
            print(corr_results['correlation'].head(5))
        else:
            print(corr_results)

        print("\n=== ОТЧЕТ ЗАВЕРШЕН ===")
        return {
            'profile': self.user_profile,
            'retention': ret_report,
            'hypotheses': hyp_results,
            'correlations': corr_results
        }

# --- ТОЧКА ВХОДА ---
if __name__ == "__main__":
    try:
        processor = DataProcessor()
        data = processor.run()
        
        full_report = processor.generate_full_report()
        
    except FileNotFoundError as e:
        print(f"\n❌ ОШИБКА ФАЙЛА: {e}")
        print("\n💡 ПОДСКАЗКА:")
        print("1. Убедитесь, что файлы лежат в папке: mle-analiz-loyalnosti-polzovatelej/data/")
        print("2. Проверьте названия файлов: afisha_raw.csv и final_tickets_tenge_df.csv")
    except Exception as e:
        print(f"\n❌ ПРОИЗОШЛА ОШИБКА: {e}")
        import traceback
        traceback.print_exc()