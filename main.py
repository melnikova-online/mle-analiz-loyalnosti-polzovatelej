import os
from pathlib import Path

# Импортируем наш класс из файла processing.py
# Убедитесь, что processing.py лежит в той же папке или в папке src
try:
    from processing import DataProcessor
except ImportError:
    # Если файл лежит в папке src, пробуем импортировать оттуда
    try:
        from src.processing import DataProcessor
    except ImportError:
        print("❌ Ошибка: Не удалось найти файл processing.py или папку src/processing.py")
        print("Проверьте структуру проекта.")
        exit(1)

def main():
    print("=" * 60)
    print("🚀 ЗАПУСК ПРОЕКТА: Анализ лояльности пользователей")
    print("=" * 60)

    # 1. Инициализация и запуск базового пайплайна (Загрузка + Очистка)
    print("\n[1/4] Инициализация и загрузка данных...")
    try:
        processor = DataProcessor(verbose=True)
        
        # Запуск загрузки и предобработки
        data = processor.run()
        print(f"✅ Данные загружены и очищены. Строк: {len(data):,}")
        
    except FileNotFoundError as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Файл не найден.")
        print(f"   {e}")
        print("\n💡 Решение: Положите файлы afisha_raw.csv и final_tickets_tenge_df.csv в папку ./data/")
        return
    except Exception as e:
        print(f"\n❌ Ошибка на этапе загрузки/очистки: {e}")
        return

    # 2. Генерация профилей пользователей и фильтрация ботов
    print("\n[2/4] Создание профилей пользователей и фильтрация ботов...")
    try:
        profile = processor.create_user_profile()
        print(f"✅ Профиль создан. Пользователей (без ботов): {len(profile):,}")
        
        # Статистика по ботам
        total_users_raw = len(processor.data['user_id'].unique())
        bots_removed = total_users_raw - len(profile)
        print(f"🤖 Отфильтровано технических аккаунтов (ботов): {bots_removed:,} ({(bots_removed/total_users_raw)*100:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Ошибка при создании профиля: {e}")
        return

    # 3. Расчет Retention и проверка гипотез
    print("\n[3/4] Расчет Retention и проверка гипотез...")
    try:
        # Расчет Retention (берем общий отчет)
        retention_report = processor.calculate_retention(cohort_size='month', metric='is_returning')
        overall_ret = retention_report.get('overall')
        
        if overall_ret is not None and not overall_ret.empty:
            avg_retention = overall_ret['retention_rate'].mean()
            print(f"📊 Средний Retention (доля вернувшихся) по когортам: {avg_retention:.2%}")
        else:
            print("⚠️ Не удалось рассчитать Retention.")

        # Проверка гипотез
        hypotheses = processor.test_hypotheses()
        
        print("\n--- Результаты проверки гипотез (Шаг 4) ---")
        
        # Гипотеза 1: Спорт vs Концерты
        h1 = hypotheses.get('hypothesis_1')
        if h1:
            status = "✅ ПОДТВЕРЖДЕНА" if h1['conclusion'] == 'Подтверждена' else "❌ ОТВЕРГНУТА"
            print(f"H1 (Спорт > Концерты по возвратности): {status}")
            print(f"   Спорт: {h1['sport_rate']:.2%}, Концерты: {h1['concert_rate']:.2%}, p-value: {h1['p_value_one_sided']:.4f}")
        
        # Гипотеза 2: Топ регионы vs Остальные
        h2 = hypotheses.get('hypothesis_2')
        if h2:
            status = "✅ ПОДТВЕРЖДЕНА" if h2['conclusion'] == 'Подтверждена' else "❌ ОТВЕРГНУТА"
            print(f"H2 (Топ регионы > Остальные): {status}")
            print(f"   Топ: {h2['top_rate']:.2%}, Остальные: {h2['other_rate']:.2%}, p-value: {h2['p_value']:.4f}")

        # Корреляция Спирмена
        corr = hypotheses.get('spearman_corr')
        if corr and 'error' not in corr:
            print(f"📉 Корреляция Спирмена (размер региона vs лояльность): rho={corr['rho']:.3f}, p={corr['p_value']:.4f}")
            print(f"   Вывод: {corr['conclusion']}")

    except Exception as e:
        print(f"\n❌ Ошибка при расчете метрик: {e}")
        return

    # 4. Финальный краткий отчет (Суть выводов Шага 5)
    print("\n" + "=" * 60)
    print("📋 ИТОГОВЫЙ ОТЧЕТ (ВЫВОДЫ ПРОЕКТА)")
    print("=" * 60)
    
    # Общие метрики
    total_users = len(profile)
    returning_users = profile['is_returning'].sum()
    avg_check = profile['avg_revenue_rub'].mean()
    avg_freq = profile['avg_days_between'].mean()
    
    print(f"\n👥 База пользователей (после чистки): {total_users:,}")
    print(f"🔄 Доля лояльных (совершили >1 заказа): {returning_users/total_users:.2%}")
    print(f"💰 Средний чек лояльного пользователя: {avg_check:,.2f} руб.")
    print(f"📅 Средний интервал между заказами: {avg_freq:.1f} дней")

    # Выводы по сегментам
    print("\n🎯 Ключевые инсайты по сегментам:")
    
    # Топ регионы по количеству пользователей
    top_regions = profile.groupby('first_region')['user_id'].count().nlargest(5)
    print("  Топ-5 регионов по объему аудитории:")
    for region, count in top_regions.items():
        print(f"    - {region}: {count:,} пользователей")

    # Выводы по гипотезам (кратко)
    print("\n🧠 Подтвержденные гипотезы:")
    if h1 and h1['conclusion'] == 'Подтверждена':
        print("  - Гипотеза 1: Пользователи, начавшие со спорта, демонстрируют более высокую лояльность, чем любители концертов.")
    else:
        print("  - Гипотеза 1: Не подтверждена (разница в лояльности между жанрами статистически незначима или обратная).")
        
    if h2 and h2['conclusion'] == 'Подтверждена':
        print("  - Гипотеза 2: Пользователи из крупнейших регионов (Топ-5) более лояльны, чем пользователи из остальных регионов.")
    else:
        print("  - Гипотеза 2: Не подтверждена (размер региона не влияет на лояльность).")

    if corr and 'error' not in corr and corr['conclusion'] == 'Значимая монотонная связь':
        print(f"  - Наблюдается значимая корреляция ({corr['rho']:.3f}) между размером региона и лояльностью пользователей.")

    print("\n✅ Анализ завершен успешно.")
    print("=" * 60)

if __name__ == "__main__":
    main()