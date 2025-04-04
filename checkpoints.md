# План работы над проектом

Встречи по обсуждению результатов проходят каждую неделю в четверг в 22:00 по МСК

- [.] 0. Постановка задачи
- [.] 0.1 Сбор функциональных, нефункциональных требований
- [.] 0.2 Формулировка актуальности, целей проекта
- [.] 0.3 Подготовка списка литературы/источников для анализа
- [.] 0.4 Обзор источников данных
- [] 1. EDA
- [] 1.1 Проведено исследование документации источников данных
- [] 1.2 Написан первичный скрипт сбора данных с OpenDota
- [] 1.3 Проведен анализ winRate в разрезе матчей/игроков/персонажей. Оценка среди всех матчей, матчей профессиональных игроков
- [] 1.4 Проведен анализ экономики (золото) в разрезе матчей/игроков/персонажей. Оценка в среднем, оценка в динамике
- [] 1.5 Проведен анализ метовых персонажей/стратегий
- [] 1.6 Проведен анализ статистик убийств/смертей в разрезе персонажей
- [] 2. Постановка ML задачи
- [] 2.1 Принято решение о разметке данных: что для нас является таргетом
- [] 2.2 Принято решение о применении ML-модели: realtime или оффлайн
- [] 2.3 Принято решение о бизнес-метриках оценки качества решения. Проведен анализ метрик
- [] 2.4 Принято решение о ML-метриках оценки качества решения.
- [] 3. Построение модели классического ML
- [] 3.1 Построение базового не ML алгоритма - статистический подход
- [] 3.2 Построение ML-модели / ансамбля моделей / эвристик + модели
- [] 3.3 Перебор оптимальных параметров модели
- [] 3.4 Fallback моделей, реализация решения при отсутствии каких-либо данных
- [] 3.5 Написаны тесты для моделей
- [] 4. Построение модели DL
- [] 4.1 Перебор архитектур, гиперпараметров, параметров обучения
- [] 4.2 Оценка качества на большой модели
- [] 4.3 Дистилляция модели, подготовка к применению
