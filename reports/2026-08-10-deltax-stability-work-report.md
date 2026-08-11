# Отчёт о работе: устойчивость CFM к сдвигу центроидов

Дата: 2026-08-10.

## Итог

Оба требуемых ΔX′ получены на 315 парных симуляциях. При `dG=0` фактический входной сдвиг 4.566 мм дал ΔX′ CFM 6.200 мм против 5.85 мм у RAP MUSIC; при `dG!=0` сдвиг 4.557 мм дал ΔX′ CFM 6.333 мм против 6.13 мм. В терминах вопроса Осадчего малые изменения входа порождают изменения оценки того же порядка, но CFM усиливает их в среднем в 1.39–1.44 раза и немного уступает RAP MUSIC. Важная оговорка: ранее заявленные MAE geo 3.6–3.8 мм были получены oracle best-of-25 с выбором posterior draw по известной истине, а честный posterior mean этого чекпоинта имеет существенно худший baseline; основной эксперимент сохранён в заданной постановке posterior mean по 25 draws.

## Главная таблица

| Сценарий | ΔX вход, мм | ΔX′ RAP MUSIC, мм | ΔX′ CFM, мм | CFM − RAP, мм | Среднее ΔX′/ΔX |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dG = 0` | 4.5663 | 5.85 | **6.1995** | +0.3495 | 1.3913 |
| `dG != 0` | 4.5567 | 6.13 | **6.3328** | +0.2028 | 1.4361 |

Основные числа евклидовы. Дополнительная средняя mesh-геодезика ΔX′ равна 12.914 мм (`dG=0`, 297 конечных строк) и 13.131 мм (`dG!=0`, 296 конечных строк).

## Что реально сделано

### 1. Подготовка данных

С публичного Яндекс.Диска скачаны:

- `/mnt/local/tester/meg/data_anna_shift/anna_dg0.npz`, 2,133,468,311 байт;
- `/mnt/local/tester/meg/data_anna_shift/anna_dgnz.npz`, 2,133,468,311 байт;
- `/mnt/local/tester/meg/data_anna_shift/Several_source_simulations_upd_07_05_2026.py`, 39,990 байт.

Код генерации по ссылке оказался одиночным `.py`, а не ZIP; ему возвращено имя из метаданных Яндекс.Диска. SHA-256 и полная инвентаризация записаны в `/mnt/local/tester/meg/data_anna_shift/DATASET_NOTES.md`.

Проверено фактически:

- по 315 строк в каждом датасете;
- формы `Ys` и `Ys_shifted`: `(315, 102, 1800)`;
- `vertices`, `edge_index`, `edge_weight_mm` побайтово совпадают с `best_examples/test_dataset_mesh.npz`;
- в `dG=0` одна уникальная `G_for_simulation`, в `dG!=0` — 315;
- сохранённый ΔX совпадает с пересчётом по координатам вершин;
- генератор переиспользует внутри пары временную активацию, lead field и шум, меняя spatial activation;
- в каждом датасете есть одна пара с нулевым сдвигом всех трёх источников.

Сырые результаты в `/mnt/local/tester/meg/reports/shift_paired_setup_20260810_135708/dataset_validation.json`.

### 2. Git и изоляция

Изначально `/mnt/local/tester/meg` не был Git-репозиторием. По команде пользователя выполнено:

- `git init -b codex/deltax-stability`;
- создан `.gitignore`, исключающий `.env`, датасеты, чекпоинты, БД, логи, модели и generated reports;
- исходный код и текстовые файлы зафиксированы коммитом `32e7285 Initialize MEG experiment repository`;
- создан worktree `/mnt/local/tester/meg/.worktrees/deltax-compute` на ветке `codex/deltax-compute`;
- тяжёлые игнорируемые датасеты и чекпоинты подключены в worktree символическими ссылками.

После восстановления ссылок полный набор тестов дал `43 passed, 1 warning`.

### 3. Реализация оценки

Созданы:

- `/mnt/local/tester/meg/new_version_mesh_experiments/cnn_shift_stability_paired.py`;
- `/mnt/local/tester/meg/new_version_mesh_experiments/tests/test_cnn_shift_stability_paired.py`;
- `/mnt/local/tester/meg/data_anna_shift/DATASET_NOTES.md`.

Новый скрипт:

- загружает только `best-model-folder/model.eqx` и соответствующие stats/result;
- считает два независимых posterior mean на `Ys` и `Ys_shifted`;
- использует одинаковые ODE draw keys для половин пары;
- проецирует posterior mean на mesh;
- независимо разрешает перестановки относительно двух ground truth;
- считает евклидову и геодезическую ΔX′, обычные localization metrics и ratio;
- пишет `summary.json`, `per_sample_metrics.csv`, `paired_predictions.npz` и `report.md`;
- отделяет JIT compile от steady-state timing.

Перед реализацией были написаны и запущены падающие тесты. Новые тесты покрывают truth-anchored permutation matching, евклидову и геодезическую парную дистанцию, нулевой знаменатель ratio и finite-only распределения.

### 4. Запуски

Smoke, по восемь пар:

- GPU 1: `/mnt/local/tester/meg/reports/shift_paired_smoke_dg0_20260810_141514/`;
- GPU 2: `/mnt/local/tester/meg/reports/shift_paired_smoke_dgnz_20260810_141514/`.

Полные прогоны:

- GPU 1, `dG=0`: `/mnt/local/tester/meg/reports/shift_paired_dg0_20260810_141840/`;
- GPU 2, `dG!=0`: `/mnt/local/tester/meg/reports/shift_paired_dgnz_20260810_141840/`.

Основной цикл 315 пар занял 17.2 и 17.1 секунды соответственно, без учёта загрузки NPZ и JIT/timing warm-up. Использованы 25 posterior draws и 96 ODE steps.

Диагностические fixed-test posterior mean re-eval:

- 25 draws: `/mnt/local/tester/meg/reports/shift_paired_reference_recheck_20260810_141514/`;
- 100 draws: `/mnt/local/tester/meg/reports/shift_paired_reference_recheck_n100_20260810_141514/`.

CPU timing:

- `/mnt/local/tester/meg/reports/shift_paired_cpu_timing_20260810_141840/`.

Steady end-to-end inference для одного `Y`:

- GPU 1: 22.378 ± 0.085 мс;
- GPU 2: 22.221 ± 0.112 мс;
- CPU: 95.099 ± 7.776 мс.

## Принятые решения

### Евклидова метрика — основная

Почему: `real_inputs_delta_m` и числа RAP MUSIC 5.85/6.13 рассчитаны обычной нормой в 3-D. Замена на mesh-геодезику сделала бы сравнение методологически неверным. Геодезика сохранена как дополнительная метрика.

### Перестановки разрешаются через истину

Почему: три выходных слота модели не имеют надёжного общего порядка между двумя инференсами. Сначала prediction A сопоставляется со своей истиной, prediction B — со своей, после чего общий истинный индекс `j` связывает оценки одного источника.

### Для двух половин пары используются общие random keys

Почему: common random numbers уменьшают вклад чистого Monte Carlo noise в разность и лучше изолируют реакцию модели на изменение `Y`.

### Сохранён posterior mean по 25 draws

Почему: это estimator, явно заданный планом. Oracle best-of-25 требует ground truth, не является реальным локализатором и поэтому не использован даже после обнаружения худшего baseline posterior mean.

### Нулевые строки не удаляются

Почему: это фактический результат снапа, а не повреждённые данные. Единственное отношение с ΔX=0 записано как undefined и исключено только из агрегирования ratio.

### Бесконечные геодезики не заменяются числом

Почему: граф имеет две крупные компоненты-полусферы и одну изолированную вершину. Переход оценки между компонентами действительно не имеет конечного пути; средние геодезики приведены finite-only вместе с количеством `inf`.

## Проблемы и как они решены

### Репозиторий отсутствовал

Создан новый Git-репозиторий. До `git add` проверены объём и список неигнорируемых файлов; `.env` и бинарные артефакты исключены. В начальный коммит вошло около 3.5 МБ, а не 30+ ГБ experiment data.

### Worktree не видел чекпоинты

Первый baseline test run дал `42 passed, 1 failed`: старый тест ожидал исторические `.eqx/.npz`, которые правильно игнорируются Git. Проверено, что все шесть файлов существуют в основной директории. Каталоги подключены ссылками, после чего получено `43 passed` без изменения теста.

### Smoke baseline оказался значительно хуже ожидания

На первых восьми строках MAE geo был 7.49/8.67 мм. Полный fixed-test re-eval тем же рекомендованным posterior-mean скриптом дал 9.48 мм при 25 draws и 9.09 мм при 100 draws.

Корневая причина найдена в старом evaluation protocol: `trial81_fulltest_search.evaluate_split` и `eval_cnn_checkpoints_25march.evaluate_dataset` перебирают posterior draws и оставляют draw с минимальной ошибкой относительно ground truth. Именно такой oracle estimator породил headline 3.6–3.8 мм. Это не ошибка загрузки чекпоинта: SHA-256 модели/stats/result в `best-model-folder` и исходной `trial81_push_09` совпадают.

Решение: не подменять estimator. Полные paired runs выполнены с posterior mean, расхождение явно вынесено в научный и рабочий отчёты.

### Геодезические значения `inf`

Проверка connected components показала размеры `2336`, `1`, `2362`. Для cross-component predicted shifts оставлены `inf`; отчёт показывает finite-only статистику и число пропусков.

## Что не сошлось

- Ожидаемый baseline MAE geo 3.6–4.6 мм не воспроизвёлся для posterior mean. На данных Анны finite-only baseline равен 7.68–8.13 мм; на fixed test — 9.48 мм при 25 draws.
- Причина не в новом paired pipeline, а в несопоставимости estimator: прежняя цифра oracle best-of-N.
- Фактический ΔX равен 4.566 и 4.557 мм, а не ровно 4.54 мм.
- Датасеты оказались не маленькими: по 2.13 ГБ каждый из-за 315 полных `G_for_simulation` в `float64`.
- `meta.n_samples` внутри NPZ равно 10000 и унаследовано от базового набора; реальное число строк равно 315.
- В каждом наборе одна пара полностью не сдвинулась после снапа; у многих строк есть нулевой сдвиг отдельного источника.
- Геодезическая ΔX′ имеет 18/19 бесконечных средних из-за несвязного mesh.

## Артефакты

Научный отчёт:

- `/mnt/local/tester/meg/reports/shift_paired_report_20260810_141840.md`

Объединённые данные:

- `/mnt/local/tester/meg/reports/shift_paired_per_sample_metrics_20260810_141840.csv` — 630 строк;
- `/mnt/local/tester/meg/reports/shift_paired_summary_20260810_141840.json`.

Основные сырые прогоны:

- `/mnt/local/tester/meg/reports/shift_paired_dg0_20260810_141840/`;
- `/mnt/local/tester/meg/reports/shift_paired_dgnz_20260810_141840/`;
- `/mnt/local/tester/meg/reports/shift_paired_cpu_timing_20260810_141840/`.

Данные и их описание:

- `/mnt/local/tester/meg/data_anna_shift/`;
- `/mnt/local/tester/meg/data_anna_shift/DATASET_NOTES.md`.

Код:

- `/mnt/local/tester/meg/new_version_mesh_experiments/cnn_shift_stability_paired.py`;
- `/mnt/local/tester/meg/new_version_mesh_experiments/tests/test_cnn_shift_stability_paired.py`.

## Что дальше

1. Согласовать с Анной, относятся ли 5.85/6.13 к тем же 315 строкам и как RAP MUSIC обрабатывал одну нулевую пару и перестановки.
2. Обсудить единый не-oracle estimator для сравнения качества локализации; переименовать старые headline metrics как `oracle best-of-N`, чтобы не смешивать их с posterior mean.
3. Проверить permutation-aligned posterior mean или posterior medoid: усреднение draws до согласования источников может быть одной из причин слабого baseline.
4. Посчитать paired bootstrap confidence intervals для разницы CFM − RAP MUSIC, если Анна отдаст per-sample RAP metrics.
5. Разобрать 18–21 cross-component случаев и отдельно посчитать стабильность без hemisphere swaps, не скрывая исходную метрику.
