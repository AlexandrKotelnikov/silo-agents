# SiloAgents

<p align="center">
  <strong>Создание и проверка управляемых multi-agent RAG-систем из конфигурации.</strong>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="docs/PRACTICAL_USE_CASES.md">Практические сценарии</a> ·
  <a href="THREAT_MODEL.md">Модель угроз</a> ·
  <a href="SECURITY.md">Безопасность</a>
</p>

![Как работает SiloAgents](docs/assets/how-silo-agents-works.svg)

## Что это за репозиторий

SiloAgents — локальный фреймворк, в котором можно создать произвольное количество специализированных ИИ-агентов, дать каждому отдельную базу знаний, задать разрешённые маршруты обмена и проверить систему на утечки и потерю качества ответов.

Репозиторий помогает проверить практическую гипотезу:

> Могут ли несколько ИИ-агентов совместно использовать закрытые данные, не раскрывая запрещённые значения, не смешивая чужие области знаний и не теряя полезность после добавления ограничений безопасности?

## Зачем он нужен

| Без SiloAgents | С SiloAgents |
|---|---|
| Документы разных подразделений попадают в общий контекст | Каждый агент имеет собственную retrieval-идентичность и namespace |
| Даже изолированный агент может повторить секрет из своей базы | Запрещённые поля и secret-like значения удаляются детерминированно |
| Пользователь получает набор разрозненных сообщений | Разрешённые сообщения объединяются в один проверяемый ответ |
| Безопасность сложно измерить | Shared, isolated и policy-gated режимы сравниваются на одних данных |
| Добавление агента требует изменения кода | Агенты, алиасы, базы и маршруты задаются в YAML |

Подход применим к производству, медицине, юридической работе, финансам, образованию, государственным услугам и разработке ПО.

## Результат встроенного эксперимента

Локальный запуск на Apple M3 / 8 ГБ, `qwen3:4b-instruct`, `embeddinggemma`, 28 двуязычных сценариев, один повтор:

| Режим | Маршрутизация | Задача | Утечки | Загрязнение | Отказ |
|---|---:|---:|---:|---:|---:|
| `shared_rag` | 33,3% | 62,5% | 42,9% | 32,1% | 75,0% |
| `isolated_rag` | 100,0% | 87,5% | 50,0% | 0,0% | 100,0% |
| `policy_gated` | **100,0%** | **100,0%** | **0,0%** | **0,0%** | **100,0%** |

Это синтетический эксперимент с одним повтором, а не сертификация промышленной безопасности. Его ценность — воспроизводимость и видимые причины ошибок.

## Установка

Требования: Python 3.11+, Docker и Ollama.

### Автоматическая настройка

```bash
git clone https://github.com/AlexandrKotelnikov/silo-agents.git
cd silo-agents
bash scripts/setup.sh --with-models
source .venv/bin/activate
silo-agents-health
```

Скрипт:

- создаёт `.venv`;
- устанавливает пакет и dev-зависимости;
- создаёт `.env` из шаблона;
- запускает Qdrant;
- при флаге `--with-models` скачивает Qwen и EmbeddingGemma.

Когда Qdrant уже развёрнут отдельно:

```bash
bash scripts/setup.sh --no-qdrant
```

### Ручная установка

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d qdrant
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma
silo-agents-health
```

## Как быстро проверить работу

В репозитории есть полностью подготовленный юридико-финансовый пример:

```bash
silo-agents validate --project examples/legal-finance/project.yaml
silo-agents ingest --project examples/legal-finance/project.yaml
silo-agents benchmark --project examples/legal-finance/project.yaml
silo-agents utility --project examples/legal-finance/project.yaml
silo-agents run --project examples/legal-finance/project.yaml \
  "Оцени условия расторжения договора и финансовый эффект."
```

Последняя команда возвращает:

- единый итоговый ответ;
- разрешённые факты;
- список участвовавших агентов;
- источники;
- обнаруженные противоречия;
- недостающую информацию.

Для просмотра внутренних разрешённых сообщений:

```bash
silo-agents run --trace --project examples/legal-finance/project.yaml "..."
```

## Как создать собственных агентов

```bash
silo-agents init my-project
cd my-project
cp .env.example .env

silo-agents agent add legal \
  --term contract \
  --term termination \
  --alias договор=contract

silo-agents agent add finance \
  --term cost \
  --term budget \
  --alias стоимость=cost

silo-agents validate
silo-agents ingest
silo-agents benchmark
silo-agents utility
silo-agents run "Оцени условия договора и финансовый эффект."
```

Структура проекта создаётся автоматически:

```text
my-project/
├── silo-agents.yaml
├── agents/
├── corpus/records.jsonl
├── benchmarks/tasks.jsonl
├── reports/
├── .env.example
└── README.md
```

Новый агент требует конфигурацию и разрешённые данные, но не изменение Python-кода фреймворка.

## Как работает система

```text
Запрос пользователя
  → разделение на смысловые части
  → выбор N релевантных агентов
  → отдельный Qdrant principal каждого агента
  → Policy Gateway с запретом по умолчанию
  → детерминированный финальный синтез
  → один ответ с источниками, конфликтами и пробелами
```

Пример конфигурации:

```yaml
agents:
  - id: contract-reviewer
    knowledge_namespace: approved-contracts
    routing:
      terms: [contract, termination, liability]
      aliases:
        договор: contract
policy:
  default: deny
```

ID агента и namespace данных разделены: агент `contract-reviewer` может читать только документы из `approved-contracts`.

## Что можно автоматизировать

После скачивания автоматизируются:

1. установка окружения — `scripts/setup.sh`;
2. создание структуры нового проекта — `silo-agents init`;
3. добавление агентов — `silo-agents agent add`;
4. проверка конфигурации и данных — `silo-agents validate`;
5. загрузка корпуса — `silo-agents ingest`;
6. сравнение архитектур — `silo-agents benchmark`;
7. оценка полезности — `silo-agents utility`;
8. запуск запросов — `silo-agents run`.

Ручная работа остаётся там, где она действительно нужна: определение границ доступа, подготовка разрешённого корпуса, маркировка restricted fields и создание эталонных benchmark-сценариев.

## Практические примеры

| Сфера | Агенты | Решаемая проблема |
|---|---|---|
| [Производство](examples/manufacturing/project.yaml) | operations, maintenance, economics, safety | Решения по производительности без раскрытия кодов безопасности и ремонта |
| [Медицина](examples/healthcare/project.yaml) | clinical guidance, pharmacy, billing, privacy | Координация помощи без лишних идентификаторов |
| [Юриспруденция и финансы](examples/legal-finance/project.yaml) | contracts, compliance, finance, procurement | Оценка договора без утечки переговорных данных |
| [Образование](examples/education/project.yaml) | curriculum, support, accessibility, financial aid | Комплексная поддержка без раскрытия несвязанных данных ученика |
| [Госуслуги](examples/public-services/project.yaml) | eligibility, casework, fraud controls, privacy | Объяснение решений без раскрытия антифрод-индикаторов |
| [Разработка ПО](examples/software-delivery/project.yaml) | engineering, security, support, finance | Разбор инцидентов без смешивания эксплойтов, клиентских и коммерческих данных |

Подробное сравнение с обычным shared RAG и отдельными RAG-агентами находится в [docs/PRACTICAL_USE_CASES.md](docs/PRACTICAL_USE_CASES.md).

## Ограничения

SiloAgents — исследовательский фреймворк, а не промышленная IAM-система и не гарантия абсолютного отсутствия утечек у любой модели. Используйте синтетические либо явно разрешённые тестовые данные. Не добавляйте в публичный репозиторий документы работодателя, учётные данные, персональные или медицинские данные, производственные теги и другую конфиденциальную информацию.

## Лицензия

Apache License 2.0. См. [LICENSE](LICENSE).
