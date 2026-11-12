"""Таксономия навыков: каноническое имя, категория, синонимы (RU/EN), связанные навыки.

Связанные навыки дают «частичное совпадение» при сопоставлении: например,
PostgreSQL частично закрывает требование MySQL, PyTorch — TensorFlow.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...] = ()
    related: tuple[str, ...] = field(default=())
    description: str = ""


_RAW = [
    # name, category, aliases, related, description (для эмбеддинга)
    ("Python", "language", ("python3", "питон", "пайтон"), ("Go", "Java"), "язык программирования python разработка скриптов"),
    ("Java", "language", ("java se", "джава"), ("Kotlin", "Scala"), "язык программирования java jvm"),
    ("Kotlin", "language", ("котлин",), ("Java",), "язык kotlin android jvm"),
    ("Go", "language", ("golang", "голанг"), ("Python",), "язык go golang backend"),
    ("C++", "language", ("cpp", "c plus plus"), ("C",), "язык c++ системное программирование"),
    ("C", "language", ("язык c", "ansi c"), ("C++",), "язык c низкоуровневый"),
    ("JavaScript", "language", ("js", "javascript es6", "джаваскрипт"), ("TypeScript",), "язык javascript фронтенд"),
    ("TypeScript", "language", ("ts", "тайпскрипт"), ("JavaScript",), "язык typescript типизация фронтенд"),
    ("Scala", "language", ("скала",), ("Java", "Apache Spark"), "язык scala функциональное программирование"),
    ("SQL", "data", ("sql запросы", "язык sql", "t-sql", "pl/sql"), ("PostgreSQL",), "язык запросов sql базы данных"),
    ("R", "language", ("язык r", "rstudio"), ("Python",), "язык r статистика"),
    ("PostgreSQL", "database", ("postgres", "постгрес", "postgresql 15", "psql"), ("MySQL", "SQL"), "реляционная база данных postgresql"),
    ("MySQL", "database", ("mariadb", "майскл"), ("PostgreSQL", "SQL"), "реляционная база данных mysql"),
    ("MongoDB", "database", ("mongo", "монго"), ("Cassandra",), "документная nosql база mongodb"),
    ("Redis", "database", ("редис",), ("Memcached",), "кэш key value хранилище redis"),
    ("ClickHouse", "database", ("кликхаус", "clickhouse db"), ("PostgreSQL",), "колоночная аналитическая база clickhouse"),
    ("Elasticsearch", "database", ("elastic", "эластик", "opensearch", "elk"), ("Solr",), "полнотекстовый поиск elasticsearch"),
    ("Cassandra", "database", ("apache cassandra",), ("MongoDB",), "распределённая nosql база cassandra"),
    ("Apache Kafka", "streaming", ("kafka", "кафка"), ("RabbitMQ",), "брокер сообщений потоковая обработка kafka"),
    ("RabbitMQ", "streaming", ("rabbit", "кролик", "amqp"), ("Apache Kafka",), "очередь сообщений rabbitmq"),
    ("Apache Spark", "bigdata", ("spark", "pyspark", "спарк"), ("Hadoop",), "распределённая обработка данных spark"),
    ("Hadoop", "bigdata", ("hdfs", "хадуп", "mapreduce"), ("Apache Spark",), "экосистема больших данных hadoop"),
    ("Apache Airflow", "bigdata", ("airflow", "эйрфлоу"), ("Luigi",), "оркестрация пайплайнов данных airflow etl dag расписание регламентная загрузка выгрузка данных"),
    ("dbt", "bigdata", ("data build tool",), ("SQL",), "трансформации данных dbt"),
    ("Docker", "devops", ("докер", "docker-compose", "контейнеризация"), ("Kubernetes",), "контейнеры docker образ контейнера упаковка сервиса окружение compose"),
    ("Kubernetes", "devops", ("k8s", "кубернетес", "кубер"), ("Docker",), "оркестрация контейнеров kubernetes кластер деплой helm манифест под"),
    ("Terraform", "devops", ("терраформ",), ("Ansible",), "инфраструктура как код terraform"),
    ("Ansible", "devops", ("ансибл",), ("Terraform",), "управление конфигурацией ansible"),
    ("CI/CD", "devops", ("ci cd", "gitlab ci", "github actions", "jenkins", "непрерывная интеграция"), ("Git",), "непрерывная интеграция и доставка автоматическая сборка выкладка релиз пайплайн мёрж"),
    ("Git", "devops", ("гит", "github", "gitlab"), ("CI/CD",), "система контроля версий git"),
    ("Linux", "devops", ("линукс", "unix", "bash"), (), "операционная система linux администрирование"),
    ("AWS", "cloud", ("amazon web services", "ec2", "s3"), ("GCP", "Azure"), "облако amazon aws"),
    ("GCP", "cloud", ("google cloud", "bigquery"), ("AWS",), "облако google cloud"),
    ("Azure", "cloud", ("microsoft azure",), ("AWS",), "облако microsoft azure"),
    ("Yandex Cloud", "cloud", ("яндекс облако", "yandex.cloud"), ("AWS",), "облако yandex cloud"),
    ("Prometheus", "devops", ("прометеус",), ("Grafana",), "мониторинг метрики prometheus"),
    ("Grafana", "devops", ("графана",), ("Prometheus",), "дашборды мониторинга grafana"),
    ("FastAPI", "backend", ("fast api", "фастапи"), ("Flask", "Django"), "веб фреймворк python fastapi rest"),
    ("Django", "backend", ("джанго", "drf", "django rest framework"), ("Flask", "FastAPI"), "веб фреймворк python django"),
    ("Flask", "backend", ("фласк",), ("FastAPI", "Django"), "микрофреймворк python flask"),
    ("Spring", "backend", ("spring boot", "спринг"), ("Java",), "фреймворк java spring boot"),
    ("REST API", "backend", ("rest", "restful", "рест апи", "http api"), ("gRPC",), "проектирование rest api http эндпоинты openapi спецификация"),
    ("gRPC", "backend", ("grpc", "protobuf"), ("REST API",), "rpc протокол grpc protobuf"),
    ("GraphQL", "backend", ("графкьюэл",), ("REST API",), "язык запросов api graphql"),
    ("Microservices", "architecture", ("микросервисы", "микросервисная архитектура независимые сервисы распил монолита отдельные базы", "microservice"), ("Docker",), "микросервисная архитектура независимые сервисы распил монолита отдельные базы"),
    ("React", "frontend", ("react.js", "reactjs", "реакт"), ("Vue.js",), "библиотека интерфейсов react"),
    ("Vue.js", "frontend", ("vue", "вью"), ("React",), "фреймворк интерфейсов vue"),
    ("HTML/CSS", "frontend", ("html", "css", "верстка", "вёрстка"), ("JavaScript",), "вёрстка html css"),
    ("pandas", "data", ("пандас",), ("NumPy",), "анализ табличных данных pandas"),
    ("NumPy", "data", ("numpy", "нампай"), ("pandas",), "численные вычисления numpy"),
    ("scikit-learn", "ml", ("sklearn", "склерн", "scikit learn"), ("CatBoost",), "классическое машинное обучение scikit-learn"),
    ("CatBoost", "ml", ("катбуст",), ("XGBoost", "LightGBM"), "градиентный бустинг catboost"),
    ("XGBoost", "ml", ("xgb",), ("CatBoost", "LightGBM"), "градиентный бустинг xgboost"),
    ("LightGBM", "ml", ("lgbm", "lightgbm"), ("CatBoost", "XGBoost"), "градиентный бустинг lightgbm"),
    ("PyTorch", "ml", ("torch", "пайторч"), ("TensorFlow",), "глубокое обучение pytorch нейросети"),
    ("TensorFlow", "ml", ("tf", "keras", "тензорфлоу"), ("PyTorch",), "глубокое обучение tensorflow keras"),
    ("Machine Learning", "ml", ("машинное обучение", "ml", "мл"), ("Deep Learning", "Statistics"), "машинное обучение обучение моделей классификация регрессия гиперпараметры признаки"),
    ("Deep Learning", "ml", ("глубокое обучение", "нейронные сети", "нейросети", "dl"), ("Machine Learning", "PyTorch"), "глубокое обучение нейронные сети"),
    ("NLP", "ml", ("обработка естественного языка", "natural language processing", "nlp"), ("LLM", "Deep Learning"), "обработка естественного языка классификация текстов извлечение сущностей документы"),
    ("Computer Vision", "ml", ("компьютерное зрение", "cv", "opencv"), ("Deep Learning",), "компьютерное зрение изображения"),
    ("LLM", "ml", ("large language models", "большие языковые модели", "gpt", "rag", "langchain"), ("NLP",), "большие языковые модели llm rag чат-бот генерация поиск по базе знаний"),
    ("Hugging Face", "ml", ("transformers", "huggingface", "хаггинг фейс"), ("PyTorch", "NLP"), "библиотека transformers hugging face"),
    ("MLOps", "ml", ("mlflow", "mlops", "ml ops", "kubeflow"), ("Docker", "CI/CD"), "эксплуатация ml моделей mlops"),
    ("Statistics", "analytics", ("статистика", "мат статистика", "статистический анализ"), ("A/B Testing",), "математическая статистика гипотезы доверительный интервал распределение выборка"),
    ("A/B Testing", "analytics", ("a/b тесты", "аб тесты", "ab testing", "экспериментирование"), ("Statistics",), "a/b тестирование онлайн эксперименты контрольная группа конверсия прирост метрики"),
    ("Tableau", "analytics", ("табло",), ("Power BI",), "bi визуализация tableau"),
    ("Power BI", "analytics", ("powerbi", "пауэр би"), ("Tableau",), "bi визуализация power bi"),
    ("Excel", "analytics", ("эксель", "ms excel", "google sheets"), (), "электронные таблицы excel"),
    ("Data Analysis", "analytics", ("анализ данных", "data analytics", "аналитика данных"), ("Statistics", "SQL"), "анализ данных аналитические отчёты исследование метрик продукта"),
    ("Unit Testing", "quality", ("юнит тесты", "pytest", "junit", "unit tests", "тестирование кода"), ("CI/CD",), "модульное тестирование автотесты фикстуры моки покрытие кода тестами"),
    ("System Design", "architecture", ("проектирование систем", "system design", "архитектура систем"), ("Microservices",), "проектирование архитектуры высоконагруженного сервиса масштабирование rps отказоустойчивость"),
    ("Agile", "management", ("scrum", "скрам", "kanban", "канбан"), (), "гибкие методологии scrum"),
    ("Project Management", "management", ("управление проектами", "pm", "jira"), ("Agile",), "управление проектами сроки бюджет"),
    ("Team Leadership", "management", ("руководство командой", "тимлид", "team lead", "управление командой"), ("Project Management",), "руководство командой разработчиков подчинённые one-to-one найм ревью группа"),
    ("English", "soft", ("английский", "english b2", "upper-intermediate", "английский язык"), (), "английский язык"),
    ("Communication", "soft", ("коммуникабельность", "коммуникация", "communication skills"), (), "коммуникативные навыки выступления взаимодействие с заказчиками и командами переговоры"),
]

SKILLS: dict[str, Skill] = {n: Skill(n, c, a, r, d) for n, c, a, r, d in _RAW}


def alias_index() -> dict[str, str]:
    """lowercase alias -> canonical name (включая само имя)."""
    idx = {}
    for s in SKILLS.values():
        idx[s.name.lower()] = s.name
        for a in s.aliases:
            idx[a.lower()] = s.name
    return idx


def related(a: str, b: str) -> bool:
    return b in SKILLS.get(a, Skill(a, "")).related or a in SKILLS.get(b, Skill(b, "")).related
