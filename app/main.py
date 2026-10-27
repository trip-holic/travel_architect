import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import openai

# 1. Настройка базового логирования (Observability)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Travel Architect")

# Проверяем наличие ключа при старте, чтобы не узнать об этом только при запросе
#api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("GROG_API_KEY")
if not api_key:
    logger.error("КРИТИЧЕСКАЯ ОШИБКА: API_KEY не найден в окружении!")

# Инициализируем клиента OpenAI с защитой от зависаний (Resilience)
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    timeout=30.0,       # Ждем максимум 30 секунд
    max_retries=2       # Автоматически делаем 2 повторные попытки при сетевых сбоях
)

# Описываем модель входящего запроса
class TravelRequest(BaseModel):
    destination: str = "Маврикий"
    days: int = 6
    vibe: str = "upscale, hidden gems, уединенность, необычный гастрономический опыт"

@app.get("/")
async def root():
    return {"message": "Сервер работает. ИИ готов к приему запросов."}

@app.post("/api/v1/generate")
async def generate_itinerary(req: TravelRequest):
    prompt = f"Создай концепт нестандартного путешествия. Место: {req.destination}. Дней: {req.days}. Фокус и настроение: {req.vibe}."
    logger.info(f"Начало обработки запроса для локации: {req.destination}")

    try:
        response = await client.chat.completions.create(
            #model="gpt-4o-mini", # Используем быструю и недорогую модель для тестов
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system", 
                    "content": "Ты — премиальный travel-архитектор. Твоя задача — создавать эксклюзивные, логичные и насыщенные маршруты без банальных туристических клише. Отвечай структурированно."
                },
                {"role": "user", "content": prompt}
            ]
        )
        logger.info(f"Ответ от OpenAI успешно получен. Использовано токенов: {response.usage.total_tokens}")
        # Возвращаем текст, который сгенерировала нейросеть
        return {"concept": response.choices[0].message.content}
        
# 3. Гранулярная обработка ошибок
    except openai.APITimeoutError:
        logger.error("Таймаут: OpenAI не ответил за 30 секунд.")
        raise HTTPException(status_code=504, detail="Сервер ИИ перегружен и не ответил вовремя.")
    except openai.APIConnectionError:
        logger.error("Ошибка сети: Невозможно подключиться к OpenAI.")
        raise HTTPException(status_code=502, detail="Сбой сети при обращении к ИИ-провайдеру.")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера генерации.")