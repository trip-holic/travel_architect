import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI(title="Travel Architect")

# Инициализируем клиента OpenAI (он сам найдет ключ OPENAI_API_KEY из .env)
client = AsyncOpenAI()

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
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Используем быструю и недорогую модель для тестов
            messages=[
                {
                    "role": "system", 
                    "content": "Ты — премиальный travel-архитектор. Твоя задача — создавать эксклюзивные, логичные и насыщенные маршруты без банальных туристических клише. Отвечай структурированно."
                },
                {"role": "user", "content": prompt}
            ]
        )
        # Возвращаем текст, который сгенерировала нейросеть
        return {"concept": response.choices[0].message.content}
        
    except Exception as e:
        # Если OpenAI недоступен или ключ неверный, выдаем ошибку
        raise HTTPException(status_code=500, detail=str(e))