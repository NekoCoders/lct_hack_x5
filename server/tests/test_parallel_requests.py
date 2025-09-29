import asyncio, httpx

URL = "http://127.0.0.1:8000/api/predict"
PARALLEL_REQUESTS_NUMBER = 10

async def go(i):
    async with httpx.AsyncClient() as c:
        r = await c.post(
            URL,
            json={"input": f"молоко домик в деревне очень вкусное"}
        )
        return r.json()

async def main():
    tasks = [asyncio.create_task(go(i)) for i in range(PARALLEL_REQUESTS_NUMBER)]
    await asyncio.gather(*tasks)

asyncio.run(main())
