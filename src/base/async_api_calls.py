import asyncio
import aiohttp

async def fetch_data(session, url, data):
    async with session.post(url, data=data) as response:
        return await response.text()

async def main():
    url = "https://your-api-endpoint.com"
    data = [{"arg1": "value1"}, {"arg2": "value2"}, ...]  # List of different arguments

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url, d) for d in data]
        results = await asyncio.gather(*tasks)

    # Process the results as needed

if __name__ == "__main__":
    asyncio.run(main())