import asyncpg
import asyncio

async def fetch_doctors_with_user():
    conn = await asyncpg.connect(
        "postgres://avnadmin:AVNS_JyHETleS33sI01tOgCM@pg-2e5df4d0-doctor-app-new.h.aivencloud.com:15652/defaultdb?sslmode=require"
    )
    
    query = """
    SELECT u.name AS doctor_name, d.ratings AS rating, d.specialization
    FROM "Doctor" d
    JOIN "User" u ON d."userId" = u.id
    LIMIT 10;
    """
    
    rows = await conn.fetch(query)
    
    result = []
    for row in rows:
        result.append({
            "name": row['doctor_name'],
            "rating": row['rating'],
            "Specialization": row['specialization']
        })
    
    await conn.close()
    return result


# print(asyncio.run(fetch_doctors_with_user()))