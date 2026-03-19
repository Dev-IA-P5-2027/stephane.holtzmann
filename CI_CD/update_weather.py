import os
from datetime import datetime

import requests

API_KEY = os.environ.get("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
CITIES = ["Paris", "London", "New York", "Tokyo", "Sydney"]


def get_weather_emoji(condition):
    emojis = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Rain": "🌧️",
        "Drizzle": "🌦️",
        "Thunderstorm": "⛈️",
        "Snow": "❄️",
        "Mist": "🌫️",
        "Fog": "🌫️",
        "Haze": "🌫️",
    }
    return emojis.get(condition, "🌡️")


def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "fr",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        return {
            "city": city,
            "temp": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind": round(data["wind"]["speed"] * 3.6, 1),
            "icon": get_weather_emoji(data["weather"][0]["main"]),
        }
    except Exception as e:
        print(f"Erreur pour {city}: {e}")
        return None


def generate_readme(weather_data):
    now = datetime.utcnow().strftime("%d/%m/%Y à %H:%M UTC")

    readme = f"""# Dashboard Météo - CI/CD 🌤️

> Ce README est mis à jour automatiquement par GitHub Actions.

## Météo actuelle - {now}

| Ville | Météo | Temp | Ressenti | Humidité | Vent |
|------|------|------|----------|----------|------|
"""

    for w in weather_data:
        if w:
            readme += (
                f"| {w['icon']} {w['city']} "
                f"| {w['description'].capitalize()} "
                f"| {w['temp']}°C "
                f"| {w['feels_like']}°C "
                f"| {w['humidity']}% "
                f"| {w['wind']} km/h |\n"
            )

    return readme


if __name__ == "__main__":
    if not API_KEY:
        print("ERREUR : OPENWEATHER_API_KEY non définie !")
        raise SystemExit(1)

    print("Récupération des données météo...")
    weather_data = []

    for city in CITIES:
        data = get_weather(city)
        weather_data.append(data)
        if data:
            print(f"{city}: {data['temp']}°C")

    print("Génération du README...")
    readme_content = generate_readme(weather_data)

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("README.md mis à jour avec succès !")