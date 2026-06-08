import os
from pathlib import Path
from datetime import datetime, UTC

import requests

# =========================
# CONFIG
# =========================

API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = [
    "Paris",
    "London",
    "New York",
    "Tokyo",
    "Sydney"
]

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Chemin absolu vers le dossier du script
SCRIPT_DIR = Path(__file__).resolve().parent
README_PATH = SCRIPT_DIR / "README.md"

# =========================
# CHECK API KEY
# =========================

if not API_KEY:
    raise ValueError("ERREUR : OPENWEATHER_API_KEY non définie !")

# =========================
# FETCH WEATHER
# =========================

def get_weather(city: str) -> str:
    try:
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric",
            "lang": "fr"
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        temp = data["main"]["temp"]

        return f"{city}: {temp:.1f}°C"

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur réseau/API pour {city} : {e}")
        return f"{city}: erreur"

    except KeyError as e:
        print(f"❌ Réponse API inattendue pour {city} : clé manquante {e}")
        return f"{city}: erreur"

    except Exception as e:
        print(f"❌ Exception pour {city} : {e}")
        return f"{city}: erreur"

# =========================
# GENERATE README
# =========================

def generate_readme(weather_data: list[str]) -> None:
    now = datetime.now(UTC).strftime("%d/%m/%Y à %H:%M UTC")

    content = f"# 🌍 Weather Dashboard\n\n"
    content += f"## 📅 Dernière mise à jour\n{now}\n\n"
    content += "## 🌡️ Températures actuelles\n\n"

    for line in weather_data:
        content += f"- {line}\n"

    content += "\n---\n"
    content += "_Mise à jour automatique via GitHub Actions._\n"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

# =========================
# MAIN
# =========================

def main() -> None:
    print("Récupération des données météo...")

    weather_results = []

    for city in CITIES:
        result = get_weather(city)
        print(result)
        weather_results.append(result)

    print("Génération du README...")
    generate_readme(weather_results)

    print(f"README.md mis à jour avec succès : {README_PATH}")

# =========================

if __name__ == "__main__":
    main()