def convert_time(secondes):
    heures = int(secondes // 3600)
    reste = secondes % 3600
    minutes = int(reste // 60)
    secondes_restantes = int(reste % 60)

    parts = []
    if heures > 0:
        parts.append(f"{heures}h")
    if minutes > 0:
        parts.append(f"{minutes}min")
    parts.append(f"{secondes_restantes}s")

    return " ".join(parts)
