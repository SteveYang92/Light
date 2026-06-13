def seconds_to_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 0
        secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def srt_to_seconds(tc: str) -> float:
    hms, ms = tc.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def seconds_to_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 0
        secs += 1
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"


def seconds_to_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 0
        secs += 1
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def duration_to_srt(seconds: float) -> str:
    return seconds_to_srt(seconds)
