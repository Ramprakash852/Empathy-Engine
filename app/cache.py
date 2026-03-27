# app/cache.py
import hashlib, os
import diskcache

CACHE_DIR = os.getenv('CACHE_DIR', './.audio_cache')
_cache = diskcache.Cache(CACHE_DIR, size_limit=500_000_000)  # 500MB limit

def _make_key(text: str, emotion: str, intensity: float) -> str:
    raw = f'{text}|{emotion}|{intensity:.2f}'
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cached(text, emotion, intensity):
    key = _make_key(text, emotion, intensity)
    return _cache.get(key)   # Returns None if not found

def set_cached(text, emotion, intensity, audio_bytes: bytes):
    key = _make_key(text, emotion, intensity)
    _cache.set(key, audio_bytes, expire=86400)  # 24h TTL
