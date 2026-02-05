# projekt-ciemniacz-1

## YOLO traffic counter (osoby + pojazdy)

Skrypt `traffic_counter.py`:
- zlicza osoby i pojazdy (car/motorcycle/bus/truck),
- działa na kamerze lokalnej, RTSP, URL streamu, pliku wideo, oraz URL strony z osadzonym streamem (przez `yt-dlp`),
- zapisuje wyniki co zadany interwał (domyślnie 15 min) do CSV i XLSX,
- ma podgląd live i proste korekty ręczne liczników.

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

### 1) Lokalna kamera
```bash
python traffic_counter.py --source 0 --show 1
```

### 2) URL strony z kamerą
```bash
python traffic_counter.py --source "https://dktr.pl/Ostroda/" --show 1 --line_x 650
```

### 2a) Jak sprawdzić, czy link strony daje się odczytać
Najpierw przetestuj samo wykrycie URL streamu (bez uruchamiania YOLO):
```bash
python traffic_counter.py --source "https://dktr.pl/Ostroda/" --resolve_only 1
```
Jeśli zobaczysz `[RESOLVED_SOURCE] ...`, to ten URL powinien działać jako stream.

Gdyby strona nie dawała się automatycznie odczytać, podaj stream ręcznie:
```bash
python traffic_counter.py --stream_url "https://...m3u8" --show 1
```

### 3) RTSP
```bash
python traffic_counter.py --source "rtsp://user:pass@ip:554/stream1" --show 1
```


### Gdy zobaczysz `Unsupported URL`
To znaczy, że `yt-dlp` nie ma gotowego ekstraktora dla tej strony. Skrypt ma teraz fallback HTML
(i często sam znajdzie `m3u8/mp4`), ale jeśli nadal się nie uda, użyj ręcznie:
```bash
python traffic_counter.py --stream_url "https://...m3u8" --show 1
```

## Korekty podczas podglądu
- `[` / `]` – przesuń linię zliczania w lewo/prawo,
- `i` / `k` – zwiększ/zmniejsz `ped_in`,
- `o` / `l` – zwiększ/zmniejsz `veh_in`,
- `ESC` – zakończ.

## Zapis wyników
Domyślnie co 15 minut zapis do:
- `live_15min.csv`
- `live_15min.xlsx`

Przykład zmiany interwału i nazw plików:
```bash
python traffic_counter.py \
  --source "https://dktr.pl/Ostroda/" \
  --interval_min 15 \
  --out_csv "ruch_15min.csv" \
  --out_xlsx "ruch_15min.xlsx"
```
