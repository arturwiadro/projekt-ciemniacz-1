import argparse
import os
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import cv2
import pandas as pd
import supervision as sv
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="Zliczanie osób i pojazdów z kamer (YOLO + ByteTrack)."
    )
    p.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            'Źródło: "0" (kamera lokalna), URL strumienia (rtsp/http/m3u8), '
            'plik video.mp4 lub adres strony z osadzoną kamerą (spróbuje użyć yt-dlp).'
        ),
    )
    p.add_argument(
        "--stream_url",
        type=str,
        default="",
        help="Opcjonalnie: bezpośredni URL streamu (np. m3u8/rtsp). Nadpisuje --source.",
    )
    p.add_argument(
        "--resolve_only",
        type=int,
        default=0,
        help="1=tylko pokaż wykryty URL streamu i zakończ (pomocne do testu linku)",
    )
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Model YOLO")
    p.add_argument("--conf", type=float, default=0.35, help="Próg pewności detekcji")
    p.add_argument("--imgsz", type=int, default=640, help="Rozmiar wejścia YOLO")
    p.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Pomijanie klatek: 0=każda, 1=co druga, 2=co trzecia...",
    )
    p.add_argument("--line_x", type=int, default=320, help="Pionowa linia zliczania (x w px)")
    p.add_argument(
        "--deadzone",
        type=int,
        default=20,
        help="Martwa strefa +/- px wokół linii",
    )
    p.add_argument(
        "--show",
        type=int,
        default=1,
        help="1=pokazuj podgląd, 0=bez okna (headless)",
    )
    p.add_argument(
        "--interval_min",
        type=int,
        default=15,
        help="Co ile minut zapisywać wynik (domyślnie 15)",
    )
    p.add_argument(
        "--out_csv", type=str, default="live_15min.csv", help="CSV z agregacją interwałową"
    )
    p.add_argument(
        "--out_xlsx",
        type=str,
        default="live_15min.xlsx",
        help="Excel z agregacją interwałową (ustaw pusty string, aby wyłączyć)",
    )
    p.add_argument(
        "--reset_each_interval",
        type=int,
        default=1,
        help="1=zeruj liczniki po zapisie interwału, 0=narastająco",
    )
    return p.parse_args()


def side_with_deadzone(x_center: float, x_line: int, deadzone: int) -> Optional[str]:
    if x_center < x_line - deadzone:
        return "L"
    if x_center > x_line + deadzone:
        return "R"
    return None




def _extract_stream_from_html(page_url: str) -> Optional[str]:
    """Fallback: próbuje znaleźć URL streamu bezpośrednio w HTML strony."""
    req = Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=15) as resp:  # nosec - user provided URL
        html = resp.read().decode("utf-8", errors="ignore")

    # Najpierw szukamy bezpośrednich URL-i streamu
    media_patterns = [
        r'https?://[^"\'\s>]+\.m3u8[^"\'\s>]*',
        r'https?://[^"\'\s>]+\.mp4[^"\'\s>]*',
        r'rtsp://[^"\'\s>]+',
    ]
    for pattern in media_patterns:
        m = re.search(pattern, html, flags=re.IGNORECASE)
        if m:
            return m.group(0)

    # Często stream jest w iframe - spróbujmy wejść poziom głębiej
    iframe_match = re.search(r'<iframe[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if iframe_match:
        iframe_url = urljoin(page_url, iframe_match.group(1))
        req_iframe = Request(iframe_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req_iframe, timeout=15) as resp:  # nosec - user provided URL
            iframe_html = resp.read().decode("utf-8", errors="ignore")

        for pattern in media_patterns:
            m = re.search(pattern, iframe_html, flags=re.IGNORECASE)
            if m:
                return m.group(0)

    return None
def resolve_webpage_to_stream(url: str) -> str:
    """Próbuje zamienić URL strony (np. z osadzoną kamerą) na bezpośredni URL streamu."""
    info = None
    ytdlp_error = None

    try:
        from yt_dlp import YoutubeDL

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extract_flat": False,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        ytdlp_error = exc

    if isinstance(info, dict):
        if info.get("url"):
            return info["url"]

        formats = info.get("formats") or []
        for f in formats:
            candidate = f.get("url")
            if not candidate:
                continue
            protocol = (f.get("protocol") or "").lower()
            ext = (f.get("ext") or "").lower()
            if "m3u8" in protocol or ext in {"mp4", "m3u8", "ts"}:
                return candidate

    fallback = _extract_stream_from_html(url)
    if fallback:
        print("[INFO] yt-dlp nie obsłużył linku. Używam fallback HTML.")
        return fallback

    if ytdlp_error is not None:
        raise RuntimeError(
            "Nie udało się odczytać streamu ze strony przez yt-dlp ani fallback HTML. "
            "Podaj bezpośredni URL przez --stream_url (np. .m3u8). "
            f"Szczegóły yt-dlp: {ytdlp_error}"
        ) from ytdlp_error

    raise RuntimeError(f"Nie znalazłem bezpośredniego URL streamu dla: {url}")


def open_source(src: str) -> cv2.VideoCapture:
    """Otwiera już-rozwiązane źródło (kamera index / plik / bezpośredni URL streamu)."""
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def resolve_source_url(args) -> str:
    """Zwraca finalne źródło do OpenCV: stream_url, kamera lokalna albo URL/pliki."""
    if args.stream_url:
        return args.stream_url

    if args.source.isdigit():
        return args.source

    if args.source.startswith(("http://", "https://")) and not args.source.lower().endswith(
        (".m3u8", ".mp4", ".ts", ".avi", ".mov")
    ):
        resolved = resolve_webpage_to_stream(args.source)
        print(f"[INFO] Wykryto stronę WWW. Używam streamu: {resolved}")
        return resolved

    return args.source


def init_csv(path: str):
    if not os.path.exists(path):
        pd.DataFrame(
            [
                {
                    "timestamp_start": "",
                    "timestamp_end": "",
                    "ped_in": 0,
                    "ped_out": 0,
                    "veh_in": 0,
                    "veh_out": 0,
                    "ped_total": 0,
                    "veh_total": 0,
                }
            ]
        ).head(0).to_csv(path, index=False)


def append_row(csv_path: str, xlsx_path: str, row: dict):
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

    if xlsx_path:
        if os.path.exists(xlsx_path):
            old = pd.read_excel(xlsx_path)
            out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        else:
            out = pd.DataFrame([row])
        out.to_excel(xlsx_path, index=False)


def main():
    args = parse_args()

    print("=== TRAFFIC COUNTER LIVE (preview + 15min CSV/XLSX) ===")
    print(
        f"source={args.source} line_x={args.line_x} deadzone={args.deadzone} "
        f"interval={args.interval_min}min"
    )

    final_source = resolve_source_url(args)

    if args.resolve_only == 1:
        print(f"[RESOLVED_SOURCE] {final_source}")
        return

    cap = open_source(final_source)
    if not cap.isOpened():
        raise RuntimeError(
            "Nie mogę otworzyć źródła wideo. "
            f"source={args.source} final_source={final_source}. "
            "Spróbuj podać bezpośredni stream przez --stream_url."
        )

    model = YOLO(args.model)

    PERSON = 0
    VEHICLES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    TARGET = {PERSON} | VEHICLES

    tracker = sv.ByteTrack()

    last_side_ped = {}
    last_side_veh = {}

    ped_in = ped_out = 0
    veh_in = veh_out = 0

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

    interval_s = args.interval_min * 60
    interval_start = time.time()
    interval_start_dt = datetime.now()

    init_csv(args.out_csv)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        frame_idx += 1
        if args.skip > 0 and (frame_idx % (args.skip + 1) != 0):
            continue

        h, w = frame.shape[:2]
        x_line = max(0, min(args.line_x, w - 1))

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        det = sv.Detections.from_ultralytics(results)

        if det.class_id is not None and len(det) > 0:
            mask = [int(cid) in TARGET for cid in det.class_id]
            det = det[mask]

        det = tracker.update_with_detections(det)

        if len(det) > 0 and det.tracker_id is not None:
            for bbox, cid, tid in zip(det.xyxy, det.class_id, det.tracker_id):
                cid = int(cid)
                tid = int(tid)
                x_center = (bbox[0] + bbox[2]) / 2.0
                side = side_with_deadzone(x_center, x_line, args.deadzone)
                if side is None:
                    continue

                if cid == PERSON:
                    prev = last_side_ped.get(tid)
                    if prev is not None and prev != side:
                        if prev == "L" and side == "R":
                            ped_in += 1
                        elif prev == "R" and side == "L":
                            ped_out += 1
                    last_side_ped[tid] = side
                elif cid in VEHICLES:
                    prev = last_side_veh.get(tid)
                    if prev is not None and prev != side:
                        if prev == "L" and side == "R":
                            veh_in += 1
                        elif prev == "R" and side == "L":
                            veh_out += 1
                    last_side_veh[tid] = side

        if args.show == 1:
            labels = []
            if len(det) > 0 and det.tracker_id is not None:
                for cid, tid, conf in zip(det.class_id, det.tracker_id, det.confidence):
                    labels.append(f"{model.names.get(int(cid), str(cid))} #{int(tid)} {conf:.2f}")

            annotated = frame.copy()
            annotated = box_annotator.annotate(annotated, det)
            annotated = label_annotator.annotate(annotated, det, labels=labels)

            cv2.line(annotated, (x_line, 0), (x_line, h - 1), (255, 255, 255), 2)
            cv2.line(
                annotated,
                (max(0, x_line - args.deadzone), 0),
                (max(0, x_line - args.deadzone), h - 1),
                (255, 255, 255),
                1,
            )
            cv2.line(
                annotated,
                (min(w - 1, x_line + args.deadzone), 0),
                (min(w - 1, x_line + args.deadzone), h - 1),
                (255, 255, 255),
                1,
            )

            cv2.putText(
                annotated,
                f"PED IN:{ped_in} OUT:{ped_out}   VEH IN:{veh_in} OUT:{veh_out}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            left = max(0, int(interval_s - (time.time() - interval_start)))
            cv2.putText(
                annotated,
                f"save in: {left}s -> {args.out_csv} / {args.out_xlsx}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "ESC=quit | [ / ] line_x | i/k ped_in +/- | o/l veh_in +/-",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Traffic Counter LIVE", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("["):
                args.line_x = max(0, args.line_x - 10)
            elif key == ord("]"):
                args.line_x += 10
            elif key == ord("i"):
                ped_in += 1
            elif key == ord("k"):
                ped_in = max(0, ped_in - 1)
            elif key == ord("o"):
                veh_in += 1
            elif key == ord("l"):
                veh_in = max(0, veh_in - 1)

        now = time.time()
        if now - interval_start >= interval_s:
            end_dt = datetime.now()
            row = {
                "timestamp_start": interval_start_dt.isoformat(timespec="seconds"),
                "timestamp_end": end_dt.isoformat(timespec="seconds"),
                "ped_in": int(ped_in),
                "ped_out": int(ped_out),
                "veh_in": int(veh_in),
                "veh_out": int(veh_out),
                "ped_total": int(ped_in + ped_out),
                "veh_total": int(veh_in + veh_out),
            }
            append_row(args.out_csv, args.out_xlsx, row)
            print("[SAVE]", row)

            interval_start = now
            interval_start_dt = end_dt
            if args.reset_each_interval == 1:
                ped_in = ped_out = 0
                veh_in = veh_out = 0
                last_side_ped.clear()
                last_side_veh.clear()

    cap.release()
    if args.show == 1:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
