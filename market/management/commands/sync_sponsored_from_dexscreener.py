# market/management/commands/sync_sponsored_from_dexscreener.py

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import requests
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from market.models import SponsoredProject


DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{token_address}"


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            if math.isfinite(float(v)):
                return float(v)
            return None
        s = str(v).strip()
        if s == "":
            return None
        x = float(s)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            if math.isfinite(v):
                return int(v)
            return None
        s = str(v).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _pick_best_pair(pairs: list[dict]) -> Optional[dict]:
    """
    DexScreener는 token address 기준으로 여러 pair를 주는 경우가 많음.
    가장 신뢰도가 높은 것으로 보통 liquidity가 큰 pair를 선택.
    """
    if not pairs:
        return None

    def score(p: dict) -> float:
        liq = _safe_float(((p.get("liquidity") or {}).get("usd"))) or 0.0
        vol = _safe_float(((p.get("volume") or {}).get("h24"))) or 0.0
        txns = p.get("txns") or {}
        h24 = txns.get("h24") or {}
        buys = _safe_int(h24.get("buys")) or 0
        sells = _safe_int(h24.get("sells")) or 0
        # liquidity 최우선 + volume/txns 가산점
        return liq * 1_000_000 + vol * 10 + (buys + sells)

    pairs_sorted = sorted(pairs, key=score, reverse=True)
    return pairs_sorted[0]


def _extract_metrics(pair: dict) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[str]]:
    # mcap: DexScreener pair에는 marketCap이 있거나 없을 수 있음.
    # 없으면 fdv를 fallback으로 사용.
    mcap = _safe_float(pair.get("marketCap"))
    if mcap is None:
        mcap = _safe_float(pair.get("fdv"))

    vol24 = _safe_float(((pair.get("volume") or {}).get("h24")))
    txns = pair.get("txns") or {}
    h24 = txns.get("h24") or {}
    buys = _safe_int(h24.get("buys")) or 0
    sells = _safe_int(h24.get("sells")) or 0
    txns24 = buys + sells if (buys + sells) > 0 else None

    dex_url = pair.get("url") or pair.get("pairUrl")  # 케이스별 대응

    return mcap, vol24, txns24, dex_url


class Command(BaseCommand):
    help = "Sync SponsoredProject Dex metrics (mcap/volume/txns/dex_url) from DexScreener by token_address."

    def add_arguments(self, parser):
        parser.add_argument(
            "--only-active",
            action="store_true",
            help="Sync only active sponsored projects (is_active=True).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Limit number of projects to sync (0 = no limit).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch and show what would change, but do not write to DB.",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=10.0,
            help="HTTP timeout seconds (default 10).",
        )

    def handle(self, *args, **options):
        only_active: bool = options["only_active"]
        limit: int = options["limit"] or 0
        dry_run: bool = options["dry_run"]
        timeout: float = options["timeout"]

        qs = SponsoredProject.objects.all().order_by("order", "id")
        if only_active:
            qs = qs.filter(is_active=True)

        # token_address 없는 건 제외
        qs = qs.exclude(token_address__isnull=True).exclude(token_address__exact="")

        if limit > 0:
            qs = qs[:limit]

        projects = list(qs)
        if not projects:
            self.stdout.write(self.style.WARNING("No SponsoredProject to sync (token_address empty or none matched)."))
            return

        self.stdout.write(f"Sync target: {len(projects)} project(s) | only_active={only_active} | dry_run={dry_run}")

        session = requests.Session()
        now = timezone.now()

        updated = 0
        skipped = 0
        failed = 0

        for sp in projects:
            token = (sp.token_address or "").strip()
            url = DEXSCREENER_TOKEN_URL.format(token_address=token)

            try:
                r = session.get(url, timeout=timeout)
                if r.status_code != 200:
                    failed += 1
                    self.stdout.write(self.style.ERROR(f"[{sp.id}] {sp.name} ({token}) -> HTTP {r.status_code}"))
                    continue

                data: Dict[str, Any] = r.json() or {}
                pairs = data.get("pairs") or []
                best = _pick_best_pair(pairs)

                if not best:
                    skipped += 1
                    self.stdout.write(self.style.WARNING(f"[{sp.id}] {sp.name} ({token}) -> No pairs"))
                    continue

                mcap, vol24, txns24, dex_url = _extract_metrics(best)

                changes = {}
                if mcap is not None and sp.mcap_usd != mcap:
                    changes["mcap_usd"] = mcap
                if vol24 is not None and sp.volume_24h_usd != vol24:
                    changes["volume_24h_usd"] = vol24
                if txns24 is not None and sp.txns_24h != txns24:
                    changes["txns_24h"] = txns24
                if dex_url and sp.dex_url != dex_url:
                    changes["dex_url"] = dex_url

                # 변화가 있을 때만 업데이트 타임스탬프 갱신
                if changes:
                    changes["dex_updated_at"] = now

                    msg = f"[{sp.id}] {sp.name} -> " + ", ".join(f"{k}={v}" for k, v in changes.items() if k != "dex_updated_at")
                    self.stdout.write(self.style.SUCCESS(msg))

                    if not dry_run:
                        for k, v in changes.items():
                            setattr(sp, k, v)
                        with transaction.atomic():
                            sp.save(update_fields=list(changes.keys()))
                    updated += 1
                else:
                    skipped += 1
                    self.stdout.write(f"[{sp.id}] {sp.name} -> no change")

            except Exception as e:
                failed += 1
                self.stdout.write(self.style.ERROR(f"[{sp.id}] {sp.name} ({token}) -> ERROR: {e!r}"))

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Done. updated={updated}, skipped={skipped}, failed={failed}"))
