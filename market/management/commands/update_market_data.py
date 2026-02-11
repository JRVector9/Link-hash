import time
from django.core.management.base import BaseCommand
from market.models import TokenSnapshot
from market.services.dexscreener import (
    discover_tokens_solana,
    best_pair_snapshot,
    compute_filters,
)

class Command(BaseCommand):
    help = "Fetch Solana tokens from DexScreener and update majors/potential snapshots"

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=200)
        parser.add_argument("--sleep", type=float, default=0.25)

    def handle(self, *args, **opts):
        limit = opts["limit"]
        sleep_sec = opts["sleep"]

        self.stdout.write(self.style.SUCCESS(f"Discovering tokens... limit={limit}"))
        candidates = discover_tokens_solana(limit=limit)
        self.stdout.write(self.style.SUCCESS(f"Candidates: {len(candidates)}"))

        majors_rows = []
        potential_rows = []

        for i, addr in enumerate(candidates, 1):
            try:
                snap = best_pair_snapshot("solana", addr)
                if not snap:
                    continue
                snap = compute_filters(snap)

                mcap = snap.get("market_cap")
                lp = snap.get("liquidity_usd")
                vol = snap.get("volume_24h")

                # majors filter
                if (mcap is not None and mcap >= 1_000_000) and (lp is not None and lp >= 20_000) and (vol is not None and vol >= 10_000):
                    snap["category"] = "major"
                    majors_rows.append(snap)

                # potential filter
                hard = (mcap is not None and mcap < 1_000_000) and (lp is not None and lp >= 15_000)
                if hard and snap.get("potential_score_hits", 0) >= 2:
                    snap["category"] = "potential"
                    potential_rows.append(snap)

            except Exception:
                pass

            time.sleep(sleep_sec)

        # 정렬
        majors_rows.sort(key=lambda x: ((x.get("market_cap") or 0), (x.get("volume_24h") or 0)), reverse=True)
        potential_rows.sort(key=lambda x: ((x.get("potential_score_hits") or 0), (x.get("vol_mcap_ratio") or 0), (x.get("volume_24h") or 0)), reverse=True)

        # DB 갱신: 간단하게 category별로 싹 지우고 다시 넣기 (운영에선 upsert 권장)
        TokenSnapshot.objects.filter(category="major").delete()
        TokenSnapshot.objects.filter(category="potential").delete()

        TokenSnapshot.objects.bulk_create([TokenSnapshot(**r) for r in majors_rows[:60]])
        TokenSnapshot.objects.bulk_create([TokenSnapshot(**r) for r in potential_rows[:200]])

        self.stdout.write(self.style.SUCCESS(f"Saved majors={min(len(majors_rows),60)}, potential={min(len(potential_rows),200)}"))
