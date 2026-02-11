from decimal import Decimal
import logging

from celery import shared_task, current_app
from django.db import transaction
from django.utils import timezone

from market.models import GameSession, Participation, Winner, Game, HourlyCryptoPrediction
from market.engine import (
    _start_new_fomo_session,
    _broadcast_game_update,
    _broadcast_past_result,
    _schedule_fomo_settlement,
    _get_king_participation,
)

log = logging.getLogger(__name__)


# ============================================================
# ✅ (1) Hourly Crypto Predictions — runs at :00 every hour
# ============================================================
@shared_task
def update_hourly_predictions():
    """
    Fetch BTC/ETH/SOL predictions via OpenAI pipeline and store in DB.
    """
    from market.views import run_pipeline
    from datetime import datetime, timezone as dt_tz, timedelta

    assets = ["BTC", "ETH", "SOL"]
    now_utc = datetime.now(dt_tz.utc)
    hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
    next_hour = hour_start + timedelta(hours=1)

    for asset in assets:
        try:
            result = run_pipeline(
                symbol=f"{asset}/USDT",
                timeframe="1h",
                limit=300,
                market_type="swap",
                use_web_search=True,
                web_hours=24,
            )
            HourlyCryptoPrediction.objects.update_or_create(
                asset=asset,
                defaults={
                    "verdict": result.get("verdict", ""),
                    "reasoning": result.get("reasoning", ""),
                    "as_of_utc": hour_start,
                    "next_update_utc": next_hour,
                },
            )
            log.info("Updated prediction for %s: %s", asset, result.get("verdict"))
        except Exception as e:
            log.exception("Failed to update prediction for %s: %s", asset, e)


# ============================================================
# ✅ (2a) DexScreener sync — runs at :02 every hour
# ============================================================
@shared_task
def sync_sponsored_dex_metrics():
    """Wrap sync_sponsored_from_dexscreener management command."""
    from django.core.management import call_command
    try:
        call_command("sync_sponsored_from_dexscreener", "--only-active")
        log.info("sync_sponsored_dex_metrics completed.")
    except Exception as e:
        log.exception("sync_sponsored_dex_metrics failed: %s", e)


# ============================================================
# ✅ (2b) Market data update — runs at :05 every hour
# ============================================================
@shared_task
def update_market_snapshots():
    """Wrap update_market_data management command."""
    from django.core.management import call_command
    try:
        call_command("update_market_data")
        log.info("update_market_snapshots completed.")
    except Exception as e:
        log.exception("update_market_snapshots failed: %s", e)


# ============================================================
# ✅ FOMO Settlement task (updated to use _get_king_participation)
# ============================================================
@shared_task(bind=True)
def settle_fomo_session(self, session_id: int):
    """
    Runs around session.fomo_ends_at.

    ✅ Idempotent & concurrency-safe (row lock)
    ✅ Self-heals if executed early: re-schedules itself
    ✅ Broadcasts AFTER commit
    ✅ Uses _get_king_participation for limit-mode block_time ordering
    """
    result_payload = None

    game_id = None
    settled_session_id = None
    new_session_id = None
    had_winner = False
    winner_wallet = None
    payout_str = None

    with transaction.atomic():
        try:
            session = GameSession.objects.select_for_update().select_related("game").get(pk=session_id)
        except GameSession.DoesNotExist:
            return {"ok": False, "error": "session_not_found"}

        game = session.game
        game_id = game.id
        settled_session_id = session.id

        if game.category != Game.TYPE_FOMO:
            return {"ok": False, "error": "not_fomo"}

        if session.status != GameSession.STATUS_OPEN:
            return {"ok": True, "status": "already_not_open", "session_status": session.status}

        if not session.fomo_ends_at:
            return {"ok": False, "error": "missing_fomo_ends_at"}

        now = timezone.now()

        # Self-heal: if executed early, re-schedule and exit
        if now < session.fomo_ends_at:
            _schedule_fomo_settlement(session)
            return {
                "ok": True,
                "status": "executed_early_rescheduled",
                "now": now.isoformat(),
                "ends_at": session.fomo_ends_at.isoformat(),
                "session_id": session.id,
                "task_id": self.request.id,
            }

        # Lock session (state transition)
        session.status = GameSession.STATUS_LOCKED
        session.save(update_fields=["status"])

        # ✅ Use king helper (block_time ordering for limit mode)
        last_verified = _get_king_participation(game, session)

        # No winner path
        if not last_verified:
            session.status = GameSession.STATUS_SETTLED
            session.settled_at = now
            session.settle_task_id = ""
            session.save(update_fields=["status", "settled_at", "settle_task_id"])

            new_sess = _start_new_fomo_session(game)
            new_session_id = new_sess.id

            result_payload = {"ok": True, "status": "settled_no_winner", "new_session_id": new_sess.id}

        else:
            # Winner payout
            gross = session.pot_total_sol or Decimal("0")
            fee_pct = session.fee_pct_snapshot or Decimal("0")
            fee_amt = (gross * fee_pct / Decimal("100")).quantize(Decimal("0.000000001"))
            payout = (gross - fee_amt).quantize(Decimal("0.000000001"))
            if payout < 0:
                payout = Decimal("0")

            Winner.objects.create(
                session=session,
                user_profile=last_verified.user_profile,
                participation=last_verified,
                payout_amount_sol=payout,
                payout_status=Winner.PAYOUT_PENDING,
            )

            session.status = GameSession.STATUS_SETTLED
            session.settled_at = now
            session.settle_task_id = ""
            session.save(update_fields=["status", "settled_at", "settle_task_id"])

            had_winner = True
            winner_wallet = last_verified.user_profile.wallet_address
            payout_str = str(payout)

            new_sess = _start_new_fomo_session(game)
            new_session_id = new_sess.id

            result_payload = {
                "ok": True,
                "status": "settled_with_winner",
                "winner_wallet": winner_wallet,
                "payout": payout_str,
                "new_session_id": new_sess.id,
            }

        def _after_commit_broadcast():
            try:
                g = Game.objects.get(pk=game_id)
            except Exception as e:
                log.exception("AFTER_COMMIT: failed to refetch game_id=%s: %s", game_id, e)
                return

            if had_winner:
                try:
                    s = GameSession.objects.get(pk=settled_session_id)
                    _broadcast_past_result(g, s)
                    log.info("AFTER_COMMIT: past_result broadcast ok game=%s settled_session=%s", game_id, settled_session_id)
                except Exception as e:
                    log.exception(
                        "AFTER_COMMIT: past_result broadcast FAILED game=%s settled_session=%s: %s",
                        game_id, settled_session_id, e
                    )

            try:
                ns = GameSession.objects.get(pk=new_session_id)
                _broadcast_game_update(g, ns)
                log.info("AFTER_COMMIT: game_update broadcast ok game=%s new_session=%s", game_id, new_session_id)
            except Exception as e:
                log.exception(
                    "AFTER_COMMIT: game_update broadcast FAILED game=%s new_session=%s: %s",
                    game_id, new_session_id, e
                )

        transaction.on_commit(_after_commit_broadcast)

    return result_payload